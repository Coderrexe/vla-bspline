"""Bounded read-only Dora observation/prediction test.

Uses the observer placeholder, which has NO outputs. Never creates a session,
publishes an action, connects to a control box, or changes the live runtime.
"""
import argparse
import json
import os
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from apollo_legacy_state import CheckpointStateAdapter, arm_stream_to_current_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--predictions', type=int, default=5)
    p.add_argument('--diagnose-constant-features', action='store_true',
                   help='Read-only ablation of zero-variance inputs; never authorizes execution')
    args = p.parse_args()
    if not 0 < args.seconds <= 120 or not 1 <= args.predictions <= 30:
        raise ValueError('Probe must be bounded')
    args.output.mkdir(parents=True, exist_ok=False)
    from apollo_predictor import ApolloPredictor
    import torch
    torch.set_num_threads(4)
    model = ApolloPredictor(args.checkpoint, device=args.device)
    state_adapter = CheckpointStateAdapter(args.checkpoint)
    with np.load(args.checkpoint/'prediction_fixture.npz') as fixture:
        for _ in range(3):
            model.predict_chunk(fixture['state'], fixture['view_wrist'], fixture['grip_wrist'])
    model.reset()
    torch.manual_seed(20260912)
    with urllib.request.urlopen(args.url+'/api/dora', timeout=5) as response:
        info = json.load(response)
    if info['state'] != 'attached' or 'observer' not in info['placeholders']:
        raise RuntimeError('No existing observer placeholder; do not modify the dataflow')
    os.environ['DORA_ZENOH_CONNECT'] = info['zenoh_connect']
    os.environ['DORA_ZENOH_LISTEN'] = 'tcp/127.0.0.1:0'
    os.environ['DORA_ZENOH_MULTICAST'] = 'off'
    pa.array([0.0]).to_numpy()  # warm Arrow interop before attaching
    from dora import Node
    node = Node('observer', daemon_port=info['daemon_port'])
    frames, states, counts, rows = {}, None, Counter(), []
    report = {'mode': 'READ_ONLY_OBSERVER', 'action_messages_sent': 0,
              'runtime_api_writes': 0, 'checkpoint': str(args.checkpoint),
              'state_pose_convention': state_adapter.pose_convention,
              'dataflow_id': info['dataflow_id'], 'predictions': rows,
              'constant_feature_diagnostic': args.diagnose_constant_features}
    start = time.monotonic()
    next_predict = start
    try:
        while time.monotonic()-start < args.seconds and len(rows) < args.predictions:
            event = node.next(timeout=.2)
            if event is None or event.get('type') in ('STOP', 'INPUT_CLOSED'):
                break
            if event.get('type') != 'INPUT':
                continue
            topic, meta = event['id'], dict(event.get('metadata') or {})
            counts[topic] += 1
            if meta.get('mavis_schema') != 1:
                continue
            if topic in ('cam_grip_wrist', 'cam_view_wrist'):
                if meta.get('encoding') != 'rgb8' or (meta.get('height'), meta.get('width')) != (480, 640):
                    raise ValueError('Real RGB camera format differs from training')
                image = event['value'].to_numpy().reshape(480, 640, 3)
                if image.dtype != np.uint8:
                    raise ValueError('Camera is not uint8 RGB')
                frames[topic[4:]] = (image.copy(), float(meta['frame_t_mono']), int(meta['frame_seq']))
            elif topic == 'arm_state':
                try:
                    state = arm_stream_to_current_state(event['value'].to_numpy(), meta)
                except ValueError as exc:
                    counts['invalid_arm_state'] += 1
                    report['last_state_refusal'] = str(exc)
                    continue
                states = (state, float(meta['t_mono']), meta)
            if len(frames) != 2 or states is None or time.monotonic() < next_predict:
                continue
            now = time.monotonic()
            ages = {k: now-v[1] for k, v in frames.items()}
            if not -.02 <= now-states[1] <= .2 or any(not -.02 <= age <= .2 for age in ages.values()):
                counts['stale_snapshot_skipped'] += 1
                continue
            current, stamp, state_meta = states
            # Retain a read-only snapshot even when compatibility refuses inference;
            # it lets the operator inspect the setup without changing the robot.
            if not (args.output/'live_state.npz').exists():
                np.savez_compressed(args.output/'live_state.npz', state_current=current)
                for camera, (image, _, _) in frames.items():
                    Image.fromarray(image).save(args.output/f'{camera}.png')
            try:
                legacy = state_adapter(current)
            except ValueError as exc:
                from apollo_legacy_state import STATE_NAMES
                # Refusal diagnostics must use the checkpoint's convention too:
                # lamp uses corrected TCP, unlike the older drawer recordings.
                converted = state_adapter.to_training_state(current)
                report['refusal'] = str(exc)
                report['constant_feature_differences'] = [
                    {'name': STATE_NAMES[i], 'current': float(converted[i]),
                     'training': float(state_adapter.reference[i]),
                     'difference': float(converted[i]-state_adapter.reference[i])}
                    for i in np.flatnonzero(state_adapter.mask)
                    if abs(converted[i]-state_adapter.reference[i]) > state_adapter.tolerance
                ]
                if not args.diagnose_constant_features:
                    report['status'] = 'INPUT_COMPATIBILITY_REFUSED'
                    break
                # DIAGNOSTIC ONLY: this observer has no declared Dora outputs.
                # The guarded motion-capable adapter still refuses these inputs.
                legacy = converted.copy()
                legacy[state_adapter.mask] = state_adapter.reference[state_adapter.mask]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            actions = model.predict_chunk(legacy, frames['view_wrist'][0], frames['grip_wrist'][0])
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-t0
            row = {'index': len(rows), 'latency_ms': elapsed*1000,
                   'state_age_at_prediction_s': now-stamp, 'image_ages_s': ages,
                   'image_seq': {k: v[2] for k, v in frames.items()},
                   'arm_state_source': state_meta.get('source'),
                   'measured_gripper_open_fraction': float(current[7]),
                   'model_input_gripper_open_fraction': float(legacy[7]),
                   'grip_rail_m': float(current[8]), 'view_rail_m': float(current[24]),
                   'action_shape': list(actions.shape), 'finite': bool(np.isfinite(actions).all()),
                   'first_action_grip': actions[0, :8].tolist(),
                   'translation_norm_max_m': float(np.linalg.norm(actions[:, :3], axis=1).max()),
                   'translation_path_sum_m': float(np.linalg.norm(actions[:, :3], axis=1).sum()),
                   'rotation_path_sum_rad': float(np.linalg.norm(actions[:, 3:6], axis=1).sum()),
                   'gripper_change_max': float(np.max(abs(actions[:,6]-current[7])))}
            if not row['finite'] or actions.shape != (8, 16):
                raise RuntimeError('Invalid prediction')
            if not rows:
                for camera, (image, _, _) in frames.items():
                    Image.fromarray(image).save(args.output/f'{camera}.png')
                np.savez_compressed(args.output/'live_snapshot.npz', state_current=current,
                                    state_training=legacy, predicted=actions,
                                    view_wrist=frames['view_wrist'][0], grip_wrist=frames['grip_wrist'][0])
            rows.append(row)
            print(json.dumps(row), flush=True)
            next_predict = time.monotonic()+.4
        completed = ('CONSTANT_DIAGNOSTIC_COMPLETE_NOT_EXECUTION_READY'
                     if args.diagnose_constant_features else 'PASS')
        report.setdefault('status', completed if len(rows) == args.predictions else 'INCOMPLETE')
    finally:
        report['received'] = dict(counts)
        report['elapsed_s'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        del node
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}), flush=True)
    if report['status'] not in ('PASS', 'CONSTANT_DIAGNOSTIC_COMPLETE_NOT_EXECUTION_READY'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
