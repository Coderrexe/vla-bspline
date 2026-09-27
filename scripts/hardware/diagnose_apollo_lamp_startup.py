"""Archived-input sensitivity test. No robot, network, or action publication.

Crossed state/image inputs below are diagnostic counterfactuals, never inputs
authorized for physical execution. Each case uses identical inference seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--idle-snapshot', type=Path, required=True)
    p.add_argument('--enabled-snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from apollo_predictor import ApolloPredictor
    from apollo_legacy_state import CheckpointStateAdapter, align_quaternion_hemisphere
    torch.set_num_threads(4)
    model = ApolloPredictor(args.checkpoint, device='cuda:0')
    adapter = CheckpointStateAdapter(args.checkpoint)
    with np.load(args.idle_snapshot) as f:
        idle = {k: f[k].copy() for k in f.files}
    with np.load(args.enabled_snapshot) as f:
        enabled = {k: f[k].copy() for k in f.files}
    states = {}
    for key, raw in [('idle', idle['state_current']), ('enabled', enabled['current_state'])]:
        state = adapter.to_training_state(raw)
        state[adapter.mask] = adapter.reference[adapter.mask]
        states[key] = align_quaternion_hemisphere(state, adapter.mean)
    images = {'idle': (idle['view_wrist'], idle['grip_wrist']),
              'enabled': (enabled['view_rgb'], enabled['grip_rgb'])}
    results, arrays = [], {}
    for seed in (20260912, 20260913, 20260914):
        for state_key, image_key in [('idle', 'idle'), ('enabled', 'enabled'),
                                     ('enabled', 'idle'), ('idle', 'enabled')]:
            model.reset()
            torch.manual_seed(seed)
            action = model.predict_chunk(states[state_key], *images[image_key])
            label = f'{state_key}_state_{image_key}_images_seed{seed}'
            arrays[label] = action
            results.append({'case': label, 'state': state_key, 'images': image_key,
                            'diagnostic_crossed_input': state_key != image_key,
                            'seed': seed,
                            'net_xyz_mm': (action[:, :3].sum(0)*1000).tolist(),
                            'xyz_path_mm': float(np.linalg.norm(action[:, :3], axis=1).sum()*1000),
                            'rotation_path_rad': float(np.linalg.norm(action[:, 3:6], axis=1).sum()),
                            'opening_fraction_first_last': action[[0, -1], 6].tolist()})
    report = {'mode': 'ARCHIVED_INPUT_DIAGNOSTIC_NO_EXECUTION',
              'checkpoint': str(args.checkpoint), 'action_messages_sent': 0,
              'runtime_api_writes': 0,
              'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in (args.idle_snapshot, args.enabled_snapshot,
                                    args.checkpoint/'model.safetensors')},
              'state_delta_xyz_mm': ((states['enabled'][9:12]-states['idle'][9:12])*1000).tolist(),
              'results': results}
    np.savez_compressed(args.output/'predictions.npz', **arrays)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
