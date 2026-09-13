"""Offline constant-input ablation. No Dora, HTTP, SDK, or robot connection.

Tests projecting exactly zero-variance state features to their training values,
using archived inputs only. Does not modify the live adapter, its guard, weights,
or normalizers. A finite prediction is NOT permission to execute an action.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from apollo_legacy_state import CheckpointStateAdapter, STATE_NAMES, current_tcp_to_training_state
from apollo_predictor import ApolloPredictor


def project_constants(state, adapter):
    """Offline candidate, deliberately separate from the guarded live adapter."""
    out = np.asarray(state, dtype=np.float32).copy()
    if out.shape != (32,) or not np.isfinite(out).all():
        raise ValueError('Expected finite archived state')
    out[adapter.mask] = adapter.reference[adapter.mask]
    return out


def action_summary(actions, state):
    return {
        'finite': bool(np.isfinite(actions).all()),
        'first_row': actions[0, :8].tolist(),
        'first_translation_norm_mm': float(np.linalg.norm(actions[0, :3])*1000),
        'first_rotation_norm_rad': float(np.linalg.norm(actions[0, 3:6])),
        'first_gripper_change': float(abs(actions[0, 6]-state[7])),
        'eight_row_translation_path_mm': float(np.linalg.norm(actions[:, :3], axis=1).sum()*1000),
        'eight_row_net_translation_mm': (actions[:, :3].sum(0)*1000).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, nargs='+', required=True)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    snapshots = {}
    for name, path in [('before', args.before), ('after', args.after)]:
        with np.load(path/'live_state.npz', allow_pickle=False) as f:
            current = f['state_current'].copy()
        images = [np.asarray(Image.open(path/f'{camera}.png').convert('RGB'))
                  for camera in ('view_wrist', 'grip_wrist')]
        snapshots[name] = (current_tcp_to_training_state(current), *images)
    report = {'status': 'OFFLINE_ONLY_NOT_MOTION_AUTHORIZATION',
              'robot_connected': False, 'action_messages_sent': 0,
              'live_adapter_changed': False, 'models': []}
    arrays = {}
    for checkpoint in args.models:
        model = ApolloPredictor(checkpoint, device='cuda:0')
        adapter = CheckpointStateAdapter(checkpoint)
        with (checkpoint/'model.safetensors').open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        with np.load(checkpoint/'prediction_fixture.npz', allow_pickle=False) as f:
            fixture = [f[k].copy() for k in ('state', 'view_wrist', 'grip_wrist')]
            fixture_seed = int(f['seed'])
        projected_fixture = project_constants(fixture[0], adapter)
        row = {'model': checkpoint.name, 'weight_sha256': digest,
               'constant_fields': [STATE_NAMES[i] for i in np.flatnonzero(adapter.mask)],
               'fixture_input_identical': bool(np.array_equal(fixture[0], projected_fixture)),
               'snapshots': []}
        if not row['fixture_input_identical']:
            raise ValueError('Projection changes an archived training-interface fixture')
        model.reset()
        torch.manual_seed(fixture_seed)
        original = model.predict_chunk(*fixture)
        model.reset()
        torch.manual_seed(fixture_seed)
        projected = model.predict_chunk(projected_fixture, *fixture[1:])
        row['fixture_prediction_identical'] = bool(np.array_equal(original, projected))
        if not row['fixture_prediction_identical']:
            raise ValueError('Repeated fixture predictions are not identical')
        for name, (state, view, grip) in snapshots.items():
            projected = project_constants(state, adapter)
            item = {'name': name,
                    'varying_features_unchanged': bool(np.array_equal(state[~adapter.mask], projected[~adapter.mask])),
                    'max_constant_feature_difference_native': float(np.max(abs(state[adapter.mask]-projected[adapter.mask]))),
                    'variants': []}
            for variant, candidate in [('unprojected', state), ('projected', projected)]:
                normalized = model.prepare(candidate, view, grip)['observation.state'].detach().float().cpu().numpy()
                v = {'name': variant,
                     'max_abs_normalized_constant': float(np.max(abs(normalized[..., adapter.mask]))),
                     'predictions': []}
                for seed in (20260912, 20260913, 20260914):
                    model.reset()
                    torch.manual_seed(seed)
                    actions = model.predict_chunk(candidate, view, grip)
                    arrays[f'{checkpoint.name}_{name}_{variant}_{seed}'] = actions
                    v['predictions'].append({'seed': seed, **action_summary(actions, candidate)})
                item['variants'].append(v)
            row['snapshots'].append(item)
        report['models'].append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'predictions.npz', **arrays)


if __name__ == '__main__':
    main()
