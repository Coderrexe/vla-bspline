"""Archived-input quaternion-sign ablation. No robot or network interface.

q and -q encode the same orientation. Positive-w canonicalization can still
create a discontinuity in a component-normalized learned policy. Test choosing
the hemisphere closest to the recorded mean, without changing physical poses.
This diagnostic is not motion authorization and does not alter the live adapter.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation
import torch

from apollo_legacy_state import CheckpointStateAdapter, current_tcp_to_training_state
from apollo_predictor import ApolloPredictor
from apollo_parked_state import SessionParkedStateAdapter
from diagnose_apollo_constant_features import action_summary


def align_quaternions(state, mean):
    out = np.asarray(state, dtype=np.float32).copy()
    for offset in (12, 28):
        q = out[..., offset:offset+4]
        reference = mean[offset:offset+4]
        if np.linalg.norm(reference) < .5:
            raise ValueError('Training orientations are too dispersed for a mean-hemisphere assumption')
        flip = np.sum(q*reference, axis=-1) < 0
        out[..., offset:offset+4] = np.where(np.asarray(flip)[..., None], -q, q)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, nargs='+', required=True)
    p.add_argument('--snapshots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = {'purpose': 'OFFLINE_QUATERNION_SIGN_ABLATION', 'actions_published': 0,
              'runtime_api_writes': 0, 'live_adapter_changed': False, 'models': []}
    arrays = {}
    for checkpoint in args.models:
        adapter = CheckpointStateAdapter(checkpoint)
        stats = load_file(str(next(checkpoint.glob('policy_preprocessor_step_*_normalizer_processor.safetensors'))))
        mean, std = [stats[f'observation.state.{key}'] for key in ('mean', 'std')]
        model = ApolloPredictor(checkpoint, device='cuda:0')
        item = {'model': checkpoint.name, 'state_mean': mean.tolist(), 'snapshots': []}
        with np.load(checkpoint/'prediction_fixture.npz', allow_pickle=False) as fixture:
            state, view, grip = [fixture[k].copy() for k in ('state', 'view_wrist', 'grip_wrist')]
            aligned = align_quaternions(state, mean)
            item['fixture_input_identical'] = bool(np.array_equal(state, aligned))
            predictions = []
            for candidate in (state, aligned):
                model.reset()
                torch.manual_seed(int(fixture['seed']))
                predictions.append(model.predict_chunk(candidate, view, grip))
            item['fixture_prediction_identical'] = bool(np.array_equal(*predictions))
        for index, path in enumerate(args.snapshots):
            with np.load(path, allow_pickle=False) as f:
                current, view, grip = [f[k].copy() for k in ('current_state', 'view_rgb', 'grip_rgb')]
            original = current_tcp_to_training_state(current)
            original[adapter.mask] = adapter.reference[adapter.mask]
            aligned = align_quaternions(original, mean)
            # Replay an archived, held observation through the real guarded
            # adapter. Synthetic timestamps here are explicitly offline, not
            # a claim that an archived observation is fresh on the robot.
            guard = SessionParkedStateAdapter(adapter, operator_confirmed=True,
                                               align_to_training_hemisphere=True)
            for sample in range(11):
                assert guard.adapt(current, session_id='offline-replay', epoch='offline',
                                   t_mono=sample*.32) is None
            guarded = guard.adapt(current, session_id='offline-replay', epoch='offline', t_mono=3.52)
            np.testing.assert_array_equal(guarded, aligned)
            other = np.ones(32, dtype=bool)
            other[12:16] = False
            other[28:32] = False
            np.testing.assert_array_equal(original[other], aligned[other])
            for offset in (12, 28):
                qa, qb = [Rotation.from_quat(x[offset:offset+4][[1,2,3,0]]) for x in (original, aligned)]
                if (qa.inv()*qb).magnitude() > 1e-12:
                    raise ValueError('Hemisphere mapping changed orientation')
            result = {'snapshot': str(path), 'physical_orientation_preserved': True,
                      'guarded_input_matches_candidate_exactly': True, 'variants': []}
            for label, state in [('positive_w', original), ('training_hemisphere', aligned)]:
                v = {'variant': label, 'normalized_grip_quaternion':
                     ((state[12:16]-mean[12:16])/(std[12:16]+1e-8)).tolist(), 'predictions': []}
                for seed in (20260912, 20260913, 20260914):
                    model.reset()
                    torch.manual_seed(seed)
                    actions = model.predict_chunk(state, view, grip)
                    arrays[f'{checkpoint.name}_{index}_{label}_{seed}'] = actions
                    v['predictions'].append({'seed': seed, **action_summary(actions, current)})
                result['variants'].append(v)
            item['snapshots'].append(result)
        report['models'].append(item)
        print(json.dumps(item), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'predictions.npz', **arrays)


if __name__ == '__main__':
    main()
