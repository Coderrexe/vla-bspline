"""Select lamp checkpoints by a fixed held-out command-error rule, not lab trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score_predictions(prediction, target, opening):
    """Equal weight to gripper and cumulative xyz error relative to hold baselines."""
    prediction, target, opening = map(np.asarray, (prediction, target, opening))
    if (prediction.ndim != 4 or prediction.shape[1:] != (3, 8, 16)
            or target.shape != (len(prediction), 8, 16)
            or opening.shape != (len(prediction),)
            or not all(np.isfinite(x).all() for x in (prediction, target, opening))):
        raise ValueError('Invalid prediction comparison inputs')
    grip_baseline = float(np.abs(opening[:, None]-target[:, :, 6]).mean())
    pose_baseline = float(np.sqrt(np.mean(target[:, :, :3].sum(1)**2)))
    if grip_baseline <= 1e-8 or pose_baseline <= 1e-8:
        raise ValueError('Degenerate hold reference')
    grip = float(np.abs(prediction[:, :, :, 6]-target[:, None, :, 6]).mean())
    pose = float(np.sqrt(np.mean((prediction[:, :, :, :3].sum(2)-target[:, None, :, :3].sum(2))**2)))
    return {'score': .5*(grip/grip_baseline+pose/pose_baseline),
            'gripper_mae_mm': grip*84, 'hold_gripper_mae_mm': grip_baseline*84,
            'eight_step_xyz_component_rmse_mm': pose*1000,
            'zero_xyz_component_rmse_mm': pose_baseline*1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs_manifest = json.loads((args.inputs/'manifest.json').read_text())
    with np.load(args.inputs/'inputs.npz', allow_pickle=False) as archive:
        target = archive['targets']; state = archive['states']
    samples = inputs_manifest['samples']
    held = np.array([s['split'] == 'validation' for s in samples])
    if held.sum() != 60 or len({s['episode'] for s in samples if s['split'] == 'validation'}) != 5:
        raise ValueError('Unexpected held-out selection')
    candidates = {}
    expected_runs = ('lamp_assembling_spline_native_tcp_v1_2466234',
                     'lamp_assembling_waypoint_native_tcp_v1_2466235')
    for run in expected_runs:
        candidates[run] = []
        for step in (10000, 20000, 30000, 40000):
            folder = args.reports/f'{run}_{step:06d}'
            report = json.loads((folder/'report.json').read_text())
            if len(report['samples']) != len(samples) or any(
                    any(actual[k] != expected[k] for k in ('episode', 'split', 'phase', 'frame', 'offset', 'original_parquet_sha256'))
                    for actual, expected in zip(report['samples'], samples)):
                raise ValueError('Checkpoint report observations differ')
            checkpoint = Path(report['checkpoint'])
            training = json.loads((checkpoint/'hardware_training_manifest.json').read_text())
            with (checkpoint/'model.safetensors').open('rb') as stream:
                actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if training['steps'] != step or actual_hash != report['checkpoint_sha256']:
                raise ValueError('Checkpoint provenance mismatch')
            with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
                if not np.array_equal(archive['target'], target):
                    raise ValueError('Checkpoint targets differ')
                result = score_predictions(archive['predictions'][held], target[held], state[held, 7])
            candidates[run].append({'step': step, 'checkpoint_sha256': actual_hash, **result})
    selected = {run: min(rows, key=lambda x: (x['score'], x['step']))['step'] for run, rows in candidates.items()}
    result = {'purpose': 'OFFLINE_VALIDATION_SELECTION_NOT_EXECUTION_AUTHORIZATION',
              'rule': 'Minimize mean of held-out gripper MAE / hold-gripper MAE and eight-step xyz component RMSE / zero-motion RMSE. Same six offsets at closure/release; ties select earlier checkpoint.',
              'independent_validation_episodes': 5, 'observations': 60,
              'inference_noise_seeds': inputs_manifest['seeds'], 'candidates': candidates,
              'selected_steps': selected, 'physical_robot_success': 'NOT_EVALUATED'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
