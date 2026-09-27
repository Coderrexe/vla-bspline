"""Compare an archived live pose with demonstrations; never connect to a robot.

Distances are descriptive data-support measurements, not commands, localization,
collision clearance, or execution authorization. Original inputs are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from apollo_legacy_state import training_to_current_tcp_state


def rotation(state, offset=0):
    return Rotation.from_quat(state[offset+12:offset+16][[1, 2, 3, 0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--trials', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.snapshot, allow_pickle=False) as saved:
        live = saved['current_state'].copy()
        proposed = saved['proposed_actions'].copy() if 'proposed_actions' in saved else None
    if live.shape != (32,) or not np.isfinite(live).all():
        raise ValueError('Expected a finite archived native state')
    nearest, starts, first_close, bounds_low, bounds_high, sources = [], [], [], [], [], {}
    for path in sorted((args.dataset/'episodes').glob('*/frames.parquet')):
        table = pq.read_table(path, columns=['observation.state', 'action', 'timestamp'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if (state.ndim != 2 or state.shape[1] != 32 or not len(state)
                or action.shape != (len(state), 16) or not np.isfinite(action).all()):
            raise ValueError(f'Invalid recorded state/action arrays: {path}')
        current = training_to_current_tcp_state(state)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        starts.append(current[0])
        bounds_low.append(current[:, 9:12].min(0))
        bounds_high.append(current[:, 9:12].max(0))
        distance = np.linalg.norm(current[:, 9:12]-live[9:12], axis=1)
        j = int(distance.argmin())
        nearest.append({
            'episode': path.parent.name, 'frame': j,
            'timestamp_s': float(table['timestamp'][j].as_py()),
            'distance_mm': float(distance[j]*1000),
            'orientation_difference_deg': float(np.degrees(
                (rotation(current[j])*rotation(live).inv()).magnitude())),
            'measured_gripper_mm': float(current[j, 7]*84),
            'next8_translation_sum_mm': (action[j:j+8, :3].sum(0)*1000).tolist(),
            'next8_gripper_targets_mm': (action[j:j+8, 6]*84).tolist(),
        })
        candidates = np.flatnonzero(action[:, 6] < .5)
        if len(candidates):
            first_close.append(current[candidates[0]])
    if not starts:
        raise ValueError('No episodes found')
    starts = np.stack(starts)
    reference = starts[0]
    report = {
        'purpose': 'ARCHIVED_POSE_SUPPORT_NOT_COMMANDS_OR_SAFETY_LIMITS',
        'actions_published': 0, 'runtime_api_writes': 0,
        'snapshot': str(args.snapshot),
        'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        'episode_sources_sha256': sources, 'episodes': len(starts),
        'current_state': live.tolist(),
        'recorded_start_reference': reference.tolist(),
        'start_view_state_constant_across_episodes': bool(np.array_equal(
            starts[:, 16:], np.broadcast_to(reference[16:], starts[:, 16:].shape))),
        'view_translation_difference_mm': ((live[25:28]-reference[25:28])*1000).tolist(),
        'view_orientation_difference_deg': float(np.degrees(
            (rotation(live, 16)*rotation(reference, 16).inv()).magnitude())),
        'recorded_tcp_min_m': np.min(bounds_low, axis=0).tolist(),
        'recorded_tcp_max_m': np.max(bounds_high, axis=0).tolist(),
        'nearest_10': sorted(nearest, key=lambda x: x['distance_mm'])[:10],
    }
    if first_close:
        positions = np.stack(first_close)[:, 9:12]
        report['first_target_below_half_open'] = {
            'episodes': len(positions), 'median_tcp_m': np.median(positions, axis=0).tolist(),
            'median_relative_to_live_mm': ((np.median(positions, axis=0)-live[9:12])*1000).tolist(),
            'minimum_distance_to_live_mm': float(np.linalg.norm(positions-live[9:12], axis=1).min()*1000),
        }
    if proposed is not None:
        report['refused_proposal'] = {
            'translation_sum_mm': (proposed[:, :3].sum(0)*1000).tolist(),
            'gripper_targets_mm': (proposed[:, 6]*84).tolist(),
            'prefix_positions_m': (live[9:12]+np.cumsum(proposed[:, :3], axis=0)).tolist(),
        }
    if args.trials:
        history = []
        for path in sorted(args.trials.glob('drawer_trial_*/session_check.json')):
            trial = json.loads(path.read_text())
            arms = trial.get('last', {}).get('arms', [])
            arm = next((a for a in arms if a.get('arm_id') == 'view'), None)
            if not arm or not arm.get('ee_pose'):
                continue
            history.append({'trial': path.parent.name,
                'last_view_position_m': arm['ee_pose']['position'],
                'last_view_joints_rad': arm['q']})
        report['view_history'] = history
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('episodes', 'view_translation_difference_mm',
        'view_orientation_difference_deg', 'first_target_below_half_open') if k in report}, indent=2))


if __name__ == '__main__':
    main()
