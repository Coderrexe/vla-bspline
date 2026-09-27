"""Audit saved delta/absolute commands and package one episode OFFLINE.

No robot, Dora, SDK, HTTP, or session imports. The output is data, not an
execution grant. It does not establish collision clearance or replay success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotations_from_columns(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError('Expected finite rot6d rows')
    first, second = values[:, :3], values[:, 3:]
    norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate first rotation column')
    first = first/norm
    second = second-(first*second).sum(1, keepdims=True)*first
    norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate second rotation column')
    second = second/norm
    return Rotation.from_matrix(np.stack([first, second, np.cross(first, second)], axis=2))


def metrics(values, scale=1.):
    values = np.asarray(values, dtype=np.float64)*scale
    return {'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': float(np.max(values))}


def compare_commands(state, delta, absolute):
    state, delta, absolute = [np.asarray(a, dtype=np.float64) for a in (state, delta, absolute)]
    n = len(state)
    if n < 2 or state.shape != (n, 32) or delta.shape != (n, 16) or absolute.shape != (n, 22):
        raise ValueError('Unexpected two-arm recorded layout')
    if not all(np.isfinite(a).all() for a in (state, delta, absolute)):
        raise ValueError('Nonfinite recording')
    report = {}
    for arm, s, d, a in [('grip', 0, 0, 0), ('view', 16, 8, 11)]:
        target_r = rotations_from_columns(absolute[:, a+3:a+9])
        measured_r = Rotation.from_quat(state[:, s+12:s+16][:, [1, 2, 3, 0]])
        # Row k targets commanded pose k+1. Consecutive absolute targets
        # therefore differ by delta[k], NOT delta[k-1]. Ignore row 0 here:
        # measured initial pose need not equal the recorder's commanded anchor.
        translation_error = np.linalg.norm(np.diff(absolute[:, a:a+3], axis=0)-delta[1:, d:d+3], axis=1)
        composed = Rotation.from_rotvec(delta[1:, d+3:d+6])*target_r[:-1]
        rotation_error = (composed.inv()*target_r[1:]).magnitude()
        rail_error = np.abs(np.diff(absolute[:, a+10])-delta[1:, d+7])
        grip_error = np.abs(absolute[:, a+9]-delta[:, d+6])
        if translation_error.max() > 2e-5 or rotation_error.max() > 2e-4 or rail_error.max() > 2e-5 or grip_error.max() > 1e-6:
            raise ValueError(f'{arm}: absolute commands do not match spatial delta integration')
        first_expected = state[0, s+9:s+12]+delta[0, d:d+3]
        first_r = Rotation.from_rotvec(delta[0, d+3:d+6])*measured_r[0]
        # Dense command path is not the measured trajectory. Keep tracking
        # residuals and command consistency separate, including terminal row.
        tracking = np.linalg.norm(absolute[:-1, a:a+3]-state[1:, s+9:s+12], axis=1)
        report[arm] = {
            'command_translation_residual_mm': metrics(translation_error, 1000),
            'command_rotation_residual_rad': metrics(rotation_error),
            'command_rail_residual_mm': metrics(rail_error, 1000),
            'command_gripper_residual_fraction': metrics(grip_error),
            'initial_measured_vs_command_anchor_mm': float(np.linalg.norm(absolute[0, a:a+3]-first_expected)*1000),
            'initial_measured_vs_command_anchor_rad': float((first_r.inv()*target_r[0]).magnitude()),
            'command_to_next_retained_measured_position_mm': metrics(tracking, 1000),
            'total_delta_translation_m': delta[:, d:d+3].sum(0).tolist(),
            'translation_path_m': float(np.linalg.norm(delta[:, d:d+3], axis=1).sum()),
            'max_row_translation_mm': float(np.linalg.norm(delta[:, d:d+3], axis=1).max()*1000),
            'command_tcp_min_m': absolute[:, a:a+3].min(0).tolist(),
            'command_tcp_max_m': absolute[:, a:a+3].max(0).tolist(),
        }
    return report


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--pose-audit', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    info = manifest['features']['action.abs_ee']['info']
    if (info.get('label') != 'commanded_tcp_at_next_frame'
            or info.get('rotation') != 'rot6d_first_two_columns'):
        raise ValueError('Unknown absolute command convention')
    pose_audit = json.loads(args.pose_audit.read_text())
    if pose_audit.get('contract') != 'corrected_tcp':
        raise ValueError('Replay packet requires the corrected TCP pose audit')
    pose_entries = {x['episode']: x for x in pose_audit['episodes']}
    provenance = json.loads(args.provenance.read_text())
    mapping = provenance['episode_mapping']
    # Prefer a native absolute recording: a backfilled absolute track is
    # algebraically derived from delta and adds no independent label check.
    selected = next(x['source_episode_id'] for x in mapping
                    if x['split'] == 'train' and not json.loads(
                        (args.dataset/'episodes'/x['source_episode_id']/'episode.json').read_text()
                    ).get('backfill', {}).get('abs_ee'))
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for entry in mapping:
        directory = args.dataset/'episodes'/entry['source_episode_id']
        parquet = directory/'frames.parquet'
        digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
        if digest != entry['source_parquet_sha256']:
            raise ValueError('Recording hash differs from training provenance')
        audited = pose_entries.get(entry['source_episode_id'], {})
        if audited.get('parquet_sha256') != digest or audited.get('classification') != 'corrected_tcp':
            raise ValueError('Recording does not match corrected TCP pose audit')
        table = pq.read_table(parquet)
        state, delta, absolute = [np.asarray(table[key].to_pylist()) for key in
                                  ('observation.state', 'action', 'action.abs_ee')]
        timestamp = np.asarray(table['timestamp'].to_pylist())
        wallclock = np.asarray(table['wallclock_ns'].to_pylist(), dtype=np.int64)
        result = compare_commands(state, delta, absolute)
        episode = json.loads((directory/'episode.json').read_text())
        closes = np.flatnonzero((delta[1:, 6] < .5) & (delta[:-1, 6] >= .5))+1
        opens = np.flatnonzero((delta[1:, 6] >= .5) & (delta[:-1, 6] < .5))+1
        item = {'episode': entry['source_episode_id'], 'split': entry['split'],
                'parquet_sha256': digest, 'frames': len(state),
                'backfilled': bool(episode.get('backfill', {}).get('abs_ee')),
                'timestamp_step_s': metrics(np.diff(timestamp)),
                'wallclock_step_s': metrics(np.diff(wallclock)/1e9),
                'dense_duration_s': len(state)/manifest['fps'],
                'wallclock_span_s': float((wallclock[-1]-wallclock[0])/1e9),
                'first_closing_frame': int(closes[0]) if len(closes) else None,
                'release_frames': opens.tolist(), **result}
        episodes.append(item)
        if entry['source_episode_id'] == selected:
            np.savez_compressed(args.output/'recorded_episode.npz', state=state,
                                delta_commands=delta, absolute_commands=absolute,
                                recorded_timestamp=timestamp, recorded_wallclock_ns=wallclock)
    report = {'purpose': 'OFFLINE_RECORDED_COMMAND_CONSISTENCY_NOT_ROBOT_REPLAY',
              'runtime_api_calls': 0, 'actions_published': 0, 'live_authorization_created': False,
              'selected_episode': selected, 'selection': 'first natively recorded, non-backfilled training episode in frozen provenance',
              'dataset_manifest_sha256': hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
              'pose_audit_sha256': hashlib.sha256(args.pose_audit.read_bytes()).hexdigest(),
              'pose_audit_contract': pose_audit['contract'],
              'episodes': episodes,
              'interpretation': 'Delta/absolute label agreement only. Backfilled absolute labels are derived, not independent evidence. Match actual initial pose and timing; validate physical clearance on site before replay.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'episodes': len(episodes), 'selected_episode': selected,
                      'command_consistency': 'PASS', 'physical_replay': 'NOT_RUN'}))


if __name__ == '__main__':
    main()
