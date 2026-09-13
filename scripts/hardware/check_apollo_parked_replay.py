"""Replay saved hold telemetry through the parked-input guard. OFFLINE ONLY.

No hardware/network imports, session changes, or action publication. The test's
operator_confirmed flag exercises code; it is NOT approval for any live setup.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from apollo_legacy_state import CheckpointStateAdapter, current_tcp_to_training_state
from apollo_parked_state import SessionParkedStateAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reports-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = {'purpose': 'OFFLINE_PARKED_GUARD_REPLAY', 'actions_published': 0,
              'runtime_api_writes': 0, 'live_authorization_created': False, 'checks': []}
    cases = [('002', 'drawer_session_shadow_006', 'diagnostic_first_input.npz'),
             ('003', 'drawer_session_shadow_007', 'diagnostic_first_input.npz'),
             ('004', 'drawer_session_calibrated_008', 'parked_calibration_start.npz')]
    for session_num, client_folder, snapshot in cases:
        checkpoint = CheckpointStateAdapter(args.checkpoint)
        adapter = SessionParkedStateAdapter(checkpoint, operator_confirmed=True)
        rows = [json.loads(line) for line in
                (args.reports_root/f'drawer_hardware_session_{session_num}'/'telemetry.jsonl').open()]
        rows = [row for row in rows if row['session']['state'] == 'running']
        with np.load(args.reports_root/client_folder/snapshot,
                     allow_pickle=False) as f:
            live_input = f['current_state'].copy()
            previous_projection = (f['projected_training_state'].copy()
                                   if 'projected_training_state' in f else None)
        first_t = rows[0]['ts']
        latest = float('-inf')
        projected_samples = 0
        ready_at = None
        max_parked_deviation = 0.
        first_telemetry_delta = None
        for row in rows:
            if row['ts']-latest < .32:
                continue
            latest = row['ts']
            values = []
            for arm_id in ('grip', 'view'):
                arm = next(a for a in row['arms'] if a['arm_id'] == arm_id)
                if arm['error_code'] or arm['stale'] or arm['recovering'] or not arm['connected']:
                    raise ValueError('Unhealthy saved telemetry')
                values.extend(arm['q']+[arm['gripper_open_frac'], arm['rail_pos_m']]
                              +arm['ee_pose']['position']+arm['ee_pose']['orientation'])
            state = np.asarray(values, dtype=np.float32)
            if first_telemetry_delta is None:
                first_telemetry_delta = float(np.abs(state-live_input).max())
                # Verify telemetry layout against the actual Dora input snapshot.
                if first_telemetry_delta > 1e-6:
                    raise ValueError('Telemetry does not reproduce the saved Dora observation')
            projected = adapter.adapt(state, session_id=row['session']['session_id'],
                                      epoch=row['epoch'], t_mono=row['ts'])
            if ready_at is None and adapter.ready:
                ready_at = row['ts']-first_t
            if projected is not None:
                expected = current_tcp_to_training_state(state)
                np.testing.assert_array_equal(projected[~adapter.mask], expected[~adapter.mask])
                np.testing.assert_array_equal(projected[adapter.mask], adapter.model_reference[adapter.mask])
                max_parked_deviation = max(max_parked_deviation,
                                          float(np.abs(state-adapter.reference)[adapter.mask].max()))
                projected_samples += 1
        if not adapter.ready or not projected_samples:
            raise ValueError('Saved hold did not establish a usable reference')
        # Algebraic equivalence on the saved RGB-associated input, without
        # pretending that this archival input is a fresh live observation.
        projection = current_tcp_to_training_state(live_input)
        projection[adapter.mask] = adapter.model_reference[adapter.mask]
        if previous_projection is not None:
            np.testing.assert_array_equal(projection, previous_projection)
        report['checks'].append({
            'session_number': session_num, 'telemetry_rows': len(rows),
            'calibration_ready_after_s': ready_at, 'post_calibration_inputs_checked': projected_samples,
            'max_parked_deviation_native': max_parked_deviation,
            'first_telemetry_vs_dora_input_max_difference': first_telemetry_delta,
            'saved_input_projection_bit_identical': True if previous_projection is not None else None,
            'active_gripper_opening_span': float(adapter.high[7]-adapter.low[7]),
            'adapter': adapter.report()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'checks': [{k:v for k,v in check.items() if k != 'adapter'}
                                       for check in report['checks']]}, indent=2))


if __name__ == '__main__':
    main()
