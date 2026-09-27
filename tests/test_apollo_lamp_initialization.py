"""Pure recorded-start checks. Tests never connect to a robot."""
import copy
import json

import pytest

from apollo_lamp_initialization import (
    LAMP_INITIAL, LAMP_REPO, LAMP_EPISODE, LAMP_VERIFIED_REPLAY_FINAL,
    initialization_ready, validate_recorded_start, validate_reset_review,
    validate_reset_telemetry, validate_reset_transit_telemetry,
)


def telemetry():
    return {'session': {'session_id': 's', 'state': 'running'}, 'external': {},
            'collision': {'blocked': False}, 'arms': [
                {'arm_id': name, **copy.deepcopy(v), 'error_code': 0, 'stale': False,
                 'recovering': False, 'ee_pose': {'position': [.673, -.012, .173]}}
                for name, v in LAMP_INITIAL.items()]}


def test_valid_review_and_start():
    validate_reset_review({'operator_confirmed_recorded_start_reset': True,
                           'reset_repo_id': LAMP_REPO, 'reset_episode_id': LAMP_EPISODE})
    validate_recorded_start({'repo_id': LAMP_REPO, 'episode_id': LAMP_EPISODE,
                             'frames': 521, 'fps': 25., 'arms': telemetry()['arms']})
    validate_reset_telemetry(telemetry(), 's', arrived=True)


@pytest.mark.parametrize('change', ['session', 'state', 'joint', 'rail', 'floor', 'open', 'fault', 'action', 'nan'])
def test_reset_rejects_changed_or_unsafe_state(change):
    m = telemetry()
    if change == 'session': m['session']['session_id'] = 'other'
    if change == 'state': m['session']['state'] = 'fault'
    if change == 'joint': m['arms'][0]['q'][0] += .23
    if change == 'rail': m['arms'][1]['rail_pos_m'] = .004
    if change == 'floor': m['arms'][0]['ee_pose']['position'][2] = .119
    if change == 'open': m['arms'][0]['gripper_open_frac'] = .5
    if change == 'fault': m['arms'][1]['error_code'] = 1
    if change == 'action': m['external']['action_age_s'] = .1
    if change == 'nan': m['arms'][1]['q'][2] = float('nan')
    with pytest.raises(ValueError): validate_reset_telemetry(m, 's')


def test_near_start_is_not_arrival():
    m = telemetry(); m['arms'][0]['q'][1] += .01
    validate_reset_telemetry(m, 's')
    with pytest.raises(ValueError): validate_reset_telemetry(m, 's', arrived=True)


def test_reviewed_absolute_prefix_endpoint_can_return_to_start():
    m = telemetry(); m['arms'][0]['q'][1] += .20
    validate_reset_telemetry(m, 's')
    m = telemetry(); m['arms'][1]['q'][1] += .09
    with pytest.raises(ValueError): validate_reset_telemetry(m, 's')


def test_verified_replay_endpoint_and_reset_transit_are_admitted():
    import numpy as np
    m = telemetry()
    m['arms'][0]['q'] = LAMP_VERIFIED_REPLAY_FINAL['q'][:]
    m['arms'][0]['ee_pose']['position'] = [.666, .185, .121]
    validate_reset_telemetry(m, 's')
    m['arms'][0]['q'] = ((np.asarray(LAMP_INITIAL['grip']['q'])
                          + np.asarray(LAMP_VERIFIED_REPLAY_FINAL['q']))/2).tolist()
    m['arms'][0]['ee_pose']['position'] = [.665, .08, .14]
    validate_reset_transit_telemetry(m, 's')


@pytest.mark.parametrize('change', ['joint', 'floor', 'rail', 'action', 'collision'])
def test_reset_transit_still_rejects_out_of_corridor_or_unhealthy_state(change):
    m = telemetry()
    if change == 'joint': m['arms'][0]['q'][0] = .2
    if change == 'floor': m['arms'][0]['ee_pose']['position'][2] = .079
    if change == 'rail': m['arms'][0]['rail_pos_m'] = .64
    if change == 'action': m['external']['action_age_s'] = .1
    if change == 'collision': m['collision']['blocked'] = True
    with pytest.raises(ValueError): validate_reset_transit_telemetry(m, 's')


def test_wrong_recording_or_review_refused():
    with pytest.raises(ValueError): validate_reset_review({})
    data = {'repo_id': LAMP_REPO, 'episode_id': LAMP_EPISODE, 'frames': 521,
            'fps': 25., 'arms': telemetry()['arms']}
    data['arms'][0]['q'][0] += .002
    with pytest.raises(ValueError): validate_recorded_start(data)


def test_readiness_is_session_bound_and_finite(tmp_path):
    p = tmp_path/'ready.json'; s = {'session_id': 's', 'epoch': 'e'}
    assert not initialization_ready(p, s, 'r', 100.)
    valid = {'purpose': 'LAMP_RECORDED_START_VERIFIED', **s, 'review_id': 'r', 'expires_t_mono': 110.}
    p.write_text(json.dumps(valid)); assert initialization_ready(p, s, 'r', 100.)
    for changed in [{'session_id': 'other'}, {'epoch': 'x'}, {'review_id': 'x'},
                    {'expires_t_mono': 99.}, {'expires_t_mono': 116.}]:
        p.write_text(json.dumps(valid | changed))
        with pytest.raises(ValueError): initialization_ready(p, s, 'r', 100.)


def test_recorded_prefix_is_exact_and_exhausts(tmp_path):
    import numpy as np
    from apollo_lamp_recorded_prefix import RecordedLampPrefix
    state = np.zeros((521, 32), dtype=np.float32)
    state[0, :7] = LAMP_INITIAL['grip']['q']
    state[0, 16:23] = LAMP_INITIAL['view']['q']
    commands = np.zeros((521, 16), dtype=np.float32)
    commands[:, 0] = -.001; commands[:, 6] = .98; commands[:, 14] = 1
    path = tmp_path/'prefix.npz'; np.savez(path, state=state, delta_commands=commands)
    predictor = RecordedLampPrefix(path)
    actual = np.concatenate([predictor.predict_chunk(None, None, None) for _ in range(3)])
    np.testing.assert_array_equal(actual, commands[:24])
    with pytest.raises(ValueError, match='exhausted'): predictor.predict_chunk(None, None, None)
    commands[10, 6] = .5; np.savez(path, state=state, delta_commands=commands)
    with pytest.raises(ValueError, match='reviewed'): RecordedLampPrefix(path)


def test_absolute_rail_wire_holds_every_native_mm_value():
    import numpy as np
    from apollo_lamp_approach import stationary_rail_wire_value
    for mm in range(651):
        value = stationary_rail_wire_value(np.float32(mm/1000))
        # Exact observed hardware boundary: clamp then truncate m*1000.
        sdk_mm = int(min(max(float(value)*1000, 0), 650))
        assert sdk_mm == mm
        assert abs(float(value)-mm/1000) < 1e-7
    for bad in [float('nan'), float('inf'), -.001, .7, .6355]:
        with pytest.raises(ValueError): stationary_rail_wire_value(bad)
