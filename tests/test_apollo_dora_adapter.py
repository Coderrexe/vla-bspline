"""Isolated CPU tests: no Dora connection, HTTP request, robot import, or motion.

Run only this file in the isolated inference environment with our code directory
and the pinned mavis_policy_node directory on PYTHONPATH. All outputs are mocks.
"""
import json

import numpy as np
import pyarrow as pa
import pytest
from safetensors.numpy import save_file
from scipy.spatial.transform import Rotation

from apollo_legacy_state import (
    STATE_NAMES, GRIP_ACTION_NAMES, CheckpointStateAdapter, select_state,
    current_tcp_to_training_state, training_to_current_tcp_state,
    arm_stream_to_current_state,
    align_quaternion_hemisphere,
)
from apollo_dora_policy import ApolloDoraPolicy
from apollo_parked_state import SessionParkedStateAdapter
from run_apollo_dora import GuardedPolicyNode
from mavis_policy_node.contract import validate_action_metadata
from mavis_policy_node.messages import ImageFrame
from mavis_policy_node.types import Observation


class Clock:
    t = 1000.

    def __call__(self):
        return self.t


class Predictor:
    task = 'Drawer Assembling'

    def __init__(self, clock):
        self.clock = clock
        self.delay = .02
        self.raw = np.zeros((8, 16), np.float32)
        self.raw[:, 0] = .0002
        self.raw[:, 6] = 1
        self.raw[:, 14] = 1

    def reset(self):
        pass

    def predict_chunk(self, state, view, grip):
        self.last_state = state.copy()
        self.clock.t += self.delay
        return self.raw.copy()


@pytest.fixture
def setup(tmp_path):
    legacy = np.zeros(32, np.float32)
    legacy[[7, 23, 12, 28]] = 1
    legacy[8] = .636
    legacy[9:12] = [.45, .05, .30]
    lo, hi, std = legacy.copy(), legacy.copy(), np.ones(32, np.float32)
    mask = np.zeros(32, bool)
    mask[8] = True
    mask[16:] = True
    lo[~mask] -= 1
    hi[~mask] += 1
    std[mask] = 0
    (tmp_path/'apollo_interface.json').write_text(json.dumps({
        'features': {'observation.state': {'names': STATE_NAMES}}}))
    save_file({'observation.state.min': lo, 'observation.state.max': hi,
               'observation.state.std': std, 'observation.state.mean': legacy},
              str(tmp_path/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
    return tmp_path, legacy, training_to_current_tcp_state(legacy), Clock()


def session(**overrides):
    s = {'session_id': 'trial1', 'epoch': 'epoch1', 'state': 'running',
         'kind': 'hardware', 'policy_source': 'external', 'arm_ids': ['view', 'grip'],
         'frames': {'grip': 'arm_base:grip', 'view': 'arm_base:view'},
         'spec': {'mode': 'inference', 'start_from': 'keep_current', 'speed_scale': .1},
         'action_space': 'delta_ee', 'state_names': STATE_NAMES[16:]+STATE_NAMES[:16],
         'action_names': [n.replace('grip_', 'view_') for n in GRIP_ACTION_NAMES]+GRIP_ACTION_NAMES}
    s.update(overrides)
    return s


def policy_for(setup, execute=False, **kwargs):
    checkpoint, _, _, clock = setup
    predictor = Predictor(clock)
    kwargs.setdefault('action_rows',8)
    policy = ApolloDoraPolicy(checkpoint, predictor=predictor, clock=clock,
                              execute_session='trial1' if execute else None, **kwargs)
    policy.on_session(session())
    return policy


def observation(setup, oid=1, age=0):
    _, _, state, clock = setup
    return Observation(state.copy(), {'view_wrist': np.zeros((480, 640, 3), np.uint8),
                       'grip_wrist': np.zeros((480, 640, 3), np.uint8)},
                       clock()-age, 1, observation_id=oid)


def test_pose_conversion_and_roundtrip(setup):
    _, legacy, current, _ = setup
    np.testing.assert_allclose(current[9:12], legacy[9:12]+[0, 0, .172], atol=1e-7)
    np.testing.assert_allclose(abs(current[12:16]), [0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(current_tcp_to_training_state(current), legacy, atol=1e-7)
    rng = np.random.default_rng(13)
    batch = np.tile(current, (100, 1))
    for start in [0, 16]:
        rotation = Rotation.from_euler('xyz', rng.uniform(-1, 1, (100, 3)))
        batch[:, start+12:start+16] = rotation.as_quat(canonical=True)[:, [3, 0, 1, 2]]
    np.testing.assert_allclose(training_to_current_tcp_state(
        current_tcp_to_training_state(batch)), batch, atol=2e-7)


def test_state_reordering(setup):
    _, _, current, _ = setup
    order = np.arange(31, -1, -1)
    np.testing.assert_array_equal(select_state(current[order], [STATE_NAMES[i] for i in order]), current)
    with pytest.raises(ValueError):
        select_state(current, ['same']*32)
    with pytest.raises(ValueError):
        select_state(current[:-1], STATE_NAMES[:-1])


def test_constant_features_restored_but_moved_parked_arm_refused(setup):
    checkpoint, legacy, current, _ = setup
    adapter = CheckpointStateAdapter(checkpoint)
    drift = current.copy()
    drift[16] += 1e-6
    restored = adapter(drift)
    np.testing.assert_array_equal(restored[adapter.mask], legacy[adapter.mask])
    drift[16] += .01
    with pytest.raises(ValueError, match='Parked'):
        adapter(drift)
    with pytest.raises(ValueError):
        CheckpointStateAdapter(checkpoint, tolerance=.1)


def test_bad_state_refused(setup):
    checkpoint, _, current, _ = setup
    adapter = CheckpointStateAdapter(checkpoint)
    current[12:16] = 0
    with pytest.raises(ValueError, match='quaternion'):
        adapter(current)
    current[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        adapter(current)


def test_constant_feature_error_identifies_failed_field(setup):
    checkpoint, _, current, _ = setup
    adapter = CheckpointStateAdapter(checkpoint)
    drift = np.stack([current, current])
    drift[1, 17] += .005
    with pytest.raises(ValueError, match=r'view_joint2.pos=.*training'):
        adapter(drift)


def test_refused_observation_saved_without_action(setup, tmp_path):
    output = tmp_path/'refused'
    p = policy_for(setup, execute=True, output=output)
    obs = observation(setup)
    obs.state[17] += .005
    with pytest.raises(ValueError, match='Parked'):
        p.act(obs)
    assert p.disarmed and p.predictions == 0 and p.returned_chunks == 0
    row = json.loads((output/'refusals.jsonl').read_text())
    assert row['publication_requested'] is False
    assert row['session_id'] == 'trial1'
    with np.load(output/row['input_file'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['current_state'], obs.state)
        np.testing.assert_array_equal(saved['view_rgb'], obs.images['view_wrist'])
    assert p.act(observation(setup, oid=2)) is None


def test_refusal_logging_failure_preserves_disarm(setup, tmp_path, monkeypatch):
    p = policy_for(setup, execute=True, output=tmp_path/'unwritable')
    obs = observation(setup)
    obs.state[17] += .005
    def fail_save(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(np, 'savez_compressed', fail_save)
    with pytest.raises(ValueError, match='Parked'):
        p.act(obs)
    assert p.disarmed and p.returned_chunks == 0
    assert 'diagnostic save also failed' in p.last_status


def test_arm_stream_reordered_and_health_checked(setup):
    _, _, current, _ = setup
    layout = ([f'q{i}' for i in range(1, 8)]+['rail_pos']
              + [f'dq{i}' for i in range(1, 8)]+['drail']
              + [f'ee_base.{n}' for n in ['x','y','z','qw','qx','qy','qz']]
              + [f'ee_world.{n}' for n in ['x','y','z','qw','qx','qy','qz']]
              + ['gripper_open_frac','rail_pos_m'])
    data = np.zeros((2, 32))
    for row, start in [(0,16), (1,0)]:
        data[row,:7] = current[start:start+7]
        data[row,16:23] = current[start+9:start+16]
        data[row,30:32] = current[start+7:start+9]
    meta = {'layout':layout, 'arm_ids':['view','grip'], 'stale':[0,0],
            'error_code':[0,0], 'has_rail':[1,1]}
    np.testing.assert_array_equal(arm_stream_to_current_state(data.ravel(),meta),current)
    meta['stale'] = [0,1]
    with pytest.raises(ValueError, match='stale'):
        arm_stream_to_current_state(data.ravel(),meta)


def test_shadow_never_returns_action(setup):
    p = policy_for(setup)
    assert p.act(observation(setup)) is None
    assert p.predictions == 1 and p.returned_chunks == 0


def test_constant_diagnostic_forbids_motion_configuration(setup):
    with pytest.raises(ValueError, match='shadow only'):
        policy_for(setup, execute=True, shadow_constant_diagnostic=True)


def test_constant_diagnostic_records_mismatch_without_motion(setup, tmp_path):
    output = tmp_path/'constant_diagnostic'
    p = policy_for(setup, shadow_constant_diagnostic=True, output=output)
    obs = observation(setup)
    obs.state[17] += .005
    assert p.act(obs) is None
    assert p.predictions == 1 and p.returned_chunks == 0
    row = json.loads((output/'predictions.jsonl').read_text())
    assert row['constant_feature_diagnostic'] and not row['publication_requested']
    assert row['constant_feature_differences']['view_joint2.pos'] == pytest.approx(.005)
    with np.load(output/'diagnostic_first_input.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['current_state'], obs.state)
        assert f['projected_training_state'][17] == 0
    # A later accidental session assignment must not convert this diagnostic
    # into a motion-capable policy.
    p.execute_session = 'trial1'
    assert not p._may_publish()
    assert p.act(observation(setup, oid=2)) is None


def test_constant_diagnostic_still_refuses_invalid_pose(setup):
    p = policy_for(setup, shadow_constant_diagnostic=True)
    obs = observation(setup)
    obs.state[28:32] = 0
    with pytest.raises(ValueError, match='quaternion'):
        p.act(obs)
    assert p.disarmed and p.returned_chunks == 0


def test_one_chunk_budget_and_reset_disarm(setup):
    p = policy_for(setup, execute=True)
    out = p.act(observation(setup))
    assert out.actions.shape == (8,8) and p.returned_chunks == 1
    assert np.all(out.actions[:,7] == 0)
    assert p.act(observation(setup,2)) is None
    p.reset()
    assert p.disarmed


def test_commissioning_prefix_preserves_exact_first_row(setup):
    checkpoint, _, _, clock = setup
    predictor = Predictor(clock)
    predictor.raw[1:,0] = .02  # later unexecuted rows exceed first-trial budget
    p = ApolloDoraPolicy(checkpoint,predictor=predictor,clock=clock,execute_session='trial1')
    p.on_session(session())
    out = p.act(observation(setup))
    assert p.chunk_len == 1 and out.actions.shape == (1,8)
    np.testing.assert_array_equal(out.actions[0],predictor.raw[0,:8])
    assert p.act(observation(setup,2)) is None


@pytest.mark.parametrize('count',[0,9,1.5])
def test_invalid_execution_rows_refused(setup,count):
    with pytest.raises(ValueError):
        policy_for(setup,True,action_rows=count)


@pytest.mark.parametrize('changes', [
    {'session_id':'other'}, {'kind':'sim'}, {'policy_source':'local'},
    {'state':'idle'}, {'arm_ids':['grip']},
    {'frames':{'grip':'world', 'view':'arm_base:view'}},
    {'spec':{'mode':'collect','start_from':'keep_current','speed_scale':.1}},
    {'spec':{'mode':'inference','start_from':'profile','speed_scale':.1}},
    {'spec':{'mode':'inference','start_from':'keep_current','speed_scale':1}},
    {'spec':{'mode':'inference','start_from':'keep_current','speed_scale':float('nan')}},
])
def test_mismatched_session_no_action(setup, changes):
    p = policy_for(setup, execute=True)
    p.on_session(session(**changes))
    assert p.act(observation(setup)) is None and p.returned_chunks == 0


@pytest.mark.parametrize('age', [.3, -.1, float('nan')])
def test_stale_or_future_observation_no_action(setup, age):
    p = policy_for(setup, execute=True)
    assert p.act(observation(setup, age=age)) is None


def test_prediction_stales_while_computing(setup):
    p = policy_for(setup, execute=True)
    p.predictor.delay = .6
    assert p.act(observation(setup)) is None and p.disarmed


def test_deadline_and_session_heartbeat(setup):
    p = policy_for(setup, execute=True, max_seconds=.01)
    assert p.act(observation(setup)) is None
    p = policy_for(setup, execute=True)
    setup[3].t += 2.1
    assert p.act(observation(setup)) is None


@pytest.mark.parametrize('channel,value', [(0,.02), (3,.1), (6,.5), (8,.001), (0,np.nan)])
def test_invalid_or_oversized_actions_refused(setup, channel, value):
    p = policy_for(setup, execute=True)
    p.predictor.raw[:,channel] = value
    with pytest.raises(ValueError):
        p.act(observation(setup))
    assert p.disarmed and p.returned_chunks == 0


class WireStub:
    def __init__(self, outputs=None):
        self.outputs = outputs or ['spec','status','action_grip']
        self.sent = []

    def node_config(self):
        return {'outputs': self.outputs}

    def send_output(self, topic, values, metadata):
        assert topic in self.outputs
        self.sent.append((topic,values,metadata))


def json_event(topic, message):
    return {'type':'INPUT', 'id':topic, 'value':pa.array([json.dumps(message)]),
            'metadata':{'mavis_schema':1, 'epoch':'epoch1','session_id':'trial1'}}


def wire_setup(setup, execute=True, outputs=None, **policy_kwargs):
    p = policy_for(setup, execute, **policy_kwargs)
    node = GuardedPolicyNode(p, clock=setup[3], rate_hz=25, chunk_dt_s=.04)
    stub = WireStub(outputs)
    node._on_input(stub,json_event('session',session()))
    for cam in p.spec.camera_keys:
        node._frames[cam] = ImageFrame(7,setup[3](),np.zeros((480,640,3),np.uint8))
    state = setup[2]
    event = {'type':'INPUT','id':'obs_state','value':pa.array(np.r_[state[16:],state[:16]]),
             'metadata':{'mavis_schema':1,'epoch':'epoch1','session_id':'trial1',
                         't_mono':setup[3](),'observation_id':1,'wallclock_ns':1,
                         'state_names':STATE_NAMES[16:]+STATE_NAMES[:16],
                         'image_camera_ids':p.spec.camera_keys,'image_seq':[7,7],
                         'quat_order':'wxyz','engaged_arm':''}}
    return p,node,stub,event


def test_wire_exactly_one_grip_message_with_correct_metadata(setup):
    p,node,stub,event = wire_setup(setup)
    node._on_input(stub,event)
    actions = [(a,v,m) for a,v,m in stub.sent if a.startswith('action')]
    assert len(actions) == 1 and actions[0][0] == 'action_grip'
    _,values,meta = actions[0]
    assert len(values) == 64 and meta['action_dim'] == 8 and meta['chunk_len'] == 8
    assert meta['chunk_dt_s'] == .04 and meta['observation_id'] == 1
    assert meta['session_id'] == 'trial1' and meta['epoch'] == 'epoch1'
    assert validate_action_metadata(meta) == []
    assert node.stats.arm_action_fallbacks == 0


def test_shadow_wire_no_action(setup):
    p,node,stub,event = wire_setup(setup,False)
    node._on_input(stub,event)
    assert p.predictions == 1 and not any(a.startswith('action') for a,_,_ in stub.sent)


def test_initialization_receipt_defers_all_predictions(setup, tmp_path):
    p, node, stub, event = wire_setup(setup)
    p.lamp_review_id = 'r'
    node.initialization_complete = False
    node.initialization_ready_file = tmp_path/'ready.json'
    node._on_input(stub, event)
    assert p.predictions == 0 and p.returned_chunks == 0
    node.initialization_ready_file.write_text(json.dumps({
        'purpose': 'LAMP_RECORDED_START_VERIFIED', 'session_id': 'trial1',
        'epoch': 'epoch1', 'review_id': 'r', 'expires_t_mono': 1010.}))
    node._on_input(stub, event)
    assert node.initialization_complete and p.returned_chunks == 1


def test_initialization_wrong_receipt_disarms(setup, tmp_path):
    p, node, stub, event = wire_setup(setup)
    p.lamp_review_id = 'r'
    node.initialization_complete = False
    node.initialization_ready_file = tmp_path/'ready.json'
    node.initialization_ready_file.write_text('{}')
    node._on_input(stub, event)
    assert p.disarmed and p.predictions == 0


@pytest.mark.parametrize('initialized,reason,should_disarm', [
    (False, 'handback', False), (True, 'handback', True),
    (False, 'anomaly', True), (False, 'session_stop', True)])
def test_initialization_only_accepts_pre_motion_handback(setup, initialized, reason, should_disarm):
    p, node, stub, _ = wire_setup(setup, execute=False)
    node.initialization_complete = initialized
    node._on_input(stub, json_event('policy_reset', {'reason': reason, 'after_observation_id': 7}))
    assert p.disarmed == should_disarm
    assert node._watermark == 7 and p.returned_chunks == 0


@pytest.mark.parametrize('field,value',[('epoch','old'),('session_id','other'),
                                      ('quat_order','xyzw'),('engaged_arm','grip')])
def test_wire_metadata_refusal(setup,field,value):
    p,node,stub,event = wire_setup(setup)
    event['metadata'][field] = value
    node._on_input(stub,event)
    assert not any(a.startswith('action') for a,_,_ in stub.sent)


def test_wire_no_whole_cell_fallback(setup):
    p,node,stub,event = wire_setup(setup,outputs=['spec','status','action'])
    node._on_input(stub,event)
    assert p.disarmed and not any(a.startswith('action') for a,_,_ in stub.sent)


@pytest.mark.parametrize('kind',['gate','collision','policy_anomaly','session_error'])
def test_intervention_event_disarms(setup,kind):
    p,node,stub,event = wire_setup(setup)
    node._on_input(stub,json_event('events',{'kind':kind}))
    node._on_input(stub,event)
    assert p.disarmed and not any(a.startswith('action') for a,_,_ in stub.sent)


def test_camera_age_refusal_after_first_action(setup):
    p,node,stub,event = wire_setup(setup)
    p.returned_chunks = 1
    setup[3].t += .3
    event['metadata']['t_mono'] = setup[3]()
    node._on_input(stub,event)
    assert p.disarmed


def test_link_loss_disarms(setup):
    p,node,stub,event = wire_setup(setup)
    node._release(stub)
    assert p.disarmed


def parked_adapter(setup):
    return SessionParkedStateAdapter(CheckpointStateAdapter(setup[0]), operator_confirmed=True)


def fill_parked_reference(adapter, current):
    for i in range(11):
        assert adapter.adapt(current, session_id='trial1', epoch='epoch1', t_mono=1000+i*.32) is None
    assert adapter.ready


def corrected_checkpoint(setup):
    from safetensors.numpy import load_file
    checkpoint, _, current, _ = setup
    path = checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors'
    stats = load_file(str(path))
    original_mean = stats['observation.state.mean'].copy()
    for key in ('min', 'max', 'mean'):
        stats[f'observation.state.{key}'] += current-original_mean
    save_file(stats, str(path))
    (checkpoint/'apollo_observation_contract.json').write_text(json.dumps({
        'schema_version': 1, 'state_pose_convention': 'corrected_tcp',
        'action_pose_convention': 'base_frame_delta_translation_and_spatial_rotvec',
        'gripper': 'absolute_open_fraction_0_to_1', 'legacy_observation_conversion': False}))
    (checkpoint/'dataset_provenance.json').write_text(json.dumps({'task': 'Lamp Assembling'}))
    return checkpoint


def lamp_approach_case(setup, tmp_path, absolute=False, row_dt=.2):
    from apollo_lamp_approach import LampApproachPolicy, LAMP_APPROACH_LIMITS
    setup[2][9:12] = [.67, -.015, .166]
    checkpoint = corrected_checkpoint(setup)
    predictor = Predictor(setup[3]); predictor.task = 'Lamp Assembling'
    predictor.raw[:, 0] = -.001
    policy = LampApproachPolicy(
        checkpoint, lamp_review_id='lamp-test', absolute_execution=absolute,
        predictor=predictor, clock=setup[3],
        max_chunks=3, max_seconds=20 if row_dt == .4 else 10, action_rows=8, bounded_rollout=True,
        supervised_approach=True, action_row_dt_s=row_dt, confirm_parked_setup=True,
        align_to_training_hemisphere=True, output=tmp_path/'lamp_policy')
    path = tmp_path/'lamp_grant.json'
    policy.await_bounded_rollout_authorization(path)
    grant = {'purpose':'SUPERVISED_LAMP_APPROACH', 'session_id':'trial1', 'epoch':'epoch1',
             'policy_id':policy.policy_id, 'action_rows':8, 'max_chunks':3,
             'expires_t_mono':1020 if row_dt == .4 else 1010, 'chunk_dt_s':row_dt, 'total_translation_path_m':.08,
             'total_rotation_path_rad':.15, 'total_gripper_change':.1,
             'prefix_translation_path_m':.04, 'row_translation_m':.006,
             'lamp_review_id':'lamp-test', 'lamp_approach_limits':LAMP_APPROACH_LIMITS,
             'wire_action_space': 'abs_ee' if absolute else 'delta_ee'}
    return policy, path, grant


def lamp_session():
    result = session()
    result['spec']['task'] = 'Lamp Assembling'
    return result


def test_lamp_absolute_encoding_preserves_native_pose_gripper_and_rail(setup, tmp_path):
    p, path, grant = lamp_approach_case(setup, tmp_path, absolute=True)
    p.predictor.raw[:, 4] = .001
    path.write_text(json.dumps(grant)); calibrate_lamp(p, setup)
    setup[3].t = 1003.52; p.on_session(lamp_session())
    obs = observation(setup, oid=12)
    result = p.act(obs)
    assert p.spec.action_space == 'abs_ee' and len(p.spec.action_names) == 11
    assert result.actions.shape == (8, 11)
    expected_p = obs.state[9:12]+np.cumsum(p.predictor.raw[:, :3], axis=0)
    np.testing.assert_allclose(result.actions[:, :3], expected_p, atol=1e-7)
    np.testing.assert_array_equal(result.actions[:, 9], p.predictor.raw[:, 6])
    from apollo_lamp_approach import stationary_rail_wire_value
    np.testing.assert_array_equal(result.actions[:, 10], np.full(8, stationary_rail_wire_value(obs.state[8])))
    r = Rotation.from_quat(obs.state[12:16][[1, 2, 3, 0]])
    for row, native in zip(result.actions, p.predictor.raw):
        r = Rotation.from_rotvec(native[3:6])*r
        np.testing.assert_allclose(row[3:9], np.r_[r.as_matrix()[:,0], r.as_matrix()[:,1]], atol=1e-7)
    assert p.wire_translation_path < .009


def test_lamp_absolute_rejects_wrong_wire_grant(setup, tmp_path):
    p, path, grant = lamp_approach_case(setup, tmp_path, absolute=True)
    path.write_text(json.dumps(grant | {'wire_action_space': 'delta_ee'}))
    p.on_session(lamp_session())
    with pytest.raises(ValueError, match='differs'): p.act(observation(setup))
    assert p.returned_chunks == 0


def test_lamp_absolute_slow_clock_keeps_three_chunk_motion_limits(setup, tmp_path):
    p, path, grant = lamp_approach_case(setup, tmp_path, absolute=True, row_dt=.4)
    assert p.authorization_window == 20 and p.translation_budget == .08 and p.max_chunks == 3
    path.write_text(json.dumps(grant)); calibrate_lamp(p, setup)
    setup[3].t = 1003.52; p.on_session(lamp_session())
    assert p.act(observation(setup, oid=12)).actions.shape == (8,11)
    setup[3].t = 1003.85; p.on_session(lamp_session())
    assert p.act(observation(setup, oid=13)) is None


def test_lamp_absolute_rejects_large_recorded_anchor_tracking_error(setup, tmp_path):
    p, path, grant = lamp_approach_case(setup, tmp_path, absolute=True)
    path.write_text(json.dumps(grant)); calibrate_lamp(p, setup)
    p.predictor.is_recorded_prefix = True
    p.absolute_anchor = (setup[2][9:12]+[.02,0,0], p.task_guard.rotation(setup[2]))
    setup[3].t = 1003.52; p.on_session(lamp_session())
    with pytest.raises(ValueError, match='unchanged'): p.act(observation(setup, oid=12))
    assert p.returned_chunks == 0


def calibrate_lamp(policy, setup):
    for i in range(11):
        setup[3].t = 1000+i*.32
        policy.on_session(lamp_session())
        assert policy.act(observation(setup, oid=i+1)) is None
        assert policy.returned_chunks == 0
    assert policy.parked_adapter.ready


def test_lamp_approach_has_three_exact_chunks_and_stops(setup, tmp_path):
    policy, path, grant = lamp_approach_case(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_lamp(policy, setup)
    returned = 0
    for i in range(15):
        setup[3].t = 1003.52+i*.33
        policy.on_session(lamp_session())
        result = policy.act(observation(setup, oid=20+i))
        if result is not None:
            np.testing.assert_array_equal(result.actions, policy.predictor.raw[:, :8])
            returned += 1
    assert returned == 3
    setup[3].t = 1008.47; policy.on_session(lamp_session())
    assert policy.act(observation(setup, oid=40)) is None
    assert policy.returned_chunks == 3


@pytest.mark.parametrize('field,value', [
    ('purpose', 'SUPERVISED_APPROACH'), ('lamp_review_id', 'other'),
    ('lamp_approach_limits', {}), ('max_chunks', 4), ('chunk_dt_s', .04),
])
def test_lamp_grant_cannot_change_scope(setup, tmp_path, field, value):
    policy, path, grant = lamp_approach_case(setup, tmp_path)
    path.write_text(json.dumps(grant | {field: value})); policy.on_session(lamp_session())
    with pytest.raises(ValueError): policy.act(observation(setup))
    assert policy.disarmed and policy.returned_chunks == 0


def test_lamp_rejects_grasp_and_intermediate_floor_crossing():
    from apollo_lamp_approach import LampApproachGuard, LAMP_APPROACH_LIMITS
    guard = LampApproachGuard()
    state = np.zeros(32); state[12] = 1; state[7] = 1
    state[9:12] = [.67, -.015, .166]
    guard.observe(state, 0, 0, None)
    actions = np.zeros((8, 16)); actions[:, 6] = 1
    state[11] = .101
    actions[0, 2] = -.002; actions[1, 2] = .002
    with pytest.raises(ValueError, match='workspace'): guard.check_prefix(state, actions)
    actions[:, 2] = 0; actions[:, 6] = .8
    with pytest.raises(ValueError, match='closure'): guard.check_prefix(state, actions)


def test_lamp_rejects_unreviewed_start():
    from apollo_lamp_approach import LampApproachGuard, LAMP_APPROACH_LIMITS
    state = np.zeros(32); state[12] = 1; state[7] = 1
    state[9:12] = [.70, -.015, .166]
    with pytest.raises(ValueError, match='initial pose'):
        LampApproachGuard().observe(state, 0, 0, None)


@pytest.mark.parametrize('position', [
    [.67335826, -.01202143, .17344430],
    [.67098469, -.01503715, .16608219],
    [.66783726, -.01709432, .15815733],
])
def test_lamp_start_region_accepts_recorded_startup_positions(position):
    from apollo_lamp_approach import LampApproachGuard
    state = np.zeros(32); state[12] = 1; state[7] = 1; state[9:12] = position
    LampApproachGuard().observe(state, 0, 0, None)


@pytest.mark.parametrize('position', [[.64,0,.16],[.68,-.04,.16],[.67,0,.13],[.67,0,.20]])
def test_lamp_start_region_rejects_unsupported_positions(position):
    from apollo_lamp_approach import LampApproachGuard
    state = np.zeros(32); state[12] = 1; state[7] = 1; state[9:12] = position
    with pytest.raises(ValueError, match='initial pose'):
        LampApproachGuard().observe(state, 0, 0, None)


def test_corrected_tcp_checkpoint_never_applies_legacy_conversion(setup):
    checkpoint = corrected_checkpoint(setup)
    adapter = CheckpointStateAdapter(checkpoint)
    current = setup[2]
    np.testing.assert_array_equal(adapter(current), current)
    np.testing.assert_array_equal(adapter(adapter.to_current_state(current)), current)
    assert np.linalg.norm(current_tcp_to_training_state(current)[9:12]-current[9:12]) > .17
    parked = SessionParkedStateAdapter(adapter, operator_confirmed=True)
    fill_parked_reference(parked, current)
    result = parked.adapt(current, session_id='trial1', epoch='epoch1', t_mono=1003.6)
    np.testing.assert_array_equal(result, current)
    assert parked.report()['state_pose_convention'] == 'corrected_tcp'


@pytest.mark.parametrize('shadow_diagnostic', [False, True])
def test_corrected_tcp_policy_input_in_both_shadow_paths(setup, shadow_diagnostic):
    corrected_checkpoint(setup)
    policy = policy_for(setup, shadow_constant_diagnostic=shadow_diagnostic)
    assert policy.act(observation(setup)) is None  # shadow never publishes
    np.testing.assert_array_equal(policy.predictor.last_state, setup[2])


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2), ('state_pose_convention', 'flange'),
    ('legacy_observation_conversion', True), ('gripper', 'delta'),
    ('action_pose_convention', 'tool_frame')])
def test_ambiguous_observation_contract_refused(setup, field, value):
    checkpoint = corrected_checkpoint(setup)
    path = checkpoint/'apollo_observation_contract.json'
    contract = json.loads(path.read_text()); contract[field] = value
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='contract'):
        CheckpointStateAdapter(checkpoint)


def test_new_task_without_observation_contract_refused(setup):
    (setup[0]/'dataset_provenance.json').write_text(json.dumps({'task': 'Lamp Assembling'}))
    with pytest.raises(ValueError, match='explicit Apollo observation contract'):
        CheckpointStateAdapter(setup[0])


def test_corrected_parked_roundoff_is_projected_without_relaxing_band(setup):
    from safetensors.numpy import load_file
    checkpoint = corrected_checkpoint(setup)
    path = checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors'
    stats = load_file(str(path))
    stats['observation.state.std'][30] = 1.7e-8
    stats['observation.state.min'][30] -= 4e-8
    stats['observation.state.max'][30] += 4e-8
    save_file(stats, str(path))
    adapter = CheckpointStateAdapter(checkpoint)
    assert adapter.mask.sum() == 17 and adapter.tolerance == 1e-3
    state = setup[2].copy(); state[30] += 3e-8
    assert adapter.stabilize_training_state(state)[30] == adapter.mean[30]
    parked = SessionParkedStateAdapter(adapter, operator_confirmed=True)
    fill_parked_reference(parked, setup[2])
    # A genuine moved parked input is still refused.
    moved = setup[2].copy(); moved[17] += .002
    with pytest.raises(ValueError, match='Parked hardware changed'):
        parked.adapt(moved, session_id='trial1', epoch='epoch1', t_mono=1003.6)


def test_corrected_nonstationary_camera_arm_requires_new_interface(setup):
    from safetensors.numpy import load_file
    checkpoint = corrected_checkpoint(setup)
    path = checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors'
    stats = load_file(str(path)); stats['observation.state.max'][17] += .01
    save_file(stats, str(path))
    with pytest.raises(ValueError, match='nonstationary parked features'):
        CheckpointStateAdapter(checkpoint)


def test_parked_reference_requires_operator_and_exact_constant_mask(setup):
    adapter = CheckpointStateAdapter(setup[0])
    with pytest.raises(ValueError, match='Operator approval'):
        SessionParkedStateAdapter(adapter)
    adapter.mask[0] = True
    with pytest.raises(ValueError, match='17 constant'):
        SessionParkedStateAdapter(adapter, operator_confirmed=True)


def test_session_reference_restores_only_constants_after_stationary_window(setup):
    checkpoint, _, current, _ = setup
    current = current.copy()
    current[17] += .0065
    current[8] -= .001
    # The original path still refuses this mismatch: it was not weakened.
    with pytest.raises(ValueError, match='Parked'):
        CheckpointStateAdapter(checkpoint)(current)
    adapter = parked_adapter(setup)
    fill_parked_reference(adapter, current)
    # Varying grip state remains observed, never replaced with a parked baseline.
    current[0] += .02
    current[9] += .005
    result = adapter.adapt(current, session_id='trial1', epoch='epoch1', t_mono=1003.52)
    expected = current_tcp_to_training_state(current)
    np.testing.assert_array_equal(result[~adapter.mask], expected[~adapter.mask])
    np.testing.assert_array_equal(result[adapter.mask], adapter.model_reference[adapter.mask])
    report = adapter.report()
    assert report['samples'] == 11 and not report['reloadable_for_execution']
    assert report['epoch'] == 'epoch1' and report['ready']


def test_calibration_requires_time_as_well_as_samples(setup):
    adapter = parked_adapter(setup)
    for i in range(11):
        assert adapter.adapt(setup[2], session_id='trial1', epoch='epoch1', t_mono=1000+i*.1) is None
    assert not adapter.ready


@pytest.mark.parametrize('field', [0, 8, 16, 23, 24, 25])
def test_motion_during_calibration_invalidates_without_retry(setup, field):
    adapter = parked_adapter(setup)
    state = setup[2].copy()
    assert adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000) is None
    state[field] += -.002 if field == 23 else .002
    with pytest.raises(ValueError, match='changed during'):
        adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000.32)
    with pytest.raises(ValueError, match='invalidated'):
        adapter.adapt(setup[2], session_id='trial1', epoch='epoch1', t_mono=1000.64)


@pytest.mark.parametrize('field', [8, 16, 23, 24, 25])
def test_moved_parked_field_after_calibration_refused(setup, field):
    adapter = parked_adapter(setup)
    fill_parked_reference(adapter, setup[2])
    state = setup[2].copy()
    state[field] += -.002 if field == 23 else .002
    with pytest.raises(ValueError, match='hardware changed'):
        adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1003.52)
    assert adapter.invalid and not adapter.ready


@pytest.mark.parametrize('sid,epoch', [('other','epoch1'),('trial1','new'),('', 'epoch1'),('trial1',None)])
def test_reference_cannot_cross_session_or_epoch(setup, sid, epoch):
    adapter = parked_adapter(setup)
    fill_parked_reference(adapter, setup[2])
    with pytest.raises(ValueError):
        adapter.adapt(setup[2], session_id=sid, epoch=epoch, t_mono=1003.52)
    assert adapter.invalid


@pytest.mark.parametrize('timestamp', [1003.2,1003.0,1004.1,float('nan')])
def test_parked_reference_rejects_stale_or_interrupted_stream(setup, timestamp):
    adapter = parked_adapter(setup)
    fill_parked_reference(adapter, setup[2])
    with pytest.raises(ValueError):
        adapter.adapt(setup[2], session_id='trial1', epoch='epoch1', t_mono=timestamp)
    assert not adapter.ready


def test_calibration_invalid_quaternion_refused(setup):
    adapter = parked_adapter(setup)
    state = setup[2].copy()
    state[28:32] = 0
    with pytest.raises(ValueError, match='quaternion'):
        adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000)


def test_active_gripper_readout_is_not_a_parked_feature(setup):
    adapter = parked_adapter(setup)
    for i in range(11):
        state = setup[2].copy()
        state[7] = 83/84 if i < 5 else 1.0  # observed G2 opening readout update
        assert adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000+i*.32) is None
    assert adapter.ready
    state[7] = 83/84
    projected = adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1003.52)
    assert projected[7] == state[7]  # never hidden or projected to a training constant
    assert 7 not in adapter.report()['calibration_indices']


@pytest.mark.parametrize('value', [-.01, 1.01])
def test_calibration_rejects_invalid_gripper_readout(setup, value):
    adapter = parked_adapter(setup)
    state = setup[2].copy()
    state[7] = value
    with pytest.raises(ValueError, match='Gripper opening'):
        adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000)


def test_calibrated_policy_only_one_exact_row_after_no_action_window(setup, tmp_path):
    p = policy_for(setup, execute=True, action_rows=1, confirm_parked_setup=True,
                   output=tmp_path/'calibrated')
    clock = setup[3]
    for i in range(11):
        clock.t = 1000+i*.32
        p.on_session(session())
        obs = observation(setup, oid=i+1)
        obs.state[17] += .0065
        assert p.act(obs) is None
        assert p.predictions == 0 and p.returned_chunks == 0
    assert p.parked_adapter.ready
    clock.t = 1003.52
    p.on_session(session())
    obs = observation(setup, oid=12)
    obs.state[17] += .0065
    out = p.act(obs)
    np.testing.assert_array_equal(out.actions, p.predictor.raw[:1,:8])
    assert p.predictions == 1 and p.returned_chunks == 1
    assert p.act(observation(setup, oid=13)) is None
    report = json.loads((p.output/'parked_calibration.json').read_text())
    assert report['ready'] and report['session_id'] == 'trial1'
    assert (p.output/'parked_calibration_ready.npz').exists()


def test_calibrated_shadow_still_cannot_publish(setup, tmp_path):
    p = policy_for(setup, action_rows=1, confirm_parked_setup=True, output=tmp_path/'calibrated_shadow')
    for i in range(12):
        setup[3].t = 1000+i*.32
        p.on_session(session())
        assert p.act(observation(setup, oid=i+1)) is None
    assert p.predictions == 1 and p.returned_chunks == 0


def test_calibration_refusal_or_log_failure_disarms(setup, tmp_path, monkeypatch):
    p = policy_for(setup, execute=True, action_rows=1, confirm_parked_setup=True,
                   output=tmp_path/'calibration_failure')
    def fail_save(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(np, 'savez_compressed', fail_save)
    with pytest.raises(OSError, match='full disk'):
        p.act(observation(setup))
    assert p.disarmed and p.returned_chunks == 0


def test_calibration_does_not_extend_authorization_deadline(setup, tmp_path):
    p = policy_for(setup, execute=True, action_rows=1, confirm_parked_setup=True,
                   output=tmp_path/'expired_calibration', max_seconds=2)
    for i in range(12):
        setup[3].t = 1000+i*.32
        p.on_session(session())
        assert p.act(observation(setup, oid=i+1)) is None
    assert p.returned_chunks == 0 and not p.parked_adapter.ready


def test_calibration_commissioning_configuration_refuses_broader_execution(setup, tmp_path):
    for kwargs in [{'action_rows':8}, {'action_rows':1,'max_chunks':2},
                   {'action_rows':1,'shadow_constant_diagnostic':True}]:
        with pytest.raises(ValueError):
            policy_for(setup, execute=True, confirm_parked_setup=True, output=tmp_path/'config', **kwargs)
    with pytest.raises(ValueError, match='audit output'):
        policy_for(setup, action_rows=1, confirm_parked_setup=True)


def test_calibrated_wire_no_action_until_ready_then_one_row_only(setup, tmp_path):
    p, node, stub, event = wire_setup(setup, action_rows=1, confirm_parked_setup=True,
                                    output=tmp_path/'calibrated_wire')
    for i in range(13):
        # Slightly > .32 prevents float rounding from hitting the rate limiter.
        setup[3].t = 1000+i*.33
        node._on_input(stub, json_event('session', session()))
        for camera in p.spec.camera_keys:
            node._frames[camera] = ImageFrame(i+1, setup[3](), np.zeros((480,640,3), np.uint8))
        event['metadata'].update(t_mono=setup[3](), observation_id=i+1, image_seq=[i+1,i+1])
        node._on_input(stub, event)
        actions = [(a,v,m) for a,v,m in stub.sent if a.startswith('action')]
        if i < 11:
            assert not actions
        else:
            assert len(actions) == 1
    topic, values, meta = actions[0]
    assert topic == 'action_grip' and len(values) == 8 and meta['chunk_len'] == 1
    assert meta['chunk_dt_s'] == .04 and validate_action_metadata(meta) == []
    assert p.predictions == 1 and p.returned_chunks == 1


def test_hemisphere_mapping_is_sign_invariant_and_preserves_rotation(setup):
    _, legacy, _, _ = setup
    state = np.tile(legacy, (10,1))
    rng = np.random.default_rng(14)
    for offset in (12,28):
        q = Rotation.random(10, random_state=rng).as_quat()[:,[3,0,1,2]]
        state[:,offset:offset+4] = q
    aligned = align_quaternion_hemisphere(state, legacy)
    reversed_state = state.copy()
    reversed_state[:,12:16] *= -1
    reversed_state[:,28:32] *= -1
    np.testing.assert_array_equal(aligned, align_quaternion_hemisphere(reversed_state, legacy))
    for offset in (12,28):
        qa,qb = [Rotation.from_quat(x[:,offset:offset+4][:,[1,2,3,0]]) for x in (state,aligned)]
        np.testing.assert_allclose((qa.inv()*qb).magnitude(), 0, atol=1e-12)
    others = np.ones(32,bool)
    others[12:16] = others[28:32] = False
    np.testing.assert_array_equal(aligned[:,others], state[:,others])


def test_equivalent_raw_quaternion_sign_is_not_hardware_movement(setup):
    adapter = parked_adapter(setup)
    for i in range(11):
        state = setup[2].copy()
        state[12:16] *= -1 if i % 2 else 1
        state[28:32] *= -1 if i % 2 else 1
        assert adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1000+i*.32) is None
    assert adapter.ready
    state[28:32] *= -1
    assert adapter.adapt(state, session_id='trial1', epoch='epoch1', t_mono=1003.52) is not None


def test_hemisphere_option_requires_calibrated_path(setup):
    with pytest.raises(ValueError, match='session-calibrated'):
        policy_for(setup, align_to_training_hemisphere=True)


def test_calibrated_input_uses_training_hemisphere_without_changing_actions(setup, tmp_path):
    checkpoint, legacy, current, _ = setup
    # A valid negative-w training hemisphere: the converter canonicalizes it to
    # positive w, then the model adapter should select the recorded equivalent.
    legacy[12:16] = [-1,0,0,0]
    normalizer = next(checkpoint.glob('*normalizer_processor.safetensors'))
    from safetensors.numpy import load_file
    stats = load_file(str(normalizer))
    stats['observation.state.mean'][12:16] = legacy[12:16]
    save_file(stats, str(normalizer))
    p = policy_for(setup, execute=True, action_rows=1, confirm_parked_setup=True,
                   align_to_training_hemisphere=True, output=tmp_path/'hemisphere_policy')
    for i in range(12):
        setup[3].t = 1000+i*.32
        p.on_session(session())
        result = p.act(observation(setup, oid=i+1))
        if i < 11:
            assert result is None
    np.testing.assert_array_equal(p.predictor.last_state[12:16], -current_tcp_to_training_state(current)[12:16])
    assert float(p.predictor.last_state[12:16]@legacy[12:16]) > .999
    np.testing.assert_array_equal(result.actions, p.predictor.raw[:1,:8])


def grant_policy(setup, tmp_path):
    p = policy_for(setup, action_rows=1, confirm_parked_setup=True,
                   align_to_training_hemisphere=True, output=tmp_path/'grant_policy')
    path = tmp_path/'authorization.json'
    p.await_one_row_authorization(path)
    grant = {'purpose':'SUPERVISED_ONE_ROW', 'session_id':'trial1', 'epoch':'epoch1',
             'policy_id':p.policy_id, 'action_rows':1, 'max_chunks':1,
             'expires_t_mono':setup[3]()+10}
    return p, path, grant


def test_session_grant_requires_fresh_matching_scope_and_only_one_row(setup, tmp_path):
    p,path,grant = grant_policy(setup,tmp_path)
    assert p.execute_session is None and not p._may_publish()
    path.write_text(json.dumps(grant))
    for i in range(12):
        setup[3].t = 1000+i*.32
        p.on_session(session())
        out = p.act(observation(setup,oid=i+1))
        if i < 11:
            assert out is None
    assert p.execute_session == 'trial1' and p.authorization_consumed
    np.testing.assert_array_equal(out.actions,p.predictor.raw[:1,:8])
    assert p.returned_chunks == 1
    path.write_text(json.dumps({**grant,'expires_t_mono':2000}))
    assert p.act(observation(setup,oid=13)) is None
    assert p.returned_chunks == 1 and p.deadline == 1010


@pytest.mark.parametrize('key,value', [('session_id','wrong'),('epoch','wrong'),
                                     ('policy_id','wrong'),('max_chunks',2),('action_rows',8),
                                     ('expires_t_mono',999),('expires_t_mono',1020)])
def test_wrong_or_expired_grant_disarms_without_action(setup,tmp_path,key,value):
    p,path,grant=grant_policy(setup,tmp_path)
    path.write_text(json.dumps({**grant,key:value}))
    with pytest.raises(ValueError):
        p.act(observation(setup))
    assert p.disarmed and p.returned_chunks == 0 and p.authorization_consumed
    path.write_text(json.dumps(grant))
    assert p.act(observation(setup,oid=2)) is None


def test_existing_grant_cannot_be_reused_by_a_new_client(setup,tmp_path):
    p,path,grant=grant_policy(setup,tmp_path)
    path.write_text(json.dumps(grant))
    other=policy_for(setup,action_rows=1,confirm_parked_setup=True,
                     align_to_training_hemisphere=True,output=tmp_path/'other')
    with pytest.raises(ValueError,match='new one-row'):
        other.await_one_row_authorization(path)


def test_grant_cannot_enable_old_diagnostic_or_multirow_mode(setup,tmp_path):
    for options in [{'shadow_constant_diagnostic':True}, {},
                    {'confirm_parked_setup':True,'action_rows':8,'align_to_training_hemisphere':True}]:
        p=policy_for(setup,output=tmp_path/str(len(list(tmp_path.iterdir()))),**options)
        with pytest.raises(ValueError,match='one-row'):
            p.await_one_row_authorization(tmp_path/'grant')


def test_wire_cadence_is_native_25hz_but_model_call_rate_remains_capped(setup):
    p,node,stub,event = wire_setup(setup,execute=False)
    announce=node.spec_announce_kwargs()
    assert announce['rate_hz'] == 25 and announce['chunk_dt_s'] == .04
    assert node.max_inference_rate_hz == 3.125
    node._on_input(stub,event)
    assert p.predictions == 1
    for delay in [.05,.10,.15,.20,.25,.30]:
        setup[3].t = 1000+delay
        event['metadata'].update(t_mono=setup[3](),observation_id=int(delay*100)+2)
        for camera in p.spec.camera_keys:
            node._frames[camera]=ImageFrame(7,setup[3](),np.zeros((480,640,3),np.uint8))
        node._on_input(stub,event)
    assert p.predictions == 1
    with pytest.raises(ValueError,match='native action-row'):
        GuardedPolicyNode(p,rate_hz=3.125)


def bounded_grant_policy(setup, tmp_path):
    p = policy_for(setup, action_rows=8, max_chunks=3, bounded_rollout=True,
                   confirm_parked_setup=True, align_to_training_hemisphere=True,
                   output=tmp_path/'bounded_policy')
    path = tmp_path/'bounded_authorization.json'
    p.await_bounded_rollout_authorization(path)
    grant = {'purpose':'SUPERVISED_BOUNDED_ROLLOUT', 'session_id':'trial1',
             'epoch':'epoch1', 'policy_id':p.policy_id, 'action_rows':8,
             'max_chunks':3, 'expires_t_mono':1010,
             'total_translation_path_m':.01, 'total_rotation_path_rad':.15,
             'total_gripper_change':.1}
    return p, path, grant


def calibrate_bounded_policy(p, setup):
    for i in range(11):
        setup[3].t = 1000+i*.32
        p.on_session(session())
        assert p.act(observation(setup, oid=i+1)) is None
        assert p.predictions == 0 and p.returned_chunks == 0
    assert p.parked_adapter.ready


def test_bounded_grant_only_three_unmodified_chunks_after_calibration(setup, tmp_path):
    p,path,grant = bounded_grant_policy(setup,tmp_path)
    assert p.execute_session is None and not p._may_publish()
    path.write_text(json.dumps(grant))
    calibrate_bounded_policy(p,setup)
    for i in range(3):
        setup[3].t = 1003.52+i*.32
        p.on_session(session())
        out = p.act(observation(setup,oid=12+i))
        np.testing.assert_array_equal(out.actions,p.predictor.raw[:,:8])
    assert p.returned_chunks == 3
    assert p.translation_path_used == pytest.approx(.0048)
    assert p.act(observation(setup,oid=15)) is None
    assert p.returned_chunks == 3


@pytest.mark.parametrize('columns,per_row', [(slice(0,3),.0005),(slice(3,6),.008)])
def test_bounded_rollout_refuses_cumulative_path_before_exceeding_budget(setup,tmp_path,columns,per_row):
    p,path,grant = bounded_grant_policy(setup,tmp_path)
    p.predictor.raw[:,:6] = 0
    p.predictor.raw[:,columns.start] = per_row
    path.write_text(json.dumps(grant))
    calibrate_bounded_policy(p,setup)
    for i in range(2):
        setup[3].t = 1003.52+i*.32
        p.on_session(session())
        assert p.act(observation(setup,oid=12+i)) is not None
    setup[3].t = 1004.16
    p.on_session(session())
    with pytest.raises(ValueError,match='budget'):
        p.act(observation(setup,oid=14))
    assert p.disarmed and p.returned_chunks == 2
    assert p.translation_path_used <= .01 and p.rotation_path_used <= .15


def test_bounded_rollout_gripper_limit_is_relative_to_first_publication(setup,tmp_path):
    p,path,grant = bounded_grant_policy(setup,tmp_path)
    path.write_text(json.dumps(grant))
    calibrate_bounded_policy(p,setup)
    setup[3].t = 1003.52
    p.on_session(session())
    p.predictor.raw[:,6] = .95
    assert p.act(observation(setup,oid=12)) is not None
    setup[3].t = 1003.84
    p.on_session(session())
    obs = observation(setup,oid=13)
    obs.state[7] = .95
    p.predictor.raw[:,6] = .89  # small per-step change, but >0.1 from initial 1.0
    with pytest.raises(ValueError,match='budget'):
        p.act(obs)
    assert p.disarmed and p.returned_chunks == 1


@pytest.mark.parametrize('key,value', [('total_translation_path_m',.02),
    ('total_rotation_path_rad',.3), ('total_gripper_change',.2),
    ('action_rows',1), ('max_chunks',4), ('purpose','SUPERVISED_ONE_ROW')])
def test_bounded_grant_must_match_exact_limits(setup,tmp_path,key,value):
    p,path,grant = bounded_grant_policy(setup,tmp_path)
    path.write_text(json.dumps({**grant,key:value}))
    with pytest.raises(ValueError):
        p.act(observation(setup))
    assert p.disarmed and p.returned_chunks == 0


def approach_grant_policy(setup,tmp_path):
    p=policy_for(setup,action_rows=8,max_chunks=8,bounded_rollout=True,
                 supervised_approach=True,confirm_parked_setup=True,
                 align_to_training_hemisphere=True,output=tmp_path/'approach')
    path=tmp_path/'approach_grant.json'
    p.await_bounded_rollout_authorization(path)
    grant={'purpose':'SUPERVISED_APPROACH','session_id':'trial1','epoch':'epoch1',
           'policy_id':p.policy_id,'action_rows':8,'max_chunks':8,'expires_t_mono':1010,
           'total_translation_path_m':.08,'prefix_translation_path_m':.04,
           'row_translation_m':.006,'total_rotation_path_rad':.15,'total_gripper_change':.1}
    return p,path,grant


def test_approach_requires_distinct_grant_and_does_not_expand_commissioning(setup,tmp_path):
    with pytest.raises(ValueError,match='explicit bounded'):
        policy_for(setup,supervised_approach=True)
    p,path,grant=approach_grant_policy(setup,tmp_path)
    path.write_text(json.dumps({**grant,'purpose':'SUPERVISED_BOUNDED_ROLLOUT'}))
    with pytest.raises(ValueError):p.act(observation(setup))
    assert p.returned_chunks==0 and p.disarmed


def test_approach_can_execute_native_chunks_but_stops_at_total_path_limit(setup,tmp_path):
    p,path,grant=approach_grant_policy(setup,tmp_path)
    p.predictor.raw[:,0]=.003  # 24 mm per eight-row prefix
    path.write_text(json.dumps(grant))
    calibrate_bounded_policy(p,setup)
    for i in range(3):
        setup[3].t=1003.52+i*.32
        p.on_session(session())
        out=p.act(observation(setup,oid=12+i))
        np.testing.assert_array_equal(out.actions,p.predictor.raw[:,:8])
    assert p.translation_path_used==pytest.approx(.072)
    setup[3].t=1004.48;p.on_session(session())
    with pytest.raises(ValueError,match='budget'):p.act(observation(setup,oid=15))
    assert p.returned_chunks==3 and p.disarmed
    saved=np.load(p.output/'refused_observation_000015.npz')
    np.testing.assert_array_equal(saved['proposed_actions'],p.predictor.raw)


@pytest.mark.parametrize('row,grip',[(.0051,1.),(.007,1.),(.001,.8)])
def test_approach_keeps_prefix_row_and_gripper_limits(setup,tmp_path,row,grip):
    p,path,grant=approach_grant_policy(setup,tmp_path)
    p.predictor.raw[:,0]=row;p.predictor.raw[:,6]=grip
    if row==.007:p.predictor.raw[1:,0]=0 # small prefix but oversized individual row
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    setup[3].t=1003.52;p.on_session(session())
    with pytest.raises(ValueError,match='budget'):p.act(observation(setup,oid=12))
    assert p.returned_chunks==0 and p.disarmed


def test_approach_requires_open_gripper_before_first_action(setup,tmp_path):
    p,path,grant=approach_grant_policy(setup,tmp_path)
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    setup[3].t=1003.52;p.on_session(session())
    obs=observation(setup,oid=12);obs.state[7]=.5;p.predictor.raw[:,6]=.5
    with pytest.raises(ValueError,match='initially open'):p.act(obs)
    assert p.returned_chunks==0 and p.disarmed


def slow_approach_policy(setup,tmp_path):
    p=policy_for(setup,action_rows=8,max_chunks=3,bounded_rollout=True,
                 supervised_approach=True,confirm_parked_setup=True,
                 align_to_training_hemisphere=True,action_row_dt_s=.2,
                 output=tmp_path/'slow_approach')
    path=tmp_path/'slow_grant.json';p.await_bounded_rollout_authorization(path)
    grant={'purpose':'SUPERVISED_APPROACH','session_id':'trial1','epoch':'epoch1',
           'policy_id':p.policy_id,'action_rows':8,'max_chunks':3,'expires_t_mono':1010,
           'chunk_dt_s':.2,'total_translation_path_m':.08,'prefix_translation_path_m':.04,
           'row_translation_m':.006,'total_rotation_path_rad':.15,'total_gripper_change':.1}
    return p,path,grant


def test_slow_clock_preserves_rows_and_monitors_parked_state_between_predictions(setup,tmp_path):
    p,path,grant=slow_approach_policy(setup,tmp_path)
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    node=GuardedPolicyNode(p,rate_hz=5,chunk_dt_s=.2)
    assert node.spec_announce_kwargs()['rate_hz']==5
    assert node.spec_announce_kwargs()['chunk_dt_s']==.2
    setup[3].t=1003.52;p.on_session(session())
    out=p.act(observation(setup,oid=12))
    np.testing.assert_array_equal(out.actions,p.predictor.raw[:,:8])
    for i in range(1,5):
        setup[3].t=1003.52+i*.33;p.on_session(session())
        assert p.act(observation(setup,oid=12+i)) is None
    assert p.predictions==1 and p.parked_adapter.last_t==pytest.approx(1004.84)
    setup[3].t=1005.17;p.on_session(session())
    out=p.act(observation(setup,oid=17))
    np.testing.assert_array_equal(out.actions,p.predictor.raw[:,:8])
    assert p.predictions==2
    setup[3].t=1005.5;p.on_session(session());obs=observation(setup,oid=18)
    obs.state[16]+=.01
    with pytest.raises(ValueError,match='Parked hardware changed'):p.act(obs)
    assert p.disarmed and p.returned_chunks==2


def test_slow_clock_requires_matching_grant_and_wire_rate(setup,tmp_path):
    p,path,grant=slow_approach_policy(setup,tmp_path)
    with pytest.raises(ValueError,match='native action-row'):GuardedPolicyNode(p,rate_hz=25)
    path.write_text(json.dumps({**grant,'chunk_dt_s':.04}))
    with pytest.raises(ValueError):p.act(observation(setup))
    assert p.returned_chunks==0 and p.disarmed


@pytest.mark.parametrize('options',[{'action_row_dt_s':.1}, {'action_row_dt_s':.2},
    {'action_row_dt_s':.2,'supervised_approach':True,'bounded_rollout':True,'max_chunks':4}])
def test_unreviewed_execution_clock_refused(setup,tmp_path,options):
    with pytest.raises(ValueError):policy_for(setup,output=tmp_path/'invalid_clock',**options)


def extended_approach_policy(setup,tmp_path):
    p=policy_for(setup,action_rows=8,max_chunks=12,max_seconds=30,bounded_rollout=True,
                 supervised_approach=True,extended_approach=True,confirm_parked_setup=True,
                 align_to_training_hemisphere=True,action_row_dt_s=.2,output=tmp_path/'extended')
    path=tmp_path/'extended_grant.json';p.await_bounded_rollout_authorization(path)
    grant={'purpose':'SUPERVISED_EXTENDED_APPROACH','session_id':'trial1','epoch':'epoch1',
           'policy_id':p.policy_id,'action_rows':8,'max_chunks':12,'expires_t_mono':1030,
           'chunk_dt_s':.2,'total_translation_path_m':.25,'prefix_translation_path_m':.04,
           'row_translation_m':.006,'total_rotation_path_rad':.15,'total_gripper_change':.1}
    return p,path,grant


def test_extended_approach_has_distinct_grant_and_finite_cumulative_budget(setup,tmp_path):
    p,path,grant=extended_approach_policy(setup,tmp_path)
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    p.predictor.raw[:,0]=.003  # 24 mm/chunk; only ten fit into 250 mm total
    for i in range(60):
        setup[3].t=1003.52+i*.33;p.on_session(session())
        try:out=p.act(observation(setup,oid=12+i))
        except ValueError as exc:
            assert 'budget' in str(exc)
            break
    else:pytest.fail('Extended trial did not stop at the finite budget')
    assert p.returned_chunks==10 and p.translation_path_used==pytest.approx(.24)
    assert p.disarmed


@pytest.mark.parametrize('key,value',[('purpose','SUPERVISED_APPROACH'),
    ('total_translation_path_m',.5),('total_gripper_change',1.),('expires_t_mono',1040)])
def test_extended_approach_cannot_change_scope_via_grant(setup,tmp_path,key,value):
    p,path,grant=extended_approach_policy(setup,tmp_path)
    path.write_text(json.dumps({**grant,key:value}))
    with pytest.raises(ValueError):p.act(observation(setup))
    assert p.disarmed and p.returned_chunks==0


def test_extended_approach_cannot_use_native_clock(setup,tmp_path):
    with pytest.raises(ValueError,match='slow approach clock'):
        policy_for(setup,supervised_approach=True,extended_approach=True,bounded_rollout=True,
                   output=tmp_path/'bad_extended')


def grasp_stage_policy(setup, tmp_path):
    p = policy_for(setup, action_rows=8, max_chunks=12, max_seconds=30,
                   bounded_rollout=True, supervised_grasp=True, action_row_dt_s=.2,
                   confirm_parked_setup=True, align_to_training_hemisphere=True,
                   output=tmp_path/'grasp_stage')
    path = tmp_path/'grasp_grant.json'
    p.await_bounded_rollout_authorization(path)
    grant = {'purpose':'SUPERVISED_GRASP_STAGE', 'session_id':'trial1', 'epoch':'epoch1',
             'policy_id':p.policy_id, 'action_rows':8, 'max_chunks':12,
             'expires_t_mono':1030, 'chunk_dt_s':.2, 'total_translation_path_m':.25,
             'prefix_translation_path_m':.04, 'row_translation_m':.006,
             'total_rotation_path_rad':.15, 'first_gripper_target_change':.1,
             'row_gripper_target_change':.06, 'prefix_gripper_target_path':.45,
             'total_gripper_target_path':1.}
    return p, path, grant


@pytest.mark.parametrize('kwargs', [
    {'bounded_rollout':False}, {'supervised_approach':True}, {'extended_approach':True},
    {'action_row_dt_s':.04}, {'max_chunks':13}, {'max_seconds':31},
])
def test_grasp_stage_requires_distinct_finite_slow_profile(setup, tmp_path, kwargs):
    defaults = dict(action_rows=8, max_chunks=12, max_seconds=30, bounded_rollout=True,
                    supervised_grasp=True, action_row_dt_s=.2, confirm_parked_setup=True,
                    align_to_training_hemisphere=True, output=tmp_path/'bad_grasp')
    with pytest.raises(ValueError):
        policy_for(setup, **(defaults | kwargs))


@pytest.mark.parametrize('key,value', [
    ('purpose','SUPERVISED_EXTENDED_APPROACH'), ('session_id','different'),
    ('epoch','restarted'), ('chunk_dt_s',.04), ('total_translation_path_m',.5),
    ('total_rotation_path_rad',.3), ('row_translation_m',.01),
    ('first_gripper_target_change',1.), ('row_gripper_target_change',1.),
    ('prefix_gripper_target_path',1.), ('total_gripper_target_path',2.),
    ('expires_t_mono',1040),
])
def test_grasp_grant_cannot_expand_or_change_scope(setup, tmp_path, key, value):
    p, path, grant = grasp_stage_policy(setup, tmp_path)
    path.write_text(json.dumps(grant | {key:value}))
    with pytest.raises(ValueError):
        p.act(observation(setup))
    assert p.disarmed and p.authorization_consumed and p.returned_chunks == 0
    path.write_text(json.dumps(grant))
    assert p.act(observation(setup, oid=2)) is None  # no edited grant can rearm


def test_grasp_stage_can_continue_from_partial_opening_with_unmodified_actions(setup, tmp_path):
    setup[2][7] = .87
    p, path, grant = grasp_stage_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    targets = [np.linspace(.84,.49,8), np.linspace(.44,.09,8), np.linspace(.06,0,8)]
    published = []
    for i in range(40):
        setup[3].t = 1003.52+i*.33; p.on_session(session())
        p.predictor.raw[:,6] = targets[len(published)]
        out = p.act(observation(setup, oid=12+i))
        if out is not None:
            np.testing.assert_array_equal(out.actions, p.predictor.raw[:,:8])
            published.append(out)
            setup[2][7] = out.actions[-1,6]
            if len(published) == 3:
                break
    assert len(published) == 3 and not p.disarmed
    assert p.last_gripper_target == 0
    assert p.gripper_path_used == pytest.approx(.87)
    assert p.translation_path_used == pytest.approx(.0048)
    assert p.rotation_path_used == 0
    assert p.initial_gripper == pytest.approx(.87)
    p.reset()
    assert p.disarmed and not p._may_publish()


@pytest.mark.parametrize('kind', ['first_jump', 'row_jump', 'prefix_path', 'pose_row',
                                 'pose_prefix', 'rotation', 'out_of_range', 'rail'])
def test_grasp_stage_refuses_oversize_prediction_without_clipping(setup, tmp_path, kind):
    setup[2][7] = .87
    p, path, grant = grasp_stage_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    p.predictor.raw[:,6] = .87
    if kind == 'first_jump': p.predictor.raw[:,6] = .70
    if kind == 'row_jump': p.predictor.raw[4:,6] = .78
    if kind == 'prefix_path': p.predictor.raw[:,6] = np.linspace(.78,.374,8)
    if kind == 'pose_row': p.predictor.raw[0,0] = .007
    if kind == 'pose_prefix': p.predictor.raw[:,0] = .0051
    if kind == 'rotation': p.predictor.raw[:,3] = .02
    if kind == 'out_of_range': p.predictor.raw[:,6] = 1.01
    if kind == 'rail': p.predictor.raw[0,7] = .001
    setup[3].t = 1003.52; p.on_session(session())
    with pytest.raises(ValueError): p.act(observation(setup, oid=12))
    assert p.disarmed and p.returned_chunks == 0 and p.gripper_path_used == 0
    with np.load(p.output/'refused_observation_000012.npz') as saved:
        np.testing.assert_array_equal(saved['proposed_actions'], p.predictor.raw)


def test_grasp_stage_counts_gripper_reversals_across_chunks(setup, tmp_path):
    setup[2][7] = .87
    p, path, grant = grasp_stage_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    targets = [np.linspace(.8,.45,8), np.linspace(.45,.8,8), np.linspace(.8,.45,8)]
    for i in range(40):
        setup[3].t = 1003.52+i*.33; p.on_session(session())
        p.predictor.raw[:,6] = targets[p.returned_chunks]
        try:
            out = p.act(observation(setup, oid=12+i))
        except ValueError as exc:
            assert 'gripper budget' in str(exc)
            break
        if out is not None: setup[2][7] = out.actions[-1,6]
    else: pytest.fail('Cumulative gripper motion was not refused')
    assert p.disarmed and p.returned_chunks == 2
    assert p.gripper_path_used == pytest.approx(.77)


def test_grasp_stage_keeps_extended_translation_budget(setup, tmp_path):
    p, path, grant = grasp_stage_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    p.predictor.raw[:,0] = .003
    for i in range(60):
        setup[3].t = 1003.52+i*.33; p.on_session(session())
        try: p.act(observation(setup, oid=12+i))
        except ValueError as exc:
            assert 'budget' in str(exc)
            break
    else: pytest.fail('Grasp stage exceeded its finite pose budget')
    assert p.returned_chunks == 10 and p.translation_path_used == pytest.approx(.24)
    assert p.disarmed


def drawer_task_policy(setup, tmp_path, row_dt=.2, slew=False):
    from apollo_task_guard import DRAWER_TASK_LIMITS
    setup[2][9:12] = [.65, 0., .05]
    seconds = 550 if row_dt == .4 else 300
    p = policy_for(setup, action_rows=8, max_chunks=150, max_seconds=seconds,
                   bounded_rollout=True, supervised_drawer_task=True, action_row_dt_s=row_dt,
                   gripper_reference_slew=slew,
                   confirm_parked_setup=True, align_to_training_hemisphere=True,
                   output=tmp_path/'drawer_task')
    path = tmp_path/'drawer_task_grant.json'; p.await_bounded_rollout_authorization(path)
    grant = {'purpose':'SUPERVISED_DRAWER_TASK_SLEW' if slew else 'SUPERVISED_DRAWER_TASK', 'session_id':'trial1', 'epoch':'epoch1',
             'policy_id':p.policy_id, 'action_rows':8, 'max_chunks':150,
             'expires_t_mono':1000+seconds, 'chunk_dt_s':row_dt, **DRAWER_TASK_LIMITS}
    if slew: grant.update(gripper_reference_slew=True, gripper_reference_max_step=.06)
    return p, path, grant


def test_slew_changes_only_gripper_and_saves_raw_prediction(setup, tmp_path):
    p, path, grant = drawer_task_policy(setup,tmp_path,row_dt=.4,slew=True)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p,setup)
    p.predictor.raw[:,6] = 0  # Abrupt raw closure becomes a .06-per-row reference.
    setup[3].t=1003.52; p.on_session(session())
    result=p.act(observation(setup,oid=12))
    np.testing.assert_array_equal(result.actions[:,:6],p.predictor.raw[:,:6])
    np.testing.assert_array_equal(result.actions[:,7],p.predictor.raw[:,7])
    np.testing.assert_allclose(result.actions[:,6],1-.06*np.arange(1,9),atol=1e-7)
    row=json.loads((p.output/'predictions.jsonl').read_text().splitlines()[-1])
    assert row['gripper_reference_slew']
    assert np.all(np.array(row['raw_actions_grip'])[:,6]==0)
    np.testing.assert_array_equal(row['actions_grip'],result.actions)
    assert p.gripper_path_used == pytest.approx(.48)
    setup[3].t += .33; p.on_session(session())
    assert p.act(observation(setup,oid=13)) is None
    assert p.task_guard.gripper_response_since is None  # row 0, not future close.


@pytest.mark.parametrize('key,value', [
    ('purpose','SUPERVISED_DRAWER_TASK'), ('gripper_reference_slew',False),
    ('gripper_reference_max_step',.07), ('chunk_dt_s',.2), ('expires_t_mono',1551),
])
def test_slow_slew_grant_is_separate_and_cannot_expand(setup,tmp_path,key,value):
    p,path,grant=drawer_task_policy(setup,tmp_path,row_dt=.4,slew=True)
    path.write_text(json.dumps(grant | {key:value}))
    with pytest.raises(ValueError): p.act(observation(setup))
    assert p.disarmed and p.returned_chunks==0


@pytest.mark.parametrize('targets,previous', [([np.nan],.5),([1.1],.5),([0],-1),([],1)])
def test_slew_does_not_hide_invalid_predictions(targets,previous):
    from apollo_gripper_reference import slew_gripper_targets
    with pytest.raises(ValueError): slew_gripper_targets(targets,previous)


def test_slew_refusal022_regression_and_float32_step():
    from apollo_gripper_reference import slew_gripper_targets
    from apollo_task_guard import DrawerTaskGuard
    raw=np.array([70.6192856,65.1472931,62.3614349,61.4895020,
                  61.7592812,62.4531937,63.2574692,64.0392914])/84
    previous=67.520520687/84
    with pytest.raises(ValueError): DrawerTaskGuard.gripper_path(raw,66/84,previous,1.1724106)
    sent=slew_gripper_targets(raw,previous).astype(np.float32)
    assert max(abs(np.diff(np.r_[previous,sent]))) <= .0600001
    DrawerTaskGuard.gripper_path(sent,66/84,previous,1.1724106)
    # The larger spike remains refused without the explicit filter.
    with pytest.raises(ValueError): DrawerTaskGuard.gripper_path([.9,.835],.9,.9,0)


def test_reference_watchdog_uses_active_not_future_row_and_still_stops():
    from apollo_gripper_reference import PublishedGripperReference
    from apollo_task_guard import DrawerTaskGuard
    ref=PublishedGripperReference();ref.record([1,1,1,1,1,1,1,.3],10,.4)
    assert ref.at(9.9) is None
    assert ref.at(12.7)==1 and ref.at(12.9)==.3
    state=np.zeros(32);state[9:12]=[.65,0,.05];state[12]=1;state[7]=.8
    guard=DrawerTaskGuard()
    for t in (10.1,11.1,12.1,12.9,13.5,15.8): guard.observe(state,t,0,ref.at(t))
    with pytest.raises(ValueError,match='wide gripper'):
        guard.observe(state,15.91,0,ref.at(15.91))
    ref.record([.7,.8],16,.4)
    assert ref.at(15.99)==.3 and ref.at(16.01)==.7 and ref.at(20)==.8


def test_lamp_gripper_watchdog_treats_close_stall_as_contact_but_checks_opening_progress():
    from apollo_task_guard import LampTaskGuard
    state=np.zeros(32); state[9:12]=[.63,0,.1]; state[12]=1
    guard=LampTaskGuard()
    for t in (0,6,12):
        state[7]=.75; guard.observe(state,t,0,.1)
    guard=LampTaskGuard()
    for t,opening in [(0,.2),(4,.25),(8,.30),(12,.35)]:
        state[7]=opening; guard.observe(state,t,0,.9)
    state[7]=.35; guard.observe(state,16.9,0,.9)
    with pytest.raises(ValueError,match='opening'):
        guard.observe(state,17.01,0,.9)


def test_slow_slew_retains_workspace_guard(setup,tmp_path):
    p,path,grant=drawer_task_policy(setup,tmp_path,row_dt=.4,slew=True)
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    setup[3].t=1003.52;setup[2][11]=-.11;p.on_session(session())
    with pytest.raises(ValueError,match='workspace'):p.act(observation(setup,oid=12))
    assert p.disarmed and p.returned_chunks==0


def test_slow_slew_full_finite_horizon_and_chunk_boundary(setup,tmp_path):
    p,path,grant=drawer_task_policy(setup,tmp_path,row_dt=.4,slew=True)
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    last_time=None
    for i in range(1660):
        setup[3].t=1003.52+i*.33;p.on_session(session())
        result=p.act(observation(setup,oid=12+i))
        if result is not None:
            np.testing.assert_array_equal(result.actions,p.predictor.raw[:,:8])
            if last_time is not None: assert setup[3].t-last_time >= 3.2
            last_time=setup[3].t
        if p.returned_chunks==150: break
    assert p.returned_chunks==150 and not p.disarmed
    setup[3].t+=.33;p.on_session(session())
    assert p.act(observation(setup,oid=13+i)) is None
    assert p.deadline==1550


def test_slew_keeps_random_references_bounded_across_chunks():
    from apollo_gripper_reference import slew_gripper_targets
    rng=np.random.default_rng(20260913)
    previous=.96
    for _ in range(100):
        sent=slew_gripper_targets(rng.random(8),previous).astype(np.float32)
        assert np.max(abs(np.diff(np.r_[previous,sent]))) <= .0600001
        assert np.all((sent>=0)&(sent<=1))
        previous=float(sent[-1])


@pytest.mark.parametrize('kwargs', [
    {'bounded_rollout':False}, {'supervised_grasp':True}, {'supervised_approach':True},
    {'extended_approach':True}, {'action_row_dt_s':.04}, {'action_rows':1},
    {'max_chunks':151}, {'max_seconds':301}, {'execute':True},
])
def test_drawer_task_cannot_use_unreviewed_profile(setup, tmp_path, kwargs):
    defaults = dict(action_rows=8, max_chunks=150, max_seconds=300, bounded_rollout=True,
                    supervised_drawer_task=True, action_row_dt_s=.2, confirm_parked_setup=True,
                    align_to_training_hemisphere=True, output=tmp_path/'bad_task')
    with pytest.raises(ValueError): policy_for(setup, **(defaults | kwargs))


@pytest.mark.parametrize('key,value', [
    ('purpose','SUPERVISED_GRASP_STAGE'), ('session_id','not-ours'), ('epoch','new-runtime'),
    ('tcp_min_m',[.5,-.2,-.2]), ('row_translation_m',.01), ('total_translation_path_m',3.),
    ('total_rotation_path_rad',3.), ('orientation_excursion_rad',1.),
    ('gripper_response_timeout_s',30.), ('boundary_gripper_target_change',1.),
    ('total_gripper_target_path',10.), ('motion_stall_window_s',30.), ('expires_t_mono',1301),
])
def test_drawer_task_grant_cannot_expand_limits(setup, tmp_path, key, value):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    path.write_text(json.dumps(grant | {key:value}))
    with pytest.raises(ValueError): p.act(observation(setup))
    assert p.disarmed and p.returned_chunks == 0
    path.write_text(json.dumps(grant))
    assert p.act(observation(setup, oid=2)) is None


def test_drawer_task_supports_full_finite_horizon_without_rearming(setup, tmp_path):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    p.predictor.raw[:,3] = .0005
    for i in range(900):
        setup[3].t = 1003.52+i*.33; p.on_session(session())
        result = p.act(observation(setup, oid=12+i))
        if result is not None:
            np.testing.assert_array_equal(result.actions, p.predictor.raw[:,:8])
        if p.returned_chunks == 150:
            break
    else: pytest.fail('Finite task horizon never completed')
    assert p.returned_chunks == 150 and p.rotation_path_used == pytest.approx(.6)
    setup[3].t += .33; p.on_session(session())
    assert p.act(observation(setup, oid=13+i)) is None
    p.reset(); assert p.disarmed


@pytest.mark.parametrize('counter,value', [('translation_path_used',1.999),('rotation_path_used',1.499)])
def test_drawer_task_keeps_cumulative_motion_limits(setup, tmp_path, counter, value):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    p.predictor.raw[:,3] = .0005
    setattr(p, counter, value)
    setup[3].t=1003.52; p.on_session(session())
    with pytest.raises(ValueError, match='budget'): p.act(observation(setup, oid=12))
    assert p.returned_chunks == 0 and p.disarmed


def test_drawer_task_refuses_workspace_excursion_even_if_net_delta_cancels(setup, tmp_path):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    setup[2][9] = .769
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    p.predictor.raw[:,:3] = 0
    p.predictor.raw[0,0] = .003; p.predictor.raw[1,0] = -.003
    setup[3].t=1003.52; p.on_session(session())
    with pytest.raises(ValueError, match='workspace'): p.act(observation(setup, oid=12))
    assert p.returned_chunks == 0 and p.disarmed


def test_drawer_task_watchdog_runs_between_slow_predictions(setup, tmp_path):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    setup[3].t=1003.52; p.on_session(session())
    assert p.act(observation(setup, oid=12)) is not None
    setup[3].t += .33; p.on_session(session()); setup[2][11] = -.11
    with pytest.raises(ValueError, match='workspace'): p.act(observation(setup, oid=13))
    assert p.returned_chunks == 1 and p.disarmed


def test_drawer_task_deadline_does_not_renew_on_hold(setup, tmp_path):
    p, path, grant = drawer_task_policy(setup, tmp_path)
    path.write_text(json.dumps(grant)); calibrate_bounded_policy(p, setup)
    setup[3].t=1300.01; p.on_session(session())
    assert p.act(observation(setup, oid=12)) is None
    assert p.returned_chunks == 0 and p.deadline == 1300


def test_task_gripper_allows_normal_knob_contact_but_not_wide_nonresponse():
    from apollo_task_guard import DrawerTaskGuard
    state = np.zeros(32); state[9:12] = [.65,0,.05]; state[12] = 1; state[7] = .2
    guard = DrawerTaskGuard()
    for t in np.arange(0,5,.33): guard.observe(state,t,0.,0.)
    assert guard.gripper_response_since is None  # not forced to reach zero through an object
    assert guard.gripper_path(np.zeros(8),.2,.05,.8) == pytest.approx(.05)
    state[7] = .8
    guard.observe(state,6.,0.,.3)
    with pytest.raises(ValueError,match='wide gripper'): guard.observe(state,9.01,0.,.3)


def test_task_gripper_response_watch_resets_after_actual_response():
    from apollo_task_guard import DrawerTaskGuard
    state = np.zeros(32); state[9:12] = [.65,0,.05]; state[12] = 1; state[7] = .8
    guard = DrawerTaskGuard(); guard.observe(state,0.,0.,.3)
    state[7] = .5; guard.observe(state,2.,0.,.3)
    assert guard.gripper_response_since is None
    state[7] = .8; guard.observe(state,4.,0.,.3); guard.observe(state,6.,0.,.3)
    with pytest.raises(ValueError,match='wide gripper'): guard.observe(state,7.01,0.,.3)


def test_task_guard_rejects_stationary_arm_despite_continued_requests():
    from apollo_task_guard import DrawerTaskGuard
    state = np.zeros(32); state[9:12] = [.65,0,.05]; state[12] = 1; state[7] = .8
    guard = DrawerTaskGuard()
    for i in range(10): guard.observe(state,i*.5,i*.005,None)
    with pytest.raises(ValueError,match='without measured progress'):
        guard.observe(state,5.01,.051,None)


def test_task_guard_orientation_limit_is_relative_to_first_observation():
    from apollo_task_guard import DrawerTaskGuard
    state = np.zeros(32); state[9:12] = [.65,0,.05]; state[12] = 1; state[7] = .8
    guard = DrawerTaskGuard(); guard.observe(state,0.,0.,None)
    q = Rotation.from_rotvec([0,.31,0]).as_quat(); state[12:16] = q[[3,0,1,2]]
    with pytest.raises(ValueError,match='orientation excursion'): guard.observe(state,1.,0.,None)


def test_gripper_probe_has_zero_pose_and_rail_deltas_and_distinct_identity(setup,tmp_path):
    from run_apollo_gripper_check import make_policy
    path=tmp_path/'probe_grant.json'
    p=make_policy(setup[0],tmp_path/'gripper_probe',path,setup[3]);p.on_session(session())
    grant={'purpose':'SUPERVISED_GRIPPER_CHECK','session_id':'trial1','epoch':'epoch1',
           'policy_id':p.policy_id,'action_rows':1,'max_chunks':1,'expires_t_mono':1010,
           'chunk_dt_s':.04}
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    setup[3].t=1003.52;p.on_session(session())
    result=p.act(observation(setup,oid=12))
    assert p.policy_id=='apollo_gripper_response_check_20260912'
    assert result.actions.shape==(1,8)
    np.testing.assert_array_equal(result.actions[0,:6],np.zeros(6))
    assert result.actions[0,6]==pytest.approx(.92) and result.actions[0,7]==0
    assert p.returned_chunks==1 and p.translation_path_used==0
    assert p.act(observation(setup,oid=13)) is None


def test_gripper_probe_refuses_nonopen_or_invalid_input():
    from run_apollo_gripper_check import GripperCheckPredictor
    predictor=GripperCheckPredictor()
    for value in (.5,float('nan'),1.1):
        s=np.zeros(32);s[7]=value
        with pytest.raises(ValueError):predictor.predict_chunk(s,None,None)


def test_gripper_hold_never_accumulates_closure():
    from run_apollo_gripper_check import GripperCheckPredictor
    predictor=GripperCheckPredictor()
    for measured in (.976190476,.94,.896190476):
        state=np.zeros(32);state[7]=measured
        out=predictor.predict_chunk(state,None,None)
        np.testing.assert_allclose(out[:,6],.896190476,atol=1e-7)
        np.testing.assert_array_equal(out[:,:6],np.zeros((8,6)))
        assert np.all(out[:,7:14]==0) and np.all(out[:,15]==0)


def test_gripper_reopening_is_small_fixed_target_with_zero_pose():
    from run_apollo_gripper_check import GripperCheckPredictor
    predictor=GripperCheckPredictor('open')
    for measured in (.88,.92,.96):
        state=np.zeros(32);state[7]=measured
        out=predictor.predict_chunk(state,None,None)
        np.testing.assert_allclose(out[:,6],.96,atol=1e-7)
        np.testing.assert_array_equal(out[:,:6],np.zeros((8,6)))
    for invalid in (.7,.99,float('nan')):
        predictor.reset();state[7]=invalid
        with pytest.raises(ValueError):predictor.predict_chunk(state,None,None)


def test_gripper_reopening_has_distinct_identity_and_authorization(setup,tmp_path):
    from run_apollo_gripper_check import make_policy
    p=make_policy(setup[0],tmp_path/'gripper_open',tmp_path/'open_grant',setup[3],
                  repeat_hold=True,direction='open')
    assert p.policy_id=='apollo_gripper_open_check_20260912'
    assert p.authorization_purpose=='SUPERVISED_GRIPPER_OPEN_HOLD_CHECK'
    assert p.max_chunks==3 and p.chunk_len==8 and p.chunk_dt_s==.04


def test_gripper_hold_is_fixed_target_three_chunks_and_one_fresh_grant(setup,tmp_path):
    from run_apollo_gripper_check import make_policy
    path=tmp_path/'hold_grant.json'
    p=make_policy(setup[0],tmp_path/'gripper_hold',path,setup[3],repeat_hold=True)
    p.on_session(session())
    grant={'purpose':'SUPERVISED_GRIPPER_HOLD_CHECK','session_id':'trial1','epoch':'epoch1',
           'policy_id':p.policy_id,'action_rows':8,'max_chunks':3,'expires_t_mono':1010,
           'chunk_dt_s':.04,'total_translation_path_m':.01,
           'total_rotation_path_rad':.15,'total_gripper_change':.1}
    path.write_text(json.dumps(grant));calibrate_bounded_policy(p,setup)
    for i,measured in enumerate((1.,.95,.92)):
        setup[3].t=1003.52+i*.33;p.on_session(session())
        obs=observation(setup,oid=12+i);obs.state[7]=measured
        out=p.act(obs)
        assert out.actions.shape==(8,8)
        np.testing.assert_allclose(out.actions[:,6],.92)
        np.testing.assert_array_equal(out.actions[:,:6],np.zeros((8,6)))
    assert p.returned_chunks==3 and p.translation_path_used==0
    assert p.act(observation(setup,oid=15)) is None
    p.reset();assert p.disarmed
    assert p.act(observation(setup,oid=16)) is None
