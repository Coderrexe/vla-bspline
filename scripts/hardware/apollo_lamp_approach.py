"""Reviewed high-clearance lamp approach, not full assembly or contact control.

Reuses the tested bounded approach transport. Adds a higher TCP floor and a
review-bound starting pose; neither the drawer hold nor shared runtime changes.
"""
import json
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from apollo_dora_policy import ApolloDoraPolicy
from apollo_task_guard import LAMP_TASK_LIMITS, LAMP_TASK_ROW_DT_S, LampTaskGuard


LAMP_APPROACH_LIMITS = {
    'tcp_min_m': [.57, -.06, .10], 'tcp_max_m': [.72, .24, .30],
    # Trials 001/002 observed ~8 mm startup shifts with zero learned commands.
    # A small starting region replaces an exact-point check; the motion floor,
    # total displacement, orientation and open-gripper bounds are unchanged.
    'initial_tcp_min_m': [.65, -.03, .14],
    'initial_tcp_max_m': [.69, .01, .19],
    'orientation_excursion_rad': .15,
    'row_rotation_rad': .025, 'prefix_rotation_path_rad': .125,
}


def validate_lamp_approach_options(args):
    if (not args.supervised_lamp_approach or not args.lamp_approach_review
            or not args.supervised_approach or args.extended_approach
            or args.supervised_grasp or args.supervised_drawer_task
            or args.gripper_reference_slew or args.action_row_dt_s not in (.2, .4)
            or (args.action_row_dt_s == .4 and not args.lamp_absolute_execution)
            or args.action_rows != 8 or not 1 <= args.max_chunks <= 3):
        raise ValueError('Lamp approach requires 1–3 eight-row chunks at 200 ms, or 400 ms for absolute execution')


def stationary_rail_wire_value(measured):
    """Preserve a measured integer-mm rail through float32 + SDK truncation.

    Verified live config uses rail_flip=false. The hardware boundary truncates
    meters*1000 to an integer, so float32(0.635) would command 634, not 635 mm.
    Round the measured readback to its native grid, then choose the smallest
    float32 no lower than that value. Bias is at most one float32 ULP, <0.0001 mm.
    The existing hardware boundary clamps the 650 mm end stop.
    """
    value = float(measured)
    if not np.isfinite(value):
        raise ValueError('Invalid rail readback')
    mm = round(value*1000)
    if not 0 <= mm <= 650 or abs(value-mm/1000) > 1e-5:
        raise ValueError('Rail readback is not on the reviewed millimeter grid')
    encoded = np.float32(mm/1000)
    if float(encoded) < mm/1000:
        encoded = np.nextafter(encoded, np.float32(np.inf))
    return encoded


class LampApproachGuard:
    def __init__(self):
        self.initial_rotation = None

    @staticmethod
    def rotation(state):
        q = np.asarray(state[12:16], dtype=float)
        if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
            raise ValueError('Invalid lamp orientation')
        return Rotation.from_quat(q[[1, 2, 3, 0]])

    def check_pose(self, xyz, rotation):
        lim = LAMP_APPROACH_LIMITS
        if (not np.isfinite(xyz).all() or np.any(xyz < lim['tcp_min_m'])
                or np.any(xyz > lim['tcp_max_m'])):
            raise ValueError('Lamp approach leaves its high-clearance workspace')
        if (rotation*self.initial_rotation.inv()).magnitude() > lim['orientation_excursion_rad']:
            raise ValueError('Lamp approach exceeds orientation excursion')

    def observe(self, state, t_mono, requested_path, last_gripper_target):
        state = np.asarray(state)
        rotation = self.rotation(state)
        if self.initial_rotation is None:
            if (np.any(state[9:12] < LAMP_APPROACH_LIMITS['initial_tcp_min_m'])
                    or np.any(state[9:12] > LAMP_APPROACH_LIMITS['initial_tcp_max_m'])):
                raise ValueError('Lamp approach is outside the reviewed initial pose region')
            self.initial_rotation = rotation
        self.check_pose(state[9:12], rotation)
        if not .85 <= float(state[7]) <= 1:
            raise ValueError('Lamp approach requires an open gripper; no grasp is authorized')

    def check_prefix(self, state, actions):
        if self.initial_rotation is None:
            raise ValueError('Lamp observation must be checked before a prefix')
        position = np.asarray(state[9:12], dtype=float).copy()
        rotation = self.rotation(state)
        for row in actions:
            position += row[:3]
            rotation = Rotation.from_rotvec(row[3:6])*rotation
            self.check_pose(position, rotation)
        angles = np.linalg.norm(actions[:, 3:6], axis=1)
        if (angles.max() > LAMP_APPROACH_LIMITS['row_rotation_rad']
                or angles.sum() > LAMP_APPROACH_LIMITS['prefix_rotation_path_rad']):
            raise ValueError('Lamp approach exceeds angular step budget')
        if np.any(actions[:, 6] < .85):
            raise ValueError('Lamp approach does not authorize grasp closure')

    @staticmethod
    def gripper_path(targets, measured, last_target, path_used):
        # Base approach retains its stricter 0.1 opening-change constraint.
        previous = measured if last_target is None else last_target
        return float(np.abs(np.diff(np.r_[previous, targets])).sum())


class LampApproachPolicy(ApolloDoraPolicy):
    def __init__(self, checkpoint, *, lamp_review_id, absolute_execution=False, **kwargs):
        self.full_task = bool(kwargs.get('supervised_lamp_task'))
        if self.full_task:
            if (kwargs.get('supervised_approach') or kwargs.get('bounded_rollout') is not True
                    or kwargs.get('action_row_dt_s') !=
                        (LAMP_TASK_ROW_DT_S if absolute_execution else .04)
                    or kwargs.get('action_rows') != 8
                    or not 1 <= kwargs.get('max_chunks', 0) <= 100
                    or any(kwargs.get(k) for k in ('extended_approach', 'supervised_grasp',
                                                   'supervised_drawer_task'))):
                raise ValueError('Wrong profile for reviewed full lamp task')
        elif (kwargs.get('supervised_approach') is not True
                or kwargs.get('bounded_rollout') is not True
                or kwargs.get('action_row_dt_s') not in (.2, .4) or kwargs.get('action_rows') != 8
                or (kwargs.get('action_row_dt_s') == .4 and not absolute_execution)
                or not 1 <= kwargs.get('max_chunks', 0) <= 3
                or kwargs.get('supervised_lamp_task')
                or any(kwargs.get(k) for k in ('extended_approach', 'supervised_grasp',
                                               'supervised_drawer_task', 'gripper_reference_slew'))):
            raise ValueError('Wrong profile for reviewed lamp approach')
        super().__init__(checkpoint,
                         lamp_slow_clock=kwargs.get('action_row_dt_s') == .4 and not self.full_task,
                         **kwargs)
        if self.predictor.task != 'Lamp Assembling' or self.state_adapter.pose_convention != 'corrected_tcp':
            raise ValueError('Lamp approach requires the corrected-TCP lamp checkpoint')
        if not lamp_review_id:
            raise ValueError('Missing lamp clearance review')
        self.lamp_review_id = lamp_review_id
        self.task_guard = LampTaskGuard() if self.full_task else LampApproachGuard()
        self.lamp_limits = LAMP_TASK_LIMITS if self.full_task else LAMP_APPROACH_LIMITS
        self.absolute_motion_limits = ({
            'row_translation_m': LAMP_TASK_LIMITS['row_translation_m'],
            'prefix_translation_m': LAMP_TASK_LIMITS['prefix_translation_path_m'],
            'total_translation_m': LAMP_TASK_LIMITS['total_translation_path_m'],
            'row_rotation_rad': LAMP_TASK_LIMITS['row_rotation_rad'],
            'prefix_rotation_rad': LAMP_TASK_LIMITS['prefix_rotation_path_rad'],
            'total_rotation_rad': LAMP_TASK_LIMITS['total_rotation_path_rad'],
        } if self.full_task else {
            'row_translation_m': .006, 'prefix_translation_m': .04,
            'total_translation_m': .08, 'row_rotation_rad': .025,
            'prefix_rotation_rad': .125, 'total_rotation_rad': .15,
        })
        self.absolute_execution = absolute_execution
        self.absolute_anchor = None
        self.wire_translation_path = 0.
        self.wire_rotation_path = 0.
        self.absolute_rail_reference = None
        if absolute_execution:
            names = ['ee.x', 'ee.y', 'ee.z', 'ee.r00', 'ee.r10', 'ee.r20',
                     'ee.r01', 'ee.r11', 'ee.r21', 'gripper.pos', 'rail.pos']
            self.spec = replace(self.spec, action_space='abs_ee',
                                action_names=[f'grip_{x}' for x in names])
            self.policy_id += '_abs'

    def _encode_for_transport(self, obs, prefix):
        if not self.absolute_execution:
            return super()._encode_for_transport(obs, prefix)
        # Learned chunks anchor on their actual observation; a finite recorded
        # prefix keeps its original trajectory anchor across its three chunks.
        measured_r = self.task_guard.rotation(obs.state)
        measured_p = np.asarray(obs.state[9:12], dtype=float)
        if self.absolute_rail_reference is None:
            self.absolute_rail_reference = float(obs.state[8])
        elif abs(float(obs.state[8])-self.absolute_rail_reference) > .0002:
            raise ValueError('Absolute approach rail moved; no automatic correction')
        rail_wire = stationary_rail_wire_value(obs.state[8])
        if getattr(self.predictor, 'is_recorded_prefix', False) and self.absolute_anchor is not None:
            position, rotation = self.absolute_anchor
            position = position.copy()
        else:
            position, rotation = measured_p.copy(), measured_r
        previous_p, previous_r = measured_p, measured_r
        encoded, equivalent = [], []
        for row in prefix:
            position = position+row[:3]
            rotation = Rotation.from_rotvec(row[3:6])*rotation
            self.task_guard.check_pose(position, rotation)
            equivalent.append(np.r_[position-previous_p,
                                    (rotation*previous_r.inv()).as_rotvec(), row[6], 0.])
            matrix = rotation.as_matrix()
            encoded.append(np.r_[position, matrix[:, 0], matrix[:, 1], row[6], rail_wire])
            previous_p, previous_r = position.copy(), rotation
        equivalent = np.asarray(equivalent)
        dp = np.linalg.norm(equivalent[:, :3], axis=1)
        dr = np.linalg.norm(equivalent[:, 3:6], axis=1)
        lim = self.absolute_motion_limits
        if (dp.max() > lim['row_translation_m'] or dp.sum() > lim['prefix_translation_m']
                or self.wire_translation_path+dp.sum() > lim['total_translation_m']
                or dr.max() > lim['row_rotation_rad'] or dr.sum() > lim['prefix_rotation_rad']
                or self.wire_rotation_path+dr.sum() > lim['total_rotation_rad']):
            raise ValueError('Absolute execution exceeds the full lamp task limits' if self.full_task
                             else 'Absolute execution exceeds the unchanged approach motion limits')
        self._pending_absolute = (position.copy(), rotation, float(dp.sum()), float(dr.sum()))
        return np.asarray(encoded, dtype=np.float32)

    def _commit_transport(self):
        if self.absolute_execution:
            p, r, translation, rotation = self._pending_absolute
            self.absolute_anchor = (p, r)
            self.wire_translation_path += translation
            self.wire_rotation_path += rotation

    def await_bounded_rollout_authorization(self, path):
        super().await_bounded_rollout_authorization(path)
        self.authorization_purpose = ('SUPERVISED_LAMP_TASK_SLEW' if self.full_task
                                      and self.gripper_reference_slew else
                                      'SUPERVISED_LAMP_TASK' if self.full_task else
                                      'SUPERVISED_LAMP_APPROACH')

    def _consume_one_row_authorization(self):
        path = self.authorization_file
        if path is not None and not self.authorization_consumed and path.exists():
            try:
                if path.stat().st_size > 4096:
                    raise ValueError('Oversized lamp authorization')
                grant = json.loads(path.read_text())
                limits_key = 'lamp_task_limits' if self.full_task else 'lamp_approach_limits'
                if (grant.get('lamp_review_id') != self.lamp_review_id
                        or grant.get(limits_key) != self.lamp_limits
                        or grant.get('wire_action_space', 'delta_ee') != self.spec.action_space):
                    raise ValueError('Lamp grant differs from reviewed clearance limits')
            except Exception:
                self.authorization_consumed = True
                self.disarm('invalid lamp-specific grant')
                raise
        super()._consume_one_row_authorization()

    def _session_valid(self):
        return (super()._session_valid()
                and (self.session.get('spec') or {}).get('task') == 'Lamp Assembling')
