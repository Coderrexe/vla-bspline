"""SmolVLA Apollo policy plugin. Shadow mode is the default; no hardware imports.

Actual publication uses the pinned reference mavis_policy_node transport and the
existing runtime's gated executor. Enabling publication requires a specific live
session ID and a bounded budget. This module never creates or changes a session.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from mavis_policy_node.types import PolicyOutput, PolicySpec

from apollo_legacy_state import (
    CheckpointStateAdapter, GRIP_ACTION_NAMES, STATE_NAMES, current_tcp_to_training_state,
)
from apollo_parked_state import SessionParkedStateAdapter
from apollo_task_guard import (
    DRAWER_TASK_LIMITS, LAMP_TASK_LIMITS, LAMP_TASK_ROW_DT_S,
    DrawerTaskGuard, LampTaskGuard,
)
from apollo_gripper_reference import PublishedGripperReference, slew_gripper_targets


class ApolloDoraPolicy:
    chunk_len = 8
    chunk_dt_s = .04  # recorded delta commands are per 25 Hz sample, NOT per inference
    loader = 'custom'

    def __init__(self, checkpoint, *, device='cuda:0', execute_session=None,
                 max_chunks=1, max_seconds=10, action_rows=1, output=None, predictor=None,
                 clock=time.monotonic, shadow_constant_diagnostic=False,
                 confirm_parked_setup=False, align_to_training_hemisphere=False,
                 bounded_rollout=False, supervised_approach=False, action_row_dt_s=.04,
                 extended_approach=False, supervised_grasp=False, supervised_drawer_task=False,
                 supervised_lamp_task=False,
                 gripper_reference_slew=False, lamp_slow_clock=False):
        if shadow_constant_diagnostic and execute_session is not None:
            raise ValueError('Constant-feature diagnostic is shadow only; motion is forbidden')
        self.shadow_constant_diagnostic = bool(shadow_constant_diagnostic)
        if align_to_training_hemisphere and not confirm_parked_setup:
            raise ValueError('Hemisphere correction requires the session-calibrated input path')
        if confirm_parked_setup and shadow_constant_diagnostic:
            raise ValueError('Calibrated parked inputs and the unguarded shadow diagnostic are distinct modes')
        self.bounded_rollout = bool(bounded_rollout)
        self.supervised_approach = bool(supervised_approach)
        self.extended_approach = bool(extended_approach)
        self.supervised_grasp = bool(supervised_grasp)
        self.supervised_drawer_task = bool(supervised_drawer_task)
        self.supervised_lamp_task = bool(supervised_lamp_task)
        # Full lamp execution uses the same executor speed cap as the verified
        # successful absolute demonstration replay.  The learned row clock
        # remains independently slowed to 400 ms.
        self.session_speed_scale = .6 if supervised_lamp_task else .1
        full_task = self.supervised_drawer_task or self.supervised_lamp_task
        if self.supervised_drawer_task and self.supervised_lamp_task:
            raise ValueError('Choose one full-task envelope')
        self.gripper_reference_slew = bool(gripper_reference_slew)
        self.gripper_reference_max_step = (.05 if supervised_lamp_task else .06)
        if lamp_slow_clock and (not supervised_approach or not bounded_rollout
                or action_row_dt_s != .4 or max_chunks > 3 or action_rows != 8
                or full_task or supervised_grasp or extended_approach):
            raise ValueError('Lamp slow clock requires the unchanged three-chunk approach profile')
        if gripper_reference_slew and not full_task:
            raise ValueError('Gripper reference slew requires a supervised full-task profile')
        if supervised_drawer_task and (supervised_approach or extended_approach or supervised_grasp
                or not bounded_rollout or action_row_dt_s not in (.2,.4) or action_rows != 8
                or execute_session is not None):
            raise ValueError('Drawer task requires its own fresh bounded grant and eight 200 or 400 ms rows')
        if supervised_lamp_task and (supervised_approach or extended_approach or supervised_grasp
                or not bounded_rollout or action_row_dt_s not in (.04, LAMP_TASK_ROW_DT_S)
                or action_rows != 8
                or execute_session is not None):
            raise ValueError('Lamp task requires a fresh bounded grant and eight native-delta or 15 Hz absolute rows')
        if supervised_grasp and (supervised_approach or extended_approach
                                 or not bounded_rollout or action_row_dt_s != .2):
            raise ValueError('Grasp stage requires its own bounded profile and 200 ms rows')
        if extended_approach and (not supervised_approach or action_row_dt_s != .2):
            raise ValueError('Extended approach requires the reviewed slow approach clock')
        if action_row_dt_s not in (.04,LAMP_TASK_ROW_DT_S,.2,.4) or (action_row_dt_s == .4 and not (full_task or lamp_slow_clock)):
            raise ValueError('400 ms rows require a supervised full-task or reviewed lamp profile')
        if action_row_dt_s != .04 and (not (supervised_approach or supervised_grasp or full_task)
                or max_chunks > (100 if supervised_lamp_task else 150 if supervised_drawer_task
                                  else 12 if extended_approach or supervised_grasp else 3)):
            raise ValueError('Slow execution requires the supervised approach profile, at most three chunks')
        self.chunk_dt_s = float(action_row_dt_s)
        self.last_prediction_started = float('-inf')
        if supervised_approach and not bounded_rollout:
            raise ValueError('Supervised approach requires an explicit bounded-rollout grant')
        self.chunk_cap = (100 if supervised_lamp_task else 150 if supervised_drawer_task else
                          12 if extended_approach or supervised_grasp else (8 if supervised_approach else 3))
        self.authorization_window = (360 if supervised_lamp_task else
            (550 if action_row_dt_s == .4 else 300) if supervised_drawer_task else
            30 if extended_approach or supervised_grasp else 10)
        if lamp_slow_clock:
            self.authorization_window = 20
        self.translation_budget = (LAMP_TASK_LIMITS['total_translation_path_m'] if supervised_lamp_task
            else 2. if supervised_drawer_task else .25 if extended_approach or supervised_grasp
            else (.08 if supervised_approach else .01))
        self.rotation_budget = (LAMP_TASK_LIMITS['total_rotation_path_rad'] if supervised_lamp_task
            else 1.5 if supervised_drawer_task else .15)
        self.prefix_translation_budget = (LAMP_TASK_LIMITS['prefix_translation_path_m']
            if supervised_lamp_task else .04 if supervised_approach or supervised_grasp or supervised_drawer_task else .01)
        self.row_translation_budget = (LAMP_TASK_LIMITS['row_translation_m']
            if supervised_lamp_task else .006)
        if bounded_rollout and (not confirm_parked_setup or not align_to_training_hemisphere
                                or not 1 <= max_chunks <= self.chunk_cap or max_seconds > self.authorization_window):
            raise ValueError('Bounded rollout requires calibrated inputs and a finite short-trial budget')
        if (confirm_parked_setup and execute_session is not None and not bounded_rollout
                and (max_chunks != 1 or action_rows != 1)):
            raise ValueError('Parked calibration commissioning permits only one predicted row')
        if confirm_parked_setup and output is None:
            raise ValueError('Parked calibration requires an audit output directory')
        if not 1 <= max_chunks <= 400 or not 0 < max_seconds <= (self.authorization_window if full_task else 120):
            raise ValueError('Execution budget must be bounded')
        if execute_session is not None and not str(execute_session).strip():
            raise ValueError('An explicit nonempty session ID is required')
        if not isinstance(action_rows, int) or not 1 <= action_rows <= 8:
            raise ValueError('Use only a prefix of 1 to 8 predicted action rows')
        self.chunk_len = action_rows
        self.checkpoint = Path(checkpoint)
        if predictor is None:
            from apollo_predictor import ApolloPredictor
            predictor = ApolloPredictor(checkpoint, device=device)
        self.predictor = predictor
        if supervised_drawer_task and getattr(predictor, 'task', None) != 'Drawer Assembling':
            raise ValueError('This task envelope is only for the drawer-assembling checkpoint')
        if supervised_lamp_task and getattr(predictor, 'task', None) != 'Lamp Assembling':
            raise ValueError('This task envelope is only for the lamp-assembling checkpoint')
        self.state_adapter = CheckpointStateAdapter(checkpoint)
        self.parked_adapter = (SessionParkedStateAdapter(
            self.state_adapter, operator_confirmed=True,
            align_to_training_hemisphere=align_to_training_hemisphere)
                               if confirm_parked_setup else None)
        self.device, self.clock = device, clock
        self.spec = PolicySpec('delta_ee', 'arm_base:grip', GRIP_ACTION_NAMES, STATE_NAMES,
                               ['view_wrist', 'grip_wrist'], 1, arms=['grip'],
                               action_frames={'grip': 'arm_base:grip'})
        self.policy_id = f'vla_{self.checkpoint.name}_20260912'
        self.execute_session = execute_session
        self.deadline = clock()+max_seconds
        self.max_chunks = max_chunks
        self.returned_chunks = 0
        self.predictions = 0
        self.disarmed = False
        self.session = None
        self.session_received_at = float('-inf')
        self.last_observation_id = 0
        self.last_status = 'SHADOW: predictions only, no action publication'
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
        self.detail = f'{self.state_adapter.pose_convention} observations; grip only; shadow by default'
        self.authorization_file = None
        self.authorization_consumed = False
        self.authorization_purpose = 'SUPERVISED_ONE_ROW'
        self.translation_path_used = 0.
        self.rotation_path_used = 0.
        self.initial_gripper = None
        self.gripper_path_used = 0.
        self.last_gripper_target = None
        self.task_guard = (LampTaskGuard() if supervised_lamp_task else
                           DrawerTaskGuard() if supervised_drawer_task else None)
        self.gripper_reference = PublishedGripperReference()
        self.last_chunk_published_at = float('-inf')
        self.lamp_gripper_phase = 'open'

    def await_one_row_authorization(self, path):
        """Opt in to ONE future session identified by our supervised helper.

        Called before attaching. A fresh, exclusive file is created only after
        the helper receives its own session ID. The policy consumes it on the
        inference thread; no cross-thread mutation or automatic re-arming.
        """
        path = Path(path)
        if (self.execute_session is not None or self.shadow_constant_diagnostic
                or self.parked_adapter is None or not self.parked_adapter.align_to_training_hemisphere
                or self.max_chunks != 1 or self.chunk_len != 1 or self.output is None
                or self.authorization_file is not None or path.exists()):
            raise ValueError('Requires a new one-row authorization file and calibrated hemisphere-corrected policy')
        self.authorization_file = path

    def await_bounded_rollout_authorization(self, path):
        """Opt in to one finite, session-specific rollout.

        Existing commissioning/approach profiles retain their limits. The distinct
        grasp stage keeps the extended approach's pose/clock limits but allows
        gradual gripper motion. It is not full-task or unattended authorization.
        """
        path = Path(path)
        if (not self.bounded_rollout or self.execute_session is not None
                or self.shadow_constant_diagnostic or self.parked_adapter is None
                or not self.parked_adapter.align_to_training_hemisphere
                or not 1 <= self.max_chunks <= self.chunk_cap or self.output is None
                or self.authorization_file is not None or path.exists()):
            raise ValueError('Requires a new bounded-rollout authorization and calibrated inputs')
        self.authorization_file = path
        self.authorization_purpose = ('SUPERVISED_LAMP_TASK_SLEW' if self.gripper_reference_slew and self.supervised_lamp_task else
                                      'SUPERVISED_DRAWER_TASK_SLEW' if self.gripper_reference_slew else
                                      'SUPERVISED_LAMP_TASK' if self.supervised_lamp_task else
                                      'SUPERVISED_DRAWER_TASK' if self.supervised_drawer_task else
                                      'SUPERVISED_GRASP_STAGE' if self.supervised_grasp else
                                      'SUPERVISED_EXTENDED_APPROACH' if self.extended_approach else
                                      'SUPERVISED_APPROACH' if self.supervised_approach
                                      else 'SUPERVISED_BOUNDED_ROLLOUT')

    def _consume_one_row_authorization(self):
        path = self.authorization_file
        if path is None or self.authorization_consumed or not path.exists():
            return
        # Mark consumed even on failure. No edited/replaced file can re-arm us.
        self.authorization_consumed = True
        try:
            if path.stat().st_size > 4096:
                raise ValueError('Oversized authorization')
            grant = json.loads(path.read_text())
            if (grant.get('purpose') != self.authorization_purpose
                    or grant.get('session_id') != self.session.get('session_id')
                    or grant.get('epoch') != self.session.get('epoch')
                    or grant.get('policy_id') != self.policy_id
                    or grant.get('chunk_dt_s',.04) != self.chunk_dt_s
                    or grant.get('action_rows') != self.chunk_len or grant.get('max_chunks') != self.max_chunks):
                raise ValueError('Authorization does not match the current one-row policy/session')
            if self.bounded_rollout and (grant.get('total_translation_path_m') != self.translation_budget
                                         or grant.get('total_rotation_path_rad') != self.rotation_budget
                                         or (not (self.supervised_grasp or self.supervised_drawer_task or self.supervised_lamp_task) and
                                             grant.get('total_gripper_change') != .1)):
                raise ValueError('Bounded rollout must retain the total commissioning displacement limits')
            if (self.supervised_approach or self.supervised_grasp) and (grant.get('prefix_translation_path_m') != .04
                                             or grant.get('row_translation_m') != .006):
                raise ValueError('Approach grant must preserve the prefix and row displacement limits')
            if self.supervised_grasp and any(grant.get(k) != v for k, v in {
                    'first_gripper_target_change': .1, 'row_gripper_target_change': .06,
                    'prefix_gripper_target_path': .45, 'total_gripper_target_path': 1.,
            }.items()):
                raise ValueError('Grasp grant must preserve the gradual gripper-motion limits')
            if self.supervised_drawer_task and any(grant.get(k) != v for k, v in DRAWER_TASK_LIMITS.items()):
                raise ValueError('Drawer-task grant must match every reviewed limit')
            if self.supervised_lamp_task and any(grant.get(k) != v for k, v in LAMP_TASK_LIMITS.items()):
                raise ValueError('Lamp-task grant must match every reviewed limit')
            if (grant.get('gripper_reference_slew', False) is not self.gripper_reference_slew
                    or (self.gripper_reference_slew and
                        grant.get('gripper_reference_max_step') != self.gripper_reference_max_step)):
                raise ValueError('Grant must explicitly match gripper reference filtering')
            deadline = float(grant['expires_t_mono'])
            if not 0 < deadline-self.clock() <= self.authorization_window:
                raise ValueError('Authorization expired or exceeds the selected time limit')
            spec = self.session.get('spec') or {}
            speed = float(spec.get('speed_scale', 1))
            if (spec.get('start_from') != 'keep_current'
                    or (self.supervised_lamp_task and speed != self.session_speed_scale)
                    or (not self.supervised_lamp_task and not 0 < speed <= self.session_speed_scale)):
                raise ValueError('Authorization session speed does not match the reviewed profile')
            (self.output/'consumed_authorization.json').write_text(json.dumps(grant,indent=2)+'\n')
            self.deadline = deadline
            self.execute_session = grant['session_id']
        except Exception:
            self.disarm('invalid one-row session authorization')
            raise

    def disarm(self, reason):
        self.disarmed = True
        self.last_status = f'DISARMED: {reason}; start a new explicitly authorized trial to resume'

    def reset(self):
        self.predictor.reset()
        self.last_observation_id = 0
        if self.returned_chunks:
            self.disarm('reset after motion publication')

    def on_session(self, session):
        previous = self.session or {}
        self.session = session
        self.session_received_at = self.clock()
        if previous.get('session_id') and (
                previous['session_id'] != session.get('session_id')
                or previous.get('epoch') != session.get('epoch')
                or session.get('state') != 'running'):
            self.disarm('session stopped, changed, or runtime restarted')

    def _session_valid(self):
        s = self.session or {}
        if (s.get('state') != 'running' or not s.get('session_id')
                or self.clock()-self.session_received_at > 2.0):
            return False
        if s.get('kind') != 'hardware' or s.get('policy_source') != 'external':
            return False
        if set(s.get('arm_ids', [])) != {'grip', 'view'}:
            return False
        frames = s.get('frames', {})
        if any(frames.get(a) != f'arm_base:{a}' for a in ('grip', 'view')):
            return False
        spec = s.get('spec') or {}
        if self.execute_session:
            speed = float(spec.get('speed_scale', 1))
            if (spec.get('start_from') != 'keep_current'
                    or (self.supervised_lamp_task and speed != self.session_speed_scale)
                    or (not self.supervised_lamp_task and not 0 < speed <= self.session_speed_scale)):
                return False
        return spec.get('mode') == 'inference'

    def _execution_permitted(self):
        return (not self.shadow_constant_diagnostic
                and self.execute_session is not None and not self.disarmed
                and self._session_valid()
                and self.session['session_id'] == self.execute_session
                and self.clock() < self.deadline and self.returned_chunks < self.max_chunks)

    def _may_publish(self):
        return (self._execution_permitted()
                and (self.parked_adapter is None or (
                    self.parked_adapter.ready and not self.parked_adapter.invalid)))

    def _validate_grasp_gripper(self, targets, measured):
        """Reject abrupt or excessive targets, without modifying learned actions.

        Use measured opening for every prefix's first-target jump, and successive
        published targets for cumulative path. This prevents chunk boundaries or
        noisy reversals from concealing excessive commanded motion. It does not
        establish clearance, contact force, or grasp success.
        """
        previous = measured if self.last_gripper_target is None else self.last_gripper_target
        path = float(np.abs(np.diff(np.r_[previous, targets])).sum())
        if (not np.isfinite(measured) or not 0 <= measured <= 1
                or abs(float(targets[0])-measured) > .1
                or abs(float(targets[0])-previous) > .1
                or np.any(np.abs(np.diff(targets)) > .06)
                or path > .45 or self.gripper_path_used+path > 1.):
            raise ValueError('Prediction exceeds grasp-stage gripper budget')
        return path

    def _filter_lamp_gripper(self, raw_targets, previous, state):
        """Debounce the lamp grasp as an event while preserving gradual commands.

        A receding-horizon prediction jitters around the close target.  Counting
        every reversal as new physical travel can stop a valid contact, and an
        upward fluctuation can release the shade mid-transport.  Once closure
        begins, keep the reference monotone until the TCP reaches the demonstrated
        placement region and the model proposes reopening; then keep it monotone
        open.  Raw predictions remain in the audit log.
        """
        raw_targets = np.asarray(raw_targets, dtype=np.float64)
        previous = float(previous)
        xyz = np.asarray(state[9:12], dtype=np.float64)
        in_release_region = bool(xyz[0] >= .62 and xyz[1] >= .155 and xyz[2] >= .05)
        if self.lamp_gripper_phase == 'open':
            if float(np.min(raw_targets)) > .8:
                return np.full_like(raw_targets, previous, dtype=np.float64)
            self.lamp_gripper_phase = 'closing'
        if (self.lamp_gripper_phase in {'closing', 'closed'} and in_release_region
                and float(np.max(raw_targets)) >= .7):
            self.lamp_gripper_phase = 'opening'
        targets = slew_gripper_targets(
            raw_targets, previous, max_step=self.gripper_reference_max_step)
        if self.lamp_gripper_phase in {'closing', 'closed'}:
            targets = np.minimum.accumulate(np.r_[previous, targets])[1:]
            if float(targets[-1]) <= .35:
                self.lamp_gripper_phase = 'closed'
        else:
            targets = np.maximum.accumulate(np.r_[previous, targets])[1:]
        return targets

    def act(self, obs):
        if not self._session_valid():
            return None
        if self.disarmed:
            return None
        self._consume_one_row_authorization()
        if self.execute_session is not None and not self._execution_permitted():
            return None
        if self.disarmed:
            return None
        age = self.clock()-obs.t_mono
        if not -.02 <= age <= .25 or obs.observation_id <= self.last_observation_id:
            if self.returned_chunks:
                self.disarm('stale or repeated observation')
            return None
        self.last_observation_id = obs.observation_id
        raw = None
        commanded = None
        wire_actions = None
        gripper_prefix_path = 0.
        try:
            if self.parked_adapter is not None:
                was_ready = self.parked_adapter.ready
                state = self.parked_adapter.adapt(
                    obs.state, session_id=self.session['session_id'],
                    epoch=self.session.get('epoch'), t_mono=obs.t_mono)
                if self.parked_adapter.samples == 1 or (not was_ready and self.parked_adapter.ready):
                    label = 'ready' if self.parked_adapter.ready else 'start'
                    np.savez_compressed(self.output/f'parked_calibration_{label}.npz',
                                        current_state=obs.state,
                                        view_rgb=obs.images['view_wrist'], grip_rgb=obs.images['grip_wrist'])
                (self.output/'parked_calibration.json').write_text(
                    json.dumps(self.parked_adapter.report(), indent=2)+'\n')
                if state is None:
                    self.last_status = ('PARKED REFERENCE READY; next fresh observation eligible for prediction'
                                        if self.parked_adapter.ready else
                                        f'CALIBRATING: {self.parked_adapter.samples} samples; no prediction or action')
                    return None
            elif self.shadow_constant_diagnostic:
                # Hypothesis test only, with publication independently disabled
                # above. Do not use this branch to authorize learned movement.
                state = self.state_adapter.to_training_state(obs.state)
                difference = state[self.state_adapter.mask]-self.state_adapter.reference[self.state_adapter.mask]
                state[self.state_adapter.mask] = self.state_adapter.reference[self.state_adapter.mask]
            else:
                state = self.state_adapter(obs.state)
            if self.task_guard is not None:
                target = (self.gripper_reference.at(obs.t_mono) if self.chunk_dt_s == .4
                          else self.last_gripper_target)
                self.task_guard.observe(obs.state, obs.t_mono, self.translation_path_used,
                                        target)
            # Continue fresh parked-state monitoring at the original callback
            # cadence while executing a slower chunk. Never reuse an old image
            # or repeat a row to fill the extended interval.
            if ((self.chunk_dt_s > .04 or self.supervised_lamp_task) and
                    self.clock()-self.last_prediction_started < 8*self.chunk_dt_s):
                return None
            # At the slow drawer clock, finish the previous published chunk
            # before starting inference. This also preserves the slew boundary.
            if self.chunk_dt_s == .4 and self.clock()-self.last_chunk_published_at < 8*self.chunk_dt_s:
                return None
            start = self.clock()
            self.last_prediction_started = start
            raw = self.predictor.predict_chunk(state, obs.images['view_wrist'], obs.images['grip_wrist'])
            if raw.shape != (8, 16) or not np.isfinite(raw).all():
                raise ValueError('Invalid action output')
            if (not np.all(raw[:, 7:14] == 0) or not np.all(raw[:, 14] == 1)
                    or not np.all(raw[:, 15] == 0) or np.any((raw[:, 6] < 0) | (raw[:, 6] > 1))):
                raise ValueError('Parked-arm/rail or gripper contract violated')
            commanded = raw.copy()
            if self.gripper_reference_slew:
                previous = float(obs.state[7]) if self.last_gripper_target is None else self.last_gripper_target
                if self.supervised_lamp_task:
                    commanded[:, 6] = self._filter_lamp_gripper(raw[:, 6], previous, obs.state)
                else:
                    commanded[:,6] = slew_gripper_targets(
                        raw[:,6], previous, max_step=self.gripper_reference_max_step)
            if self.execute_session:
                # Conservative commissioning bounds, not a collision-safety claim.
                # Pose deltas are never clipped. The explicit slew profile ONLY
                # filters gripper references; raw and sent commands are audited.
                prefix = commanded[:self.chunk_len]
                prior_translation = self.translation_path_used if self.bounded_rollout else 0.
                prior_rotation = self.rotation_path_used if self.bounded_rollout else 0.
                grip_origin = self.initial_gripper if self.initial_gripper is not None else obs.state[7]
                distances = np.linalg.norm(prefix[:, :3], axis=1)
                if self.supervised_approach and self.initial_gripper is None and obs.state[7] < .9:
                    raise ValueError('Supervised approach requires an initially open gripper (at least 0.9)')
                if (prior_translation+distances.sum() > self.translation_budget
                        or distances.sum() > self.prefix_translation_budget
                        or ((self.supervised_approach or self.supervised_grasp or self.supervised_drawer_task
                             or self.supervised_lamp_task) and distances.max() > self.row_translation_budget)
                        or prior_rotation+np.linalg.norm(prefix[:, 3:6], axis=1).sum() > self.rotation_budget
                        or (not (self.supervised_grasp or self.supervised_drawer_task or self.supervised_lamp_task) and (
                            (self.bounded_rollout and np.max(np.abs(prefix[:,6]-grip_origin)) > .1)
                            or np.max(np.abs(prefix[:, 6]-obs.state[7])) > .1))):
                    raise ValueError('Prediction exceeds first-trial displacement/gripper budget')
                if self.supervised_grasp:
                    gripper_prefix_path = self._validate_grasp_gripper(prefix[:,6], float(obs.state[7]))
                if self.task_guard is not None:
                    self.task_guard.check_prefix(obs.state, prefix)
                    gripper_prefix_path = self.task_guard.gripper_path(
                        prefix[:,6], float(obs.state[7]), self.last_gripper_target, self.gripper_path_used)
            wire_actions = self._encode_for_transport(obs, commanded[:self.chunk_len])
        except Exception as exc:
            self.disarm('invalid input or failed prediction')
            # A refused input is particularly useful for commissioning: preserve
            # it before returning control to the reference node's error handler.
            # Logging must never cause publication or replace the original error.
            if self.output:
                try:
                    if self.parked_adapter is not None:
                        (self.output/'parked_calibration.json').write_text(
                            json.dumps(self.parked_adapter.report(), indent=2)+'\n')
                    name = f'refused_observation_{obs.observation_id:06d}.npz'
                    proposed = ({'proposed_actions':raw} if isinstance(raw,np.ndarray) else {})
                    if commanded is not None:
                        proposed['candidate_commanded_actions'] = commanded
                    np.savez_compressed(
                        self.output/name, current_state=np.asarray(obs.state),
                        view_rgb=np.asarray(obs.images['view_wrist']),
                        grip_rgb=np.asarray(obs.images['grip_wrist']), **proposed)
                    with (self.output/'refusals.jsonl').open('a') as stream:
                        stream.write(json.dumps({
                            'observation_id': obs.observation_id,
                            'session_id': self.session['session_id'],
                            'state_names': STATE_NAMES,
                            'observation_age_s': self.clock()-obs.t_mono,
                            'error': repr(exc), 'input_file': name,
                            'publication_requested': False})+'\n')
                except Exception as log_error:
                    self.last_status = f'DISARMED: input/prediction failed; diagnostic save also failed: {log_error!r}'
            raise
        self.predictions += 1
        fresh = self.clock()-obs.t_mono <= .4  # runtime rejects at .5 s
        publish = fresh and self._may_publish()
        if self.execute_session and not fresh:
            self.disarm('prediction became stale during computation')
        row = {'prediction': self.predictions, 'observation_id': obs.observation_id,
               'session_id': self.session['session_id'], 'policy_id': self.policy_id,
               'compute_ms': (self.clock()-start)*1000,
               'observation_age_s': self.clock()-obs.t_mono,
               'publication_requested': publish, 'chunk_dt_s': self.chunk_dt_s,
               'execution_rows': self.chunk_len,
               'constant_feature_diagnostic': self.shadow_constant_diagnostic,
               'session_parked_calibration': self.parked_adapter is not None,
               'training_hemisphere_alignment': bool(self.parked_adapter and
                                                     self.parked_adapter.align_to_training_hemisphere),
               'bounded_rollout': self.bounded_rollout,
               'supervised_approach': self.supervised_approach,
               'extended_approach': self.extended_approach,
               'supervised_grasp': self.supervised_grasp,
               'supervised_drawer_task': self.supervised_drawer_task,
               'supervised_lamp_task': self.supervised_lamp_task,
               'gripper_reference_slew': self.gripper_reference_slew,
               'lamp_gripper_phase': self.lamp_gripper_phase if self.supervised_lamp_task else None,
               'gripper_target_path_used_before': self.gripper_path_used,
               'gripper_target_path_prefix': gripper_prefix_path,
               'translation_budget_m': self.translation_budget,
               'translation_path_used_before_m': self.translation_path_used,
               'current_state': np.asarray(obs.state).tolist(),
               'wire_action_space': self.spec.action_space,
               'wire_actions': wire_actions.tolist(),
               'actions_grip': commanded[:, :8].tolist(),
               'raw_actions_grip': raw[:, :8].tolist()}
        if self.shadow_constant_diagnostic:
            row['constant_feature_differences'] = {
                STATE_NAMES[i]: float(delta)
                for i, delta in zip(np.flatnonzero(self.state_adapter.mask), difference)}
        if self.output:
            if self.shadow_constant_diagnostic and self.predictions == 1:
                np.savez_compressed(self.output/'diagnostic_first_input.npz',
                                    current_state=obs.state, projected_training_state=state,
                                    view_rgb=obs.images['view_wrist'], grip_rgb=obs.images['grip_wrist'])
            with (self.output/'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        if not publish:
            return None  # never a zero row: zero gripper means CLOSE, not hold
        self.returned_chunks += 1
        self.translation_path_used += float(np.linalg.norm(raw[:self.chunk_len,:3],axis=1).sum())
        self.rotation_path_used += float(np.linalg.norm(raw[:self.chunk_len,3:6],axis=1).sum())
        self.gripper_path_used += gripper_prefix_path
        self.last_gripper_target = float(commanded[self.chunk_len-1,6])
        self.last_chunk_published_at = self.clock()
        self.gripper_reference.record(commanded[:self.chunk_len,6], self.last_chunk_published_at, self.chunk_dt_s)
        if self.initial_gripper is None:
            self.initial_gripper = float(obs.state[7])
        self.last_status = f'Bounded trial: {self.returned_chunks}/{self.max_chunks} chunks returned'
        self._commit_transport()
        return PolicyOutput(wire_actions, self.spec.version, obs.t_mono)

    def _encode_for_transport(self, obs, prefix):
        return prefix[:, :8].copy()

    def _commit_transport(self):
        pass

    def pop_status(self):
        status, self.last_status = self.last_status, None
        return status

    def load_weights(self, path):
        raise RuntimeError('No weight hot-swap; stop and validate the next checkpoint separately')
