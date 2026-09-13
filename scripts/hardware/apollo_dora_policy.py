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


class ApolloDoraPolicy:
    chunk_len = 8
    chunk_dt_s = .04  # recorded delta commands are per 25 Hz sample, NOT per inference
    loader = 'custom'

    def __init__(self, checkpoint, *, device='cuda:0', execute_session=None,
                 max_chunks=1, max_seconds=10, action_rows=1, output=None, predictor=None,
                 clock=time.monotonic, shadow_constant_diagnostic=False,
                 confirm_parked_setup=False, align_to_training_hemisphere=False,
                 bounded_rollout=False, supervised_approach=False, action_row_dt_s=.04,
                 extended_approach=False):
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
        if extended_approach and (not supervised_approach or action_row_dt_s != .2):
            raise ValueError('Extended approach requires the reviewed slow approach clock')
        if action_row_dt_s not in (.04,.2):
            raise ValueError('Only native 40 ms or reviewed slow-approach 200 ms rows are supported')
        if action_row_dt_s != .04 and (not supervised_approach or max_chunks > (12 if extended_approach else 3)):
            raise ValueError('Slow execution requires the supervised approach profile, at most three chunks')
        self.chunk_dt_s = float(action_row_dt_s)
        self.last_prediction_started = float('-inf')
        if supervised_approach and not bounded_rollout:
            raise ValueError('Supervised approach requires an explicit bounded-rollout grant')
        chunk_cap = 12 if extended_approach else (8 if supervised_approach else 3)
        self.authorization_window = 30 if extended_approach else 10
        self.translation_budget = .25 if extended_approach else (.08 if supervised_approach else .01)
        self.prefix_translation_budget = .04 if supervised_approach else .01
        if bounded_rollout and (not confirm_parked_setup or not align_to_training_hemisphere
                                or not 1 <= max_chunks <= chunk_cap or max_seconds > self.authorization_window):
            raise ValueError('Bounded rollout requires calibrated inputs and a finite short-trial budget')
        if (confirm_parked_setup and execute_session is not None and not bounded_rollout
                and (max_chunks != 1 or action_rows != 1)):
            raise ValueError('Parked calibration commissioning permits only one predicted row')
        if confirm_parked_setup and output is None:
            raise ValueError('Parked calibration requires an audit output directory')
        if not 1 <= max_chunks <= 400 or not 0 < max_seconds <= 120:
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
        self.detail = 'legacy observation compatibility; grip only; 25 Hz rows; shadow by default'
        self.authorization_file = None
        self.authorization_consumed = False
        self.authorization_purpose = 'SUPERVISED_ONE_ROW'
        self.translation_path_used = 0.
        self.rotation_path_used = 0.
        self.initial_gripper = None

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
        """Short rollout; explicit approach profile permits up to 80 mm total.

        Commissioning remains 10 mm. Both preserve the 0.15 rad total rotation
        and 0.1 gripper-change limits; this is not a full grasp/task profile.
        """
        path = Path(path)
        if (not self.bounded_rollout or self.execute_session is not None
                or self.shadow_constant_diagnostic or self.parked_adapter is None
                or not self.parked_adapter.align_to_training_hemisphere
                or not 1 <= self.max_chunks <= (12 if self.extended_approach else
                                               (8 if self.supervised_approach else 3)) or self.output is None
                or self.authorization_file is not None or path.exists()):
            raise ValueError('Requires a new bounded-rollout authorization and calibrated inputs')
        self.authorization_file = path
        self.authorization_purpose = ('SUPERVISED_EXTENDED_APPROACH' if self.extended_approach else
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
                                         or grant.get('total_rotation_path_rad') != .15
                                         or grant.get('total_gripper_change') != .1):
                raise ValueError('Bounded rollout must retain the total commissioning displacement limits')
            if self.supervised_approach and (grant.get('prefix_translation_path_m') != .04
                                             or grant.get('row_translation_m') != .006):
                raise ValueError('Approach grant must preserve the prefix and row displacement limits')
            deadline = float(grant['expires_t_mono'])
            if not 0 < deadline-self.clock() <= self.authorization_window:
                raise ValueError('Authorization expired or exceeds ten seconds')
            spec = self.session.get('spec') or {}
            if spec.get('start_from') != 'keep_current' or not 0 < float(spec.get('speed_scale',1)) <= .1:
                raise ValueError('Authorization requires keep_current and at most 10% configured speed')
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
            if spec.get('start_from') != 'keep_current' or not 0 < speed <= .1:
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
                state = current_tcp_to_training_state(obs.state)
                difference = state[self.state_adapter.mask]-self.state_adapter.reference[self.state_adapter.mask]
                state[self.state_adapter.mask] = self.state_adapter.reference[self.state_adapter.mask]
            else:
                state = self.state_adapter(obs.state)
            # Continue fresh parked-state monitoring at the original callback
            # cadence while executing a slower chunk. Never reuse an old image
            # or repeat a row to fill the extended interval.
            if (self.chunk_dt_s > .04 and
                    self.clock()-self.last_prediction_started < 8*self.chunk_dt_s):
                return None
            start = self.clock()
            self.last_prediction_started = start
            raw = self.predictor.predict_chunk(state, obs.images['view_wrist'], obs.images['grip_wrist'])
            if raw.shape != (8, 16) or not np.isfinite(raw).all():
                raise ValueError('Invalid action output')
            if (not np.all(raw[:, 7:14] == 0) or not np.all(raw[:, 14] == 1)
                    or not np.all(raw[:, 15] == 0) or np.any((raw[:, 6] < 0) | (raw[:, 6] > 1))):
                raise ValueError('Parked-arm/rail or gripper contract violated')
            if self.execute_session:
                # Conservative commissioning bounds, not a collision-safety claim.
                # REFUSE an oversized prediction; do not clip it into a new policy.
                prefix = raw[:self.chunk_len]
                prior_translation = self.translation_path_used if self.bounded_rollout else 0.
                prior_rotation = self.rotation_path_used if self.bounded_rollout else 0.
                grip_origin = self.initial_gripper if self.initial_gripper is not None else obs.state[7]
                distances = np.linalg.norm(prefix[:, :3], axis=1)
                if self.supervised_approach and self.initial_gripper is None and obs.state[7] < .9:
                    raise ValueError('Supervised approach requires an initially open gripper (at least 0.9)')
                if (prior_translation+distances.sum() > self.translation_budget
                        or distances.sum() > self.prefix_translation_budget
                        or (self.supervised_approach and distances.max() > .006)
                        or prior_rotation+np.linalg.norm(prefix[:, 3:6], axis=1).sum() > .15
                        or (self.bounded_rollout and np.max(np.abs(prefix[:,6]-grip_origin)) > .1)
                        or np.max(np.abs(prefix[:, 6]-obs.state[7])) > .1):
                    raise ValueError('Prediction exceeds first-trial displacement/gripper budget')
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
               'translation_budget_m': self.translation_budget,
               'translation_path_used_before_m': self.translation_path_used,
               'current_state': np.asarray(obs.state).tolist(),
               'actions_grip': raw[:, :8].tolist()}
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
        if self.initial_gripper is None:
            self.initial_gripper = float(obs.state[7])
        self.last_status = f'Bounded trial: {self.returned_chunks}/{self.max_chunks} chunks returned'
        return PolicyOutput(raw[:self.chunk_len, :8].copy(), self.spec.version, obs.t_mono)

    def pop_status(self):
        status, self.last_status = self.last_status, None
        return status

    def load_weights(self, path):
        raise RuntimeError('No weight hot-swap; stop and validate the next checkpoint separately')
