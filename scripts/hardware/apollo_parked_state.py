"""Session-local input calibration; no robot, network, or action interface.

Use only after an operator approves the current parked arm/camera/rail setup.
The reference is captured AFTER controller startup, never restored from disk.
It is an input/stationarity check, not collision checking or task validation.
"""
from __future__ import annotations

import numpy as np

from apollo_legacy_state import STATE_NAMES, align_quaternion_hemisphere


class SessionParkedStateAdapter:
    """Keep real setup monitoring separate from constant model-input features.

    Eleven fresh observations spanning at least three seconds must be stationary
    before prediction is permitted. During calibration all state fields except the
    active gripper opening are checked; afterward only parked fields are checked.
    The active gripper remains a measured model input and has a separate action
    change limit. Any error permanently invalidates
    this reference. A new session/epoch needs a new, explicitly approved instance.
    The 1e-3 native-unit band is unchanged; its reference is now the approved live
    posture, not a zero-variance training feature. No actions are transformed.
    """

    def __init__(self, checkpoint_adapter, *, operator_confirmed=False, align_to_training_hemisphere=False):
        if operator_confirmed is not True:
            raise ValueError('Operator approval of the parked setup is required')
        expected = np.zeros(32, dtype=bool)
        expected[8] = True
        expected[16:] = True
        if not np.array_equal(checkpoint_adapter.mask, expected):
            raise ValueError('Only the audited 17 constant Apollo inputs may be calibrated')
        self.model_reference = checkpoint_adapter.reference.copy()
        self.checkpoint_adapter = checkpoint_adapter
        self.model_mean = checkpoint_adapter.mean.copy()
        self.align_to_training_hemisphere = bool(align_to_training_hemisphere)
        if not np.isfinite(self.model_reference).all():
            raise ValueError('Invalid checkpoint reference')
        self.mask = expected
        self.calibration_mask = np.ones(32, dtype=bool)
        self.calibration_mask[7] = False  # active gripper is not a parked feature
        self.tolerance = 1e-3
        self.session_key = None
        self.reference = None
        self.low = None
        self.high = None
        self.first_t = None
        self.last_t = None
        self.samples = 0
        self.ready = False
        self.invalid = False

    def adapt(self, current_state, *, session_id, epoch, t_mono):
        """Return model state after calibration; None during the no-action window."""
        try:
            return self._adapt(current_state, session_id=session_id, epoch=epoch, t_mono=t_mono)
        except Exception:
            self.invalid = True
            self.ready = False
            raise

    def _adapt(self, current_state, *, session_id, epoch, t_mono):
        if self.invalid:
            raise ValueError('Parked reference invalidated; no automatic recalibration')
        if not isinstance(session_id, str) or not session_id or not isinstance(epoch, str) or not epoch:
            raise ValueError('A nonempty session ID and runtime epoch are required')
        if not np.isfinite(t_mono):
            raise ValueError('Invalid calibration observation timestamp')
        key = (session_id, epoch)
        if self.session_key is not None and key != self.session_key:
            raise ValueError('Parked reference belongs to a different session or epoch')
        raw = np.asarray(current_state, dtype=np.float32)
        if raw.shape != (32,):
            raise ValueError('Calibration requires a single 32-field observation')
        training_state = self.checkpoint_adapter.to_training_state(raw)
        if np.any((raw[[7, 23]] < 0) | (raw[[7, 23]] > 1)):
            raise ValueError('Gripper opening must be a measured fraction in [0,1]')
        if self.last_t is not None and not 0 < t_mono-self.last_t <= .75:
            raise ValueError('Repeated, reversed, or interrupted calibration observations')
        self.last_t = float(t_mono)
        if self.reference is None:
            self.session_key = key
            self.reference = raw.copy()
            self.low = raw.copy()
            self.high = raw.copy()
            self.first_t = float(t_mono)

        # Raw q and -q are the same measured orientation. For stationarity only,
        # compare on the reference hemisphere; retain actual raw snapshots.
        comparison = align_quaternion_hemisphere(raw, self.reference)

        if not self.ready:
            self.low = np.minimum(self.low, comparison)
            self.high = np.maximum(self.high, comparison)
            if np.any((self.high-self.low)[self.calibration_mask] > self.tolerance):
                raise ValueError('Robot state changed during parked calibration; operator review required')
            self.samples += 1
            if self.samples >= 11 and t_mono-self.first_t >= 3.0:
                self.ready = True
            return None  # even the final calibration sample never produces an action

        deviation = np.abs(comparison-self.reference)
        failed = self.mask & (deviation > self.tolerance)
        if np.any(failed):
            names = ', '.join(STATE_NAMES[i] for i in np.flatnonzero(failed))
            raise ValueError(f'Parked hardware changed after calibration: {names}')
        training_state[self.mask] = self.model_reference[self.mask]
        if self.align_to_training_hemisphere:
            training_state = align_quaternion_hemisphere(training_state, self.model_mean)
        return training_state

    def report(self):
        """Audit only. This report cannot be loaded as execution authorization."""
        result = {'operator_confirmed': True, 'ready': self.ready, 'invalid': self.invalid,
                  'session_id': self.session_key[0] if self.session_key else None,
                  'epoch': self.session_key[1] if self.session_key else None,
                  'samples': self.samples, 'band_native_units': self.tolerance,
                  'first_t_mono': self.first_t, 'last_t_mono': self.last_t,
                  'state_names': STATE_NAMES, 'constant_indices': np.flatnonzero(self.mask).tolist(),
                  'calibration_indices': np.flatnonzero(self.calibration_mask).tolist(),
                  'reloadable_for_execution': False}
        result['training_hemisphere_alignment'] = self.align_to_training_hemisphere
        result['state_pose_convention'] = self.checkpoint_adapter.pose_convention
        if self.reference is not None:
            converted = self.checkpoint_adapter.to_training_state(self.reference)
            result.update(current_reference=self.reference.tolist(),
                          calibration_span_by_field=(self.high-self.low).tolist(),
                          model_constant_reference=self.model_reference[self.mask].tolist(),
                          converted_minus_training_constants=(converted-self.model_reference)[self.mask].tolist())
        return result
