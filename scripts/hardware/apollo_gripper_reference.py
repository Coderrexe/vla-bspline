"""Opt-in command-reference smoothing and timing; no hardware or transport.

This bounds changes in the requested opening, NOT physical gripper velocity or
force. Driver settings stay unchanged. Learned pose deltas are never modified.
"""
import numpy as np


def slew_gripper_targets(targets, previous, max_step=.06):
    targets = np.asarray(targets, dtype=np.float64)
    if (targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all()
            or np.any((targets < 0) | (targets > 1))
            or not np.isfinite(previous) or not 0 <= previous <= 1
            or not np.isfinite(max_step) or not 0 < max_step <= .06):
        raise ValueError('Invalid gripper reference input')
    result = targets.copy()
    for i, desired in enumerate(targets):
        previous = float(np.clip(desired, previous-max_step, previous+max_step))
        result[i] = previous
    return result


class PublishedGripperReference:
    """Expected active row of the last published chunk, not a hardware ACK.

    The same-host runtime starts rows on receipt. Publication time is an
    approximation to that time; this monitor is not an execution guarantee.
    In particular, it must not compare measurements with a future final row.
    """
    def __init__(self):
        self.targets = None
        self.published_at = None
        self.row_dt_s = None
        self.before = None

    def record(self, targets, published_at, row_dt_s):
        targets = np.asarray(targets, dtype=float)
        if (targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all()
                or np.any((targets < 0) | (targets > 1))
                or not np.isfinite(published_at) or not np.isfinite(row_dt_s) or row_dt_s <= 0
                or (self.published_at is not None and published_at < self.published_at)):
            raise ValueError('Invalid published gripper reference')
        self.before = self.at(published_at)
        self.targets = targets.copy()
        self.published_at = float(published_at)
        self.row_dt_s = float(row_dt_s)

    def at(self, t_mono):
        if not np.isfinite(t_mono):
            raise ValueError('Invalid gripper reference timestamp')
        if self.targets is None or t_mono < self.published_at:
            return self.before
        idx = min(len(self.targets)-1, int((t_mono-self.published_at)/self.row_dt_s))
        return float(self.targets[idx])
