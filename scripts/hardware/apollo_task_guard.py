"""Finite drawer-task envelope and response checks; no transport or robot imports.

The recorded workspace is an input/command support check, NOT a collision model.
The operator, existing executor protections, and hardware stop remain necessary.
These limits are separate from the unchanged short commissioning profiles.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


DRAWER_TASK_LIMITS = {
    'total_translation_path_m': 2., 'total_rotation_path_rad': 1.5,
    'prefix_translation_path_m': .04, 'row_translation_m': .006,
    'prefix_rotation_path_rad': .125, 'row_rotation_rad': .025,
    'first_gripper_target_change': .1, 'boundary_gripper_target_change': .2,
    'row_gripper_target_change': .06, 'prefix_gripper_target_path': .62,
    'total_gripper_target_path': 4.5,
    'tcp_min_m': [.53, -.16, -.10], 'tcp_max_m': [.77, .15, .23],
    'orientation_excursion_rad': .30, 'gripper_response_timeout_s': 3.,
    'motion_stall_window_s': 5.,
}

# Envelope from all 52 lamp demonstrations (23,401 frames), followed by a
# small numeric/tracking margin. Recorded extrema before margin:
# xyz [0.574556, -0.044970, -0.036840] to [0.695623, 0.221537, 0.272467] m;
# orientation excursion 0.1491 rad; row/prefix translation 6.427/39.602 mm;
# row/prefix rotation 0.01751/0.11430 rad; total path 1.2265 m / 0.6219 rad.
# The successful 2026-09-13 absolute replay traversed this physical task.
# This remains a data-support envelope, not a model of the front table.
LAMP_TASK_LIMITS = {
    'total_translation_path_m': 1.5, 'total_rotation_path_rad': .75,
    'prefix_translation_path_m': .041, 'row_translation_m': .0066,
    'prefix_rotation_path_rad': .12, 'row_rotation_rad': .02,
    'first_gripper_target_change': .12, 'boundary_gripper_target_change': .25,
    'row_gripper_target_change': .065, 'prefix_gripper_target_path': .42,
    'total_gripper_target_path': 2.5,
    'tcp_min_m': [.57, -.05, -.04], 'tcp_max_m': [.70, .225, .275],
    'orientation_excursion_rad': .17, 'gripper_response_timeout_s': 5.,
    'motion_stall_window_s': 6.,
}
LAMP_TASK_ROW_DT_S = 1.0 / 15.0


class DrawerTaskGuard:
    def __init__(self):
        self.initial_rotation = None
        self.gripper_response_since = None
        self.gripper_response_kind = None
        self.gripper_response_measurement = None
        self.history = deque()

    @staticmethod
    def _rotation(state):
        q = np.asarray(state[12:16], dtype=float)
        if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
            raise ValueError('Invalid task orientation')
        return Rotation.from_quat(q[[1, 2, 3, 0]])

    def _check_pose(self, position, rotation):
        lim = DRAWER_TASK_LIMITS
        if (not np.isfinite(position).all() or np.any(position < lim['tcp_min_m'])
                or np.any(position > lim['tcp_max_m'])):
            raise ValueError('Drawer task leaves the recorded workspace envelope')
        if (rotation*self.initial_rotation.inv()).magnitude() > lim['orientation_excursion_rad']:
            raise ValueError('Drawer task exceeds its orientation excursion limit')

    def observe(self, state, t_mono, requested_path, last_gripper_target):
        state = np.asarray(state)
        rotation = self._rotation(state)
        if self.initial_rotation is None:
            self.initial_rotation = rotation
        self._check_pose(state[9:12], rotation)
        measured = float(state[7])
        if not np.isfinite(measured) or not 0 <= measured <= 1:
            raise ValueError('Invalid measured gripper opening')
        # A knob can prevent reaching zero opening. Do not reject that normal
        # grasp merely because measured opening differs from a close target.
        # A still-wide gripper after a sustained close request is different.
        kind = None
        if last_gripper_target is not None:
            if last_gripper_target < .4 and measured > .6:
                kind = 'wide gripper did not respond to closing'
            elif last_gripper_target > .85 and measured < .65:
                kind = 'gripper did not respond to opening'
        if kind != self.gripper_response_kind:
            self.gripper_response_kind = kind
            self.gripper_response_since = t_mono if kind else None
        if kind and t_mono-self.gripper_response_since >= DRAWER_TASK_LIMITS['gripper_response_timeout_s']:
            raise ValueError(f'Drawer task stopped: {kind}')
        self.history.append((float(t_mono), state[9:12].copy(), float(requested_path)))
        window = DRAWER_TASK_LIMITS['motion_stall_window_s']
        while len(self.history) > 1 and self.history[1][0] <= t_mono-window:
            self.history.popleft()
        if self.history[-1][0]-self.history[0][0] >= window:
            motion = max(np.linalg.norm(x[1]-self.history[0][1]) for x in self.history)
            if requested_path-self.history[0][2] > .03 and motion < .003:
                raise ValueError('Drawer task stopped: requested motion without measured progress')

    def check_prefix(self, state, actions):
        """Check every proposed pose, not just a potentially cancelling net delta."""
        if self.initial_rotation is None:
            raise ValueError('Task observation must be validated before a prefix')
        position = np.asarray(state[9:12], dtype=float).copy()
        rotation = self._rotation(state)
        for row in actions:
            position += row[:3]
            rotation = Rotation.from_rotvec(row[3:6])*rotation
            self._check_pose(position, rotation)
        norms = np.linalg.norm(actions[:,3:6], axis=1)
        if (norms.max() > DRAWER_TASK_LIMITS['row_rotation_rad']
                or norms.sum() > DRAWER_TASK_LIMITS['prefix_rotation_path_rad']):
            raise ValueError('Drawer task exceeds row/prefix rotation limits')

    @staticmethod
    def gripper_path(targets, measured, last_target, path_used):
        lim = DRAWER_TASK_LIMITS
        previous = measured if last_target is None else last_target
        first_limit = (lim['first_gripper_target_change'] if last_target is None
                       else lim['boundary_gripper_target_change'])
        path = float(np.abs(np.diff(np.r_[previous, targets])).sum())
        if (not np.isfinite(measured) or not 0 <= measured <= 1
                or abs(float(targets[0])-previous) > first_limit
                # float32 representation of a nominal .06 step can exceed .06
                # by a few e-8. This tolerance is < .00001 mm of opening.
                or np.any(np.abs(np.diff(targets)) > lim['row_gripper_target_change']+1e-7)
                or path > lim['prefix_gripper_target_path']
                or path_used+path > lim['total_gripper_target_path']):
            raise ValueError('Prediction exceeds drawer-task gripper budget')
        return path


class LampTaskGuard:
    """Full lamp-task checks using only the demonstrated support envelope."""

    def __init__(self):
        self.initial_rotation = None
        self.gripper_response_since = None
        self.gripper_response_kind = None
        self.gripper_response_measurement = None
        self.history = deque()

    @staticmethod
    def rotation(state):
        q = np.asarray(state[12:16], dtype=float)
        if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
            raise ValueError('Invalid lamp-task orientation')
        return Rotation.from_quat(q[[1, 2, 3, 0]])

    def check_pose(self, position, rotation):
        lim = LAMP_TASK_LIMITS
        if (not np.isfinite(position).all() or np.any(position < lim['tcp_min_m'])
                or np.any(position > lim['tcp_max_m'])):
            raise ValueError('Lamp task leaves the demonstrated workspace envelope')
        if (rotation*self.initial_rotation.inv()).magnitude() > lim['orientation_excursion_rad']:
            raise ValueError('Lamp task exceeds the demonstrated orientation envelope')

    def observe(self, state, t_mono, requested_path, last_gripper_target):
        state = np.asarray(state)
        rotation = self.rotation(state)
        if self.initial_rotation is None:
            self.initial_rotation = rotation
        self.check_pose(state[9:12], rotation)
        measured = float(state[7])
        if not np.isfinite(measured) or not 0 <= measured <= 1:
            raise ValueError('Invalid measured lamp gripper opening')
        kind = None
        if last_gripper_target is not None:
            # A close target is expected to stall on the grasped shade; that
            # stall is contact evidence, not a transport failure. Opening has
            # no analogous desired obstruction and retains a progress check.
            if last_gripper_target > .85 and measured < .65:
                kind = 'gripper did not respond to opening'
        if kind != self.gripper_response_kind:
            self.gripper_response_kind = kind
            self.gripper_response_since = t_mono if kind else None
            self.gripper_response_measurement = measured if kind else None
        elif kind:
            progress = measured-self.gripper_response_measurement
            if progress >= .03:
                self.gripper_response_since = t_mono
                self.gripper_response_measurement = measured
        if kind and t_mono-self.gripper_response_since >= LAMP_TASK_LIMITS['gripper_response_timeout_s']:
            raise ValueError(f'Lamp task stopped: {kind}')
        self.history.append((float(t_mono), state[9:12].copy(), float(requested_path)))
        window = LAMP_TASK_LIMITS['motion_stall_window_s']
        while len(self.history) > 1 and self.history[1][0] <= t_mono-window:
            self.history.popleft()
        if self.history[-1][0]-self.history[0][0] >= window:
            motion = max(np.linalg.norm(x[1]-self.history[0][1]) for x in self.history)
            if requested_path-self.history[0][2] > .03 and motion < .003:
                raise ValueError('Lamp task stopped: requested motion without measured progress')

    def check_prefix(self, state, actions):
        if self.initial_rotation is None:
            raise ValueError('Lamp task observation must be validated before a prefix')
        position = np.asarray(state[9:12], dtype=float).copy()
        rotation = self.rotation(state)
        for row in actions:
            position += row[:3]
            rotation = Rotation.from_rotvec(row[3:6])*rotation
            self.check_pose(position, rotation)
        angles = np.linalg.norm(actions[:, 3:6], axis=1)
        if (angles.max() > LAMP_TASK_LIMITS['row_rotation_rad']
                or angles.sum() > LAMP_TASK_LIMITS['prefix_rotation_path_rad']):
            raise ValueError('Lamp task exceeds row/prefix rotation limits')

    @staticmethod
    def gripper_path(targets, measured, last_target, path_used):
        lim = LAMP_TASK_LIMITS
        previous = measured if last_target is None else last_target
        first_limit = (lim['first_gripper_target_change'] if last_target is None
                       else lim['boundary_gripper_target_change'])
        path = float(np.abs(np.diff(np.r_[previous, targets])).sum())
        if (not np.isfinite(measured) or not 0 <= measured <= 1
                or abs(float(targets[0])-previous) > first_limit
                or np.any(np.abs(np.diff(targets)) > lim['row_gripper_target_change']+1e-7)
                or path > lim['prefix_gripper_target_path']
                or path_used+path > lim['total_gripper_target_path']):
            raise ValueError('Prediction exceeds lamp-task gripper budget')
        return path
