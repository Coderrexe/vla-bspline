"""Finite recorded-command control; not a learned policy or full replay."""
import hashlib
from pathlib import Path

import numpy as np

from apollo_lamp_initialization import LAMP_INITIAL

REPLAY_POLICY_ID = 'vla_lamp_recorded_prefix_20260913'


class RecordedLampPrefix:
    task = 'Lamp Assembling'
    is_recorded_prefix = True

    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as f:
            state, commands = f['state'], f['delta_commands']
            if (state.shape != (521, 32) or commands.shape != (521, 16)
                    or not np.isfinite(state).all() or not np.isfinite(commands).all()
                    or not np.allclose(state[0, :7], LAMP_INITIAL['grip']['q'], atol=1e-6, rtol=0)
                    or not np.allclose(state[0, 16:23], LAMP_INITIAL['view']['q'], atol=1e-6, rtol=0)
                    or np.any(np.abs(commands[:24, 7:14]) > 1e-12)
                    or not np.all(commands[:24, 14] == 1) or not np.all(commands[:24, 15] == 0)
                    or np.any(commands[:24, 6] < .85)):
                raise ValueError('Not the reviewed high-clearance lamp demonstration prefix')
            self.commands = commands[:24].astype(np.float32, copy=True)
            # Recorded stationary-view rotation contains a 5.55e-17 rad
            # round-off term. The view arm is never an output of this client.
            self.commands[:, 7:14] = 0
        self.reset()

    def reset(self):
        self.index = 0

    def predict_chunk(self, state, view_rgb, grip_rgb):
        # Observation validation and every proposed-pose check remain in the
        # outer policy. Recorded rows deliberately do not depend on images.
        if self.index >= 24:
            raise ValueError('Recorded prefix exhausted; no wrapping or full replay')
        out = self.commands[self.index:self.index+8].copy()
        self.index += 8
        return out
