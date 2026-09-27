"""Observation-only compatibility for the September 12 hardware checkpoints.

Their unbackfilled recordings contain SDK FLANGE positions and quaternions built
with Rx(roll) Ry(pitch) Rz(yaw). Live Apollo now publishes corrected TCP poses.
Reproduce the old observation features; NEVER apply this conversion to actions.

Verified source: Apollo core commit 4259524 and its parent, se3.py; current
hardware units.py. The manipulation TCP is flange * (translation z=0.172 m,
rotation z=pi); the perception TCP is the flange. No robot API is imported here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

ARMS = ("grip", "view")
STATE_NAMES = [
    f"{arm}_{name}"
    for arm in ARMS
    for name in ([f"joint{i}.pos" for i in range(1, 8)]
                 + ["gripper.pos", "rail.pos", "ee.x", "ee.y", "ee.z",
                    "ee.qw", "ee.qx", "ee.qy", "ee.qz"])
]
GRIP_ACTION_NAMES = [f"grip_{name}" for name in
                     ("ee.dx", "ee.dy", "ee.dz", "ee.drx", "ee.dry", "ee.drz",
                      "gripper.pos", "rail.dpos")]


def select_state(values, names):
    """Select by name, never assume Dora's arm order equals the dataset order."""
    values = np.asarray(values)
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise ValueError("State shape or unique state_names contract violated")
    index = {name: i for i, name in enumerate(names)}
    missing = set(STATE_NAMES) - index.keys()
    if missing:
        raise ValueError(f"Missing trained state fields: {sorted(missing)}")
    return np.asarray([values[index[n]] for n in STATE_NAMES], dtype=np.float32)


def arm_stream_to_current_state(values, metadata):
    """Read-only arm_state stream -> the same current TCP fields as obs_state."""
    arms = list(metadata.get("arm_ids", []))
    layout = list(metadata.get("layout", []))
    if len(set(arms)) != len(arms) or not set(ARMS).issubset(arms):
        raise ValueError("Both unique arm IDs are required")
    if len(layout) != 32 or len(set(layout)) != 32:
        raise ValueError("Unknown arm_state layout")
    for flag in ("stale", "error_code"):
        flags = metadata.get(flag)
        if flags is None or len(flags) != len(arms) or any(flags[arms.index(a)] for a in ARMS):
            raise ValueError(f"Missing or unhealthy arm_state {flag}")
    rails = metadata.get("has_rail", [])
    if len(rails) != len(arms) or any(not rails[arms.index(a)] for a in ARMS):
        raise ValueError("Recorded setup requires both rails")
    array = np.asarray(values)
    if array.shape != (32*len(arms),):
        raise ValueError("Unexpected arm_state payload shape")
    array = array.reshape(len(arms), 32)
    wanted = ([f"q{i}" for i in range(1, 8)] + ["gripper_open_frac", "rail_pos_m"]
              + [f"ee_base.{x}" for x in ("x", "y", "z", "qw", "qx", "qy", "qz")])
    cols = [layout.index(x) for x in wanted]
    state = np.concatenate([array[arms.index(a), cols] for a in ARMS]).astype(np.float32)
    _state_batch(state)
    return state


def _state_batch(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1:] != (32,) or not np.isfinite(state).all():
        raise ValueError("Expected finite native state (..., 32)")
    for start in (0, 16):
        norms = np.linalg.norm(state[..., start+12:start+16], axis=-1)
        if np.any(np.abs(norms-1) > 1e-3):
            raise ValueError("Invalid orientation quaternion; do not infer from fallback telemetry")
    return state.reshape(-1, 32).copy(), state.shape


def _wxyz(rotation):
    xyzw = rotation.as_quat(canonical=True)
    return xyzw[:, [3, 0, 1, 2]]


def current_tcp_to_training_state(state):
    """Convert current corrected TCP state to the checkpoint's legacy features.

    Joints, grippers, rails, and all ACTIONS are unchanged. Input/output use
    STATE_NAMES order. Supports batches for recorded-data validation.
    """
    out, shape = _state_batch(state)
    for arm, start in zip(ARMS, (0, 16)):
        rotation = Rotation.from_quat(out[:, start+12:start+16][:, [1, 2, 3, 0]])
        if arm == "grip":
            rotation = rotation * Rotation.from_rotvec([0, 0, -np.pi])
            out[:, start+9:start+12] -= rotation.apply([0, 0, .172])
        # xArm SDK's physical convention is extrinsic xyz. The old reader
        # interpreted those same angles as intrinsic XYZ; reproduce that only.
        rpy = rotation.as_euler("xyz")
        legacy = Rotation.from_euler("XYZ", rpy)
        out[:, start+12:start+16] = _wxyz(legacy)
    return out.reshape(shape).astype(np.float32)


def training_to_current_tcp_state(state):
    """Inverse used to test compatibility on archived observations, not commands."""
    out, shape = _state_batch(state)
    for arm, start in zip(ARMS, (0, 16)):
        legacy = Rotation.from_quat(out[:, start+12:start+16][:, [1, 2, 3, 0]])
        rotation = Rotation.from_euler("xyz", legacy.as_euler("XYZ"))
        if arm == "grip":
            out[:, start+9:start+12] += rotation.apply([0, 0, .172])
            rotation = rotation * Rotation.from_rotvec([0, 0, np.pi])
        out[:, start+12:start+16] = _wxyz(rotation)
    return out.reshape(shape).astype(np.float32)


def align_quaternion_hemisphere(state, reference):
    """Select q or -q nearest a reference; preserve orientation and other fields.

    A positive-w convention jumps at w=0. Component-wise learned normalizers
    need a consistent hemisphere instead. This changes only representation,
    never an action, coordinate frame, or physical orientation.
    """
    out, shape = _state_batch(state)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (32,) or not np.isfinite(reference).all():
        raise ValueError('Expected a finite 32-field hemisphere reference')
    for offset in (12, 28):
        qref = reference[offset:offset+4]
        if np.linalg.norm(qref) < .5:
            raise ValueError('Reference orientations are too dispersed for mean-hemisphere alignment')
        q = out[:, offset:offset+4]
        out[:, offset:offset+4] = np.where((q@qref < 0)[:, None], -q, q)
    return out.reshape(shape).astype(np.float32)


class CheckpointStateAdapter:
    """Restore constant recorded features only within a strict compatibility band.

    Both rails and all perception-arm state fields were EXACTLY constant in
    training. Their saved std is zero: even 3e-8 of roundoff otherwise becomes a
    large normalized input. Snap those fields to the saved min==max, but REFUSE
    a changed physical setup instead of silently replacing arbitrary readings.
    The 1e-3 band is an input-compatibility check, not a robot safety limit.
    """

    def __init__(self, checkpoint, tolerance=1e-3):
        import json
        from pathlib import Path
        from safetensors.numpy import load_file
        checkpoint = Path(checkpoint)
        contract_path = checkpoint/'apollo_observation_contract.json'
        if contract_path.exists():
            contract = json.loads(contract_path.read_text())
            if (contract.get('schema_version') != 1
                    or contract.get('state_pose_convention') not in ('corrected_tcp', 'legacy_flange_intrinsic_xyz')
                    or contract.get('action_pose_convention') != 'base_frame_delta_translation_and_spatial_rotvec'
                    or contract.get('gripper') != 'absolute_open_fraction_0_to_1'):
                raise ValueError('Unknown Apollo checkpoint observation/action contract')
            self.pose_convention = contract['state_pose_convention']
            expected_legacy = self.pose_convention == 'legacy_flange_intrinsic_xyz'
            if contract.get('legacy_observation_conversion') is not expected_legacy:
                raise ValueError('Inconsistent Apollo observation conversion contract')
        else:
            # Backward compatibility is restricted to the two audited original
            # task datasets. A new task must not silently inherit the old bug.
            provenance_path = checkpoint/'dataset_provenance.json'
            if provenance_path.exists():
                task = json.loads(provenance_path.read_text()).get('task')
                if task not in ('Drawer Assembling', 'Cabinet Assembling'):
                    raise ValueError('This task requires an explicit Apollo observation contract')
            self.pose_convention = 'legacy_flange_intrinsic_xyz'
        info = json.loads((checkpoint/'apollo_interface.json').read_text())
        if info['features']['observation.state']['names'] != STATE_NAMES:
            raise ValueError('Checkpoint state ordering is not the audited Apollo interface')
        files = list(checkpoint.glob('policy_preprocessor_step_*_normalizer_processor.safetensors'))
        if len(files) != 1:
            raise ValueError('Expected exactly one saved input normalizer')
        stats = load_file(str(files[0]))
        self.reference = stats['observation.state.min'].astype(np.float32)
        self.mean = stats['observation.state.mean'].astype(np.float32)
        self.mask = ((stats['observation.state.min'] == stats['observation.state.max'])
                     & (stats['observation.state.std'] == 0))
        if self.reference.shape != (32,) or self.mask.shape != (32,) or self.mean.shape != (32,):
            raise ValueError('State statistics shape mismatch')
        if self.pose_convention == 'corrected_tcp':
            # Lamp includes native and backfilled FK records. The parked view
            # quaternion differs by float rounding (~1e-8), not physical motion.
            # Its tiny nonzero std is unsuitable for accepting live pose noise.
            # Only the physically parked channels may get this treatment. Any
            # real dataset variation on those channels requires a new interface.
            parked = np.zeros(32, dtype=bool)
            parked[8] = True; parked[16:] = True
            span = stats['observation.state.max']-stats['observation.state.min']
            if np.any(span[parked] > 1e-6) or np.any(stats['observation.state.std'][parked] > 1e-6):
                raise ValueError('Corrected TCP checkpoint has nonstationary parked features')
            if np.any(self.mask & ~parked):
                raise ValueError('Unexpected constant active-arm feature')
            self.mask = parked
            self.reference[parked] = self.mean[parked]
        if not 0 < tolerance <= 1e-3:
            raise ValueError('Do not widen the audited constant-feature compatibility band')
        self.tolerance = tolerance

    def stabilize_training_state(self, state):
        out = np.asarray(state, dtype=np.float32).copy()
        _state_batch(out)
        deviation = np.abs(out[..., self.mask]-self.reference[self.mask])
        if np.any(deviation > self.tolerance):
            # Report which inputs failed, without relaxing the compatibility band
            # or assuming that a telemetry change proves physical movement.
            flat = out.reshape(-1, 32)
            delta = np.abs(flat-self.reference)
            failed = (delta > self.tolerance) & self.mask
            row_index, _ = np.argwhere(failed)[0]
            detail = ', '.join(
                f'{STATE_NAMES[i]}={float(flat[row_index, i]):.7g} '
                f'(training {float(self.reference[i]):.7g})'
                for i in np.flatnonzero(failed[row_index])[:8])
            raise ValueError(
                f'Parked arm/rail differs from training (band {self.tolerance:g}); '
                f'{detail}. Review telemetry with the operator before retrying')
        out[..., self.mask] = self.reference[self.mask]
        return out

    def to_training_state(self, current_state):
        """Convert observations only; corrected TCP checkpoints use identity."""
        if self.pose_convention == 'legacy_flange_intrinsic_xyz':
            return current_tcp_to_training_state(current_state)
        out, shape = _state_batch(current_state)
        return out.reshape(shape).astype(np.float32)

    def __call__(self, current_state):
        return self.stabilize_training_state(self.to_training_state(current_state))

    def to_current_state(self, recorded_state):
        """Recorded-input fixture conversion; not a motion target or command."""
        if self.pose_convention == 'legacy_flange_intrinsic_xyz':
            return training_to_current_tcp_state(recorded_state)
        out, shape = _state_batch(recorded_state)
        return out.reshape(shape).astype(np.float32)
