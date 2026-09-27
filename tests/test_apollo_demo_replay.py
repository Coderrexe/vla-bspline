"""Pure offline label audit tests; no network, runtime, robot, or model imports."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from prepare_apollo_demo_replay import compare_commands, rotations_from_columns


def recording():
    rng = np.random.default_rng(19)
    n = 12
    state = np.zeros((n, 32))
    state[:, [12, 28]] = 1
    delta = rng.normal(0, .001, (n, 16))
    delta[:, [6, 14]] = .8
    absolute = np.zeros((n, 22))
    for s, d, a in [(0, 0, 0), (16, 8, 11)]:
        rotation = Rotation.from_euler('xyz', [.3, -.4, .8])
        state[0, s+12:s+16] = rotation.as_quat()[[3, 0, 1, 2]]
        for k in range(n):
            rotation = Rotation.from_rotvec(delta[k, d+3:d+6])*rotation
            matrix = rotation.as_matrix()
            absolute[k, a+3:a+9] = np.r_[matrix[:, 0], matrix[:, 1]]
        absolute[:, a:a+3] = np.cumsum(delta[:, d:d+3], axis=0)
        absolute[:, a+9] = delta[:, d+6]
        absolute[:, a+10] = np.cumsum(delta[:, d+7])
    return state, delta, absolute


def test_spatial_composition_and_next_frame_index_agree():
    result = compare_commands(*recording())
    for arm in ('grip', 'view'):
        assert result[arm]['command_translation_residual_mm']['max'] < 1e-10
        assert result[arm]['command_rotation_residual_rad']['max'] < 1e-10
        assert result[arm]['initial_measured_vs_command_anchor_rad'] < 1e-10


@pytest.mark.parametrize('column', [0, 4, 9, 10, 11, 20])
def test_changed_absolute_label_refused(column):
    state, delta, absolute = recording()
    absolute[5, column] += .1
    with pytest.raises(ValueError, match='do not match'):
        compare_commands(state, delta, absolute)


def test_off_by_one_delta_refused():
    state, delta, absolute = recording()
    with pytest.raises(ValueError, match='do not match'):
        compare_commands(state, np.roll(delta, 1, axis=0), absolute)


def test_zero_and_parallel_rotation_columns_refused():
    for values in (np.zeros((1, 6)), np.array([[1, 0, 0, 2, 0, 0]])):
        with pytest.raises(ValueError, match='Degenerate'):
            rotations_from_columns(values)
