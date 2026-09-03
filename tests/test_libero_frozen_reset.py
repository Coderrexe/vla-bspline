from __future__ import annotations

import hashlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from libero_frozen_reset import (  # noqa: E402
    FrozenLiberoState,
    step_without_terminal_autoreset,
)


class _RawModel:
    pass


class _RawData:
    def __init__(self) -> None:
        self.values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)


class _Model:
    def __init__(self) -> None:
        self._model = _RawModel()
        self.xml_epoch = 0

    def get_xml(self) -> str:
        return f'<mujoco model="one-live-model-{self.xml_epoch}"/>'


class _Data:
    def __init__(self) -> None:
        self._data = _RawData()


class _Sim:
    def __init__(self) -> None:
        self.model = _Model()
        self.data = _Data()


class _Env:
    def __init__(self) -> None:
        self.sim = _Sim()
        self._elapsed_steps = 0
        self.np_random = np.random.default_rng(19)
        self.renderer_handle = object()  # must not be copied/replaced


class _TerminalLiberoEnv(_Env):
    """Models LeRobot LiberoEnv.step's internal reset-on-termination bug."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        self.sim.model.xml_epoch += 1
        return {"image": np.asarray([[0]], dtype=np.uint8)}, {}

    def step(self, action):
        self.reset()  # This is precisely the installed LiberoEnv behavior.
        return {"image": np.asarray([[4]], dtype=np.uint8)}, 1.0, True, False, {}


class _Vector:
    def __init__(self, env) -> None:
        self.envs = [env]
        self._autoreset_envs = np.asarray([False], dtype=np.bool_)
        self.autoreset_mode = types.SimpleNamespace(name="NEXT_STEP")

    def step(self, action):
        transition = self.envs[0].step(action)
        self._autoreset_envs[:] = transition[2] or transition[3]
        observation, reward, terminated, truncated, info = transition
        return (
            {key: np.expand_dims(value, 0) for key, value in observation.items()},
            np.asarray([reward]), np.asarray([terminated]), np.asarray([truncated]), info,
        )


class _Mujoco:
    class mjtState:
        mjSTATE_INTEGRATION = 16383

    forwards = 0

    @staticmethod
    def mj_stateSize(model, signature):
        assert signature == 16383
        return 3

    @staticmethod
    def mj_getState(model, data, destination, signature):
        destination[:] = data.values

    @staticmethod
    def mj_setState(model, data, source, signature):
        data.values[:] = source

    @classmethod
    def mj_forward(cls, model, data):
        cls.forwards += 1


def _hash(value) -> str:
    image = value["image"]
    return hashlib.sha256(image.dtype.str.encode() + image.tobytes()).hexdigest()


class FrozenLiberoResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_mujoco = sys.modules.get("mujoco")
        sys.modules["mujoco"] = types.SimpleNamespace(
            mjtState=_Mujoco.mjtState,
            mj_stateSize=_Mujoco.mj_stateSize,
            mj_getState=_Mujoco.mj_getState,
            mj_setState=_Mujoco.mj_setState,
            mj_forward=_Mujoco.mj_forward,
        )
        _Mujoco.forwards = 0

    def tearDown(self) -> None:
        if self.previous_mujoco is None:
            del sys.modules["mujoco"]
        else:
            sys.modules["mujoco"] = self.previous_mujoco

    def test_restore_reuses_live_model_and_restores_state_observation_and_wrapper(self) -> None:
        env = _Env()
        original_renderer = env.renderer_handle
        observation = {"image": np.asarray([[3, 4]], dtype=np.uint8)}
        frozen = FrozenLiberoState.capture(env, observation, _hash)

        env.sim.data._data.values[:] = [9, 9, 9]
        env._elapsed_steps = 77
        observation["image"][:] = 0
        restored = frozen.restore(env, _hash)

        np.testing.assert_array_equal(env.sim.data._data.values, [1, 2, 3])
        np.testing.assert_array_equal(restored["image"], [[3, 4]])
        self.assertEqual(env._elapsed_steps, 0)
        self.assertIs(env.renderer_handle, original_renderer)
        self.assertEqual(_Mujoco.forwards, 1)

    def test_restore_returns_a_fresh_observation_copy(self) -> None:
        env = _Env()
        frozen = FrozenLiberoState.capture(
            env, {"image": np.asarray([[8]], dtype=np.uint8)}, _hash
        )
        first = frozen.restore(env, _hash)
        first["image"][:] = 1
        second = frozen.restore(env, _hash)
        np.testing.assert_array_equal(second["image"], [[8]])

    def test_terminal_autoreset_is_suppressed_and_vector_flag_is_restored(self) -> None:
        base = _TerminalLiberoEnv()
        vector = _Vector(base)
        frozen = FrozenLiberoState.capture(
            base,
            {"image": np.asarray([[8]], dtype=np.uint8)},
            _hash,
            wrapper_root=vector,
        )
        transition, suppressed = step_without_terminal_autoreset(
            vector, base, np.asarray([0.0])
        )
        self.assertTrue(suppressed)
        self.assertTrue(bool(transition[2][0]))
        self.assertEqual(base.reset_calls, 0)
        self.assertEqual(base.sim.model.xml_epoch, 0)
        self.assertTrue(bool(vector._autoreset_envs[0]))

        frozen.restore(base, _hash, wrapper_root=vector)
        self.assertFalse(bool(vector._autoreset_envs[0]))

    def test_same_step_vector_autoreset_is_rejected(self) -> None:
        base = _TerminalLiberoEnv()
        vector = _Vector(base)
        vector.autoreset_mode = types.SimpleNamespace(name="SAME_STEP")
        with self.assertRaisesRegex(RuntimeError, "SAME_STEP"):
            step_without_terminal_autoreset(vector, base, np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
