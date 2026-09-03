from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from libero_hidden_state import (  # noqa: E402
    hidden_state_snapshot,
    integration_state_array,
    locate_robosuite_sim,
)


class _RawModel:
    pass


class _RawData:
    def __init__(self, values: list[float]):
        self.values = np.asarray(values, dtype=np.float64)


class _ModelWrapper:
    def __init__(self):
        self._model = _RawModel()

    def get_xml(self) -> str:
        return '<mujoco model="diagnostic"/>'


class _DataWrapper:
    def __init__(self, values: list[float]):
        self._data = _RawData(values)


class _Sim:
    def __init__(self, values: list[float]):
        self.model = _ModelWrapper()
        self.data = _DataWrapper(values)

    def get_state(self):
        raise AssertionError("incomplete robosuite get_state() must not be used")


class _LiberoControlWrapper:
    def __init__(self, sim: _Sim):
        self.sim = sim


class _LeRobotWrapper:
    def __init__(self, sim: _Sim):
        self._env = _LiberoControlWrapper(sim)

    @property
    def unwrapped(self):
        return self


class _Mujoco:
    class mjtState:
        mjSTATE_INTEGRATION = 16383

    calls: list[tuple[str, int]] = []

    @classmethod
    def mj_stateSize(cls, model, signature):
        cls.calls.append(("size", signature))
        return 4

    @classmethod
    def mj_getState(cls, model, data, destination, signature):
        cls.calls.append(("get", signature))
        destination[:] = data.values


class LiberoHiddenStateTest(unittest.TestCase):
    def setUp(self) -> None:
        _Mujoco.calls.clear()

    def test_private_lerobot_env_layer_is_traversed(self) -> None:
        sim = _Sim([1, 2, 3, 4])
        found, path = locate_robosuite_sim(_LeRobotWrapper(sim))
        self.assertIs(found, sim)
        self.assertEqual(path, "root._env.sim")

    def test_full_integration_api_bypasses_incomplete_get_state(self) -> None:
        sim = _Sim([1, 2, 3, 4])
        state = integration_state_array(sim, mujoco_module=_Mujoco)
        np.testing.assert_array_equal(state, [1, 2, 3, 4])
        self.assertEqual(
            _Mujoco.calls,
            [("size", 16383), ("get", 16383)],
        )

    def test_snapshot_changes_when_hidden_integration_state_changes(self) -> None:
        first = hidden_state_snapshot(
            _LeRobotWrapper(_Sim([1, 2, 3, 4])), mujoco_module=_Mujoco
        )
        second = hidden_state_snapshot(
            _LeRobotWrapper(_Sim([1, 2, 3, 5])), mujoco_module=_Mujoco
        )
        self.assertNotEqual(
            first["integration_state_sha256"],
            second["integration_state_sha256"],
        )
        self.assertEqual(
            first["compiled_model_xml_sha256"],
            second["compiled_model_xml_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
