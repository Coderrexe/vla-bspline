"""Frozen compiled-model reset support for paired LIBERO rollouts.

The normal LIBERO reset path can reload task XML.  That is correct for an
ordinary episode, but it defeats a within-state paired comparison when the
loader makes equivalent fixture placements through different compiled models.
This module deliberately captures one *already reset* environment and restores
only that object's MuJoCo integration state for subsequent arms.  It never
calls ``reset`` or reloads XML after capture.

Only the small, explicitly mutable wrapper fields are copied.  Simulator/model
objects, task definitions, renderer handles, and wrapper links are never
copied or replaced.  A restore is fail-closed unless the integration-state,
compiled-model XML, and supplied raw observation all match their capture-time
digests.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from libero_hidden_state import (
    integration_state_array,
    integration_state_sha256,
    locate_robosuite_sim,
    model_xml_sha256,
)


PROTOCOL = "libero_frozen_compiled_model_reset_v4"
TERMINAL_AUTORESET_PROTOCOL = "suppress_libero_terminal_reset_and_restore_vector_next_step_flag_v1"
_UNSUPPORTED = object()


def clone_observation(value: Any) -> Any:
    """Copy an observation without sharing mutable image/state buffers."""

    if isinstance(value, np.ndarray):
        return value.copy()
    # Torch is intentionally duck-typed to keep CPU unit tests dependency-free.
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_observation(item) for item in value)
    return copy.deepcopy(value)


def _wrapper_chain(root: Any) -> list[Any]:
    """Return reachable wrappers once, while preserving their live objects."""

    queue = [root]
    seen: set[int] = set()
    result: list[Any] = []
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        for name in ("_env", "env", "unwrapped"):
            try:
                child = getattr(current, name, None)
            except Exception:
                child = None
            if child is not None and child is not current:
                queue.append(child)
        # Gymnasium SyncVectorEnv owns its member environments through a list,
        # rather than an ``env`` attribute. Capturing it is necessary because a
        # terminal rollout marks ``_autoreset_envs`` for its *next* step.
        try:
            children = getattr(current, "envs", None)
        except Exception:
            children = None
        if isinstance(children, (list, tuple)):
            queue.extend(child for child in children if child is not current)
    return result


def _safe_wrapper_value(value: Any) -> Any:
    """Copy scalar/counter/RNG wrapper state, never object graph references."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return copy.deepcopy(value)
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        copied = [_safe_wrapper_value(item) for item in value]
        return tuple(copied) if all(item is not _UNSUPPORTED for item in copied) else _UNSUPPORTED
    if isinstance(value, list):
        copied = [_safe_wrapper_value(item) for item in value]
        return copied if all(item is not _UNSUPPORTED for item in copied) else _UNSUPPORTED
    if isinstance(value, dict) and all(isinstance(key, (str, int, float)) for key in value):
        copied = {key: _safe_wrapper_value(item) for key, item in value.items()}
        return copied if all(item is not _UNSUPPORTED for item in copied.values()) else _UNSUPPORTED
    return _UNSUPPORTED


def _capture_wrapper_fields(root: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, wrapper in enumerate(_wrapper_chain(root)):
        fields: dict[str, Any] = {}
        for name, value in vars(wrapper).items():
            # Links and simulator-backed objects must remain exactly the live
            # instances constructed at block capture.
            if name in {"env", "_env", "sim", "model", "data", "unwrapped"}:
                continue
            copied = _safe_wrapper_value(value)
            if copied is not _UNSUPPORTED:
                fields[name] = copied
        generator_state = None
        rng = getattr(wrapper, "np_random", None)
        if isinstance(rng, np.random.Generator):
            generator_state = copy.deepcopy(rng.bit_generator.state)
        records.append(
            {"chain_index": index, "fields": fields, "np_random_state": generator_state}
        )
    return records


def _restore_wrapper_fields(root: Any, records: list[dict[str, Any]]) -> None:
    chain = _wrapper_chain(root)
    if len(chain) != len(records):
        raise RuntimeError("LIBERO wrapper chain changed after frozen-state capture")
    for wrapper, record in zip(chain, records, strict=True):
        for name, value in record["fields"].items():
            if not hasattr(wrapper, name):
                raise RuntimeError(f"wrapper field disappeared after capture: {name}")
            setattr(wrapper, name, clone_observation(value))
        rng_state = record["np_random_state"]
        if rng_state is not None:
            rng = getattr(wrapper, "np_random", None)
            if not isinstance(rng, np.random.Generator):
                raise RuntimeError("wrapper np_random changed type after frozen-state capture")
            rng.bit_generator.state = copy.deepcopy(rng_state)


@dataclass(frozen=True)
class FrozenLiberoState:
    """A single-reset snapshot tied to one live compiled MuJoCo model."""

    integration_state: np.ndarray
    integration_state_sha256: str
    compiled_model_xml_sha256: str
    raw_observation: Any
    raw_observation_sha256: str
    simulator_path: str
    wrapper_fields: list[dict[str, Any]]

    @classmethod
    def capture(
        cls,
        env: Any,
        observation: Any,
        structured_sha256: Any,
        *,
        wrapper_root: Any | None = None,
    ) -> "FrozenLiberoState":
        """Capture simulator state plus wrapper state rooted at ``wrapper_root``.

        The simulator itself normally sits below a vector wrapper.  Keeping the
        two roots separate avoids changing how the robosuite simulator is found
        while also restoring Gymnasium's terminal-autoreset bookkeeping.
        """

        simulator, path = locate_robosuite_sim(env)
        return cls(
            integration_state=integration_state_array(simulator).copy(),
            integration_state_sha256=integration_state_sha256(simulator),
            compiled_model_xml_sha256=model_xml_sha256(simulator),
            raw_observation=clone_observation(observation),
            raw_observation_sha256=structured_sha256(observation),
            simulator_path=path,
            wrapper_fields=_capture_wrapper_fields(
                env if wrapper_root is None else wrapper_root
            ),
        )

    def restore(
        self,
        env: Any,
        structured_sha256: Any,
        *,
        wrapper_root: Any | None = None,
    ) -> Any:
        """Restore this exact model/state and return a fresh initial observation.

        ``mj_forward`` is required by MuJoCo after ``mj_setState``.  The strict
        hash checks immediately afterwards detect a binding whose forward pass
        changes integration fields, rather than silently weakening pairing.
        """

        import mujoco

        simulator, path = locate_robosuite_sim(env)
        if path != self.simulator_path:
            raise RuntimeError(f"simulator path changed: {path} != {self.simulator_path}")
        if model_xml_sha256(simulator) != self.compiled_model_xml_sha256:
            raise RuntimeError("compiled model XML changed; refusing frozen restore")
        model_wrapper, data_wrapper = simulator.model, simulator.data
        model = getattr(model_wrapper, "_model", model_wrapper)
        data = getattr(data_wrapper, "_data", data_wrapper)
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        expected_size = int(mujoco.mj_stateSize(model, signature))
        if expected_size != int(self.integration_state.size):
            raise RuntimeError("MuJoCo integration-state size changed after capture")
        _restore_wrapper_fields(
            env if wrapper_root is None else wrapper_root, self.wrapper_fields
        )
        mujoco.mj_setState(model, data, self.integration_state, signature)
        mujoco.mj_forward(model, data)
        if integration_state_sha256(simulator) != self.integration_state_sha256:
            raise RuntimeError("MuJoCo integration state did not round-trip exactly")
        if model_xml_sha256(simulator) != self.compiled_model_xml_sha256:
            raise RuntimeError("compiled model XML changed during frozen restore")
        observation = clone_observation(self.raw_observation)
        if structured_sha256(observation) != self.raw_observation_sha256:
            raise RuntimeError("captured initial raw observation was mutated")
        return observation


def step_without_terminal_autoreset(
    vector_env: Any, base_env: Any, action: Any
) -> tuple[tuple[Any, Any, Any, Any, Any], bool]:
    """Step once while blocking LIBERO's outcome-dependent terminal reset.

    LeRobot's ``LiberoEnv.step`` calls ``self.reset()`` immediately after a
    successful/terminal transition. That reset changes placement-bearing model
    fields before a frozen paired arm can restore its captured integration
    state. The reset return value is discarded by LiberoEnv, so a scoped no-op
    preserves the terminal transition (observation, reward, done, info) while
    preserving the live compiled model. Gymnasium's default vector mode is
    NEXT_STEP; its flag is restored from the frozen wrapper snapshot before the
    following arm.
    """

    mode = getattr(vector_env, "autoreset_mode", None)
    mode_name = getattr(mode, "name", str(mode)).upper()
    if "SAME_STEP" in mode_name:
        raise RuntimeError(
            "frozen paired evaluation requires non-SAME_STEP vector autoreset"
        )
    original_reset = getattr(base_env, "reset", None)
    if not callable(original_reset):
        raise RuntimeError("base LIBERO environment does not expose reset")
    suppressed = False

    def no_terminal_reset(*_args: Any, **_kwargs: Any) -> None:
        nonlocal suppressed
        suppressed = True
        # LiberoEnv.step intentionally ignores this return value.
        return None

    setattr(base_env, "reset", no_terminal_reset)
    try:
        transition = vector_env.step(action)
    finally:
        setattr(base_env, "reset", original_reset)
    if not isinstance(transition, tuple) or len(transition) != 5:
        raise RuntimeError("vector environment step did not return Gymnasium five-tuple")
    return transition, suppressed
