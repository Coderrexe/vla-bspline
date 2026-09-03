"""Read-only hidden-state telemetry for LIBERO / robosuite diagnostics.

Robosuite 1.4's compatibility ``MjSim.get_state()`` serializes only time,
``qpos``, and ``qvel``.  That is sufficient for LIBERO's published
``.pruned_init`` files, but it is not a complete replay state for modern
MuJoCo.  This helper deliberately uses MuJoCo's ``mjSTATE_INTEGRATION`` API so
warm-start acceleration, controls, applied forces, mocap state, history, and
plugin state are included as well.

The functions here do not reset or mutate an environment.  They are isolated
from the production evaluators so a diagnostic can be reviewed before it is
wired into a claim-bearing run.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np


PROTOCOL = "mujoco_integration_state_sha256_v1"


def _framed_sha256(*parts: tuple[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(PROTOCOL.encode("ascii") + b"\0")
    for tag, payload in parts:
        tag_bytes = tag.encode("ascii")
        digest.update(struct.pack("<Q", len(tag_bytes)))
        digest.update(tag_bytes)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _array_payload(value: Any) -> tuple[bytes, bytes]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object-dtype simulator state is not hashable")
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return header, array.tobytes(order="C")


def locate_robosuite_sim(environment: Any) -> tuple[Any, str]:
    """Find ``MjSim`` through Gym, LeRobot, and LIBERO wrapper layers.

    The LeRobot LIBERO wrapper uses the private ``_env`` attribute, while the
    LIBERO control wrapper uses ``env``.  Following only ``env`` / ``unwrapped``
    therefore misses the simulator in the deployed stack.
    """

    queue: list[tuple[Any, str]] = [(environment, "root")]
    seen: set[int] = set()
    while queue:
        candidate, path = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            simulator = getattr(candidate, "sim", None)
        except Exception:
            simulator = None
        if simulator is not None:
            model = getattr(simulator, "model", None)
            data = getattr(simulator, "data", None)
            if model is not None and data is not None:
                return simulator, f"{path}.sim"
        for attribute in ("_env", "env", "unwrapped"):
            try:
                child = getattr(candidate, attribute, None)
            except Exception:
                child = None
            if child is not None and child is not candidate:
                queue.append((child, f"{path}.{attribute}"))
    raise RuntimeError("could not locate robosuite MjSim through wrapper chain")


def integration_state_array(
    simulator: Any, *, mujoco_module: Any | None = None
) -> np.ndarray:
    """Copy the exact MuJoCo integration state without mutating the simulator."""

    if mujoco_module is None:
        import mujoco as mujoco_module  # type: ignore[no-redef]

    model_wrapper = simulator.model
    data_wrapper = simulator.data
    model = getattr(model_wrapper, "_model", model_wrapper)
    data = getattr(data_wrapper, "_data", data_wrapper)
    signature = int(mujoco_module.mjtState.mjSTATE_INTEGRATION)
    size = int(mujoco_module.mj_stateSize(model, signature))
    state = np.empty(size, dtype=np.float64)
    mujoco_module.mj_getState(model, data, state, signature)
    return state


def integration_state_sha256(
    simulator: Any, *, mujoco_module: Any | None = None
) -> str:
    state = integration_state_array(simulator, mujoco_module=mujoco_module)
    header, payload = _array_payload(state)
    return _framed_sha256(("integration_header", header), ("integration", payload))


def model_xml_sha256(simulator: Any) -> str:
    """Hash the compiled model XML, including mutable fixture body poses."""

    get_xml = getattr(simulator.model, "get_xml", None)
    if not callable(get_xml):
        raise RuntimeError("robosuite simulator model does not expose get_xml()")
    xml = get_xml()
    if not isinstance(xml, str):
        raise TypeError("simulator model XML must be text")
    return _framed_sha256(("compiled_model_xml", xml.encode("utf-8")))


def hidden_state_snapshot(
    environment: Any, *, mujoco_module: Any | None = None
) -> dict[str, Any]:
    """Return compact, read-only hashes suitable for per-step diagnostics."""

    simulator, path = locate_robosuite_sim(environment)
    state = integration_state_array(simulator, mujoco_module=mujoco_module)
    header, payload = _array_payload(state)
    return {
        "protocol": PROTOCOL,
        "simulator_path": path,
        "integration_state_size": int(state.size),
        "integration_state_sha256": _framed_sha256(
            ("integration_header", header), ("integration", payload)
        ),
        "compiled_model_xml_sha256": model_xml_sha256(simulator),
    }

