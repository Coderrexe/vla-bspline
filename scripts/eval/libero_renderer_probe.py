#!/usr/bin/env python3
"""Collect one backend replicate for the LIBERO renderer-comparability gate.

This is deliberately separate from the claim evaluator.  Run it in a fresh
process with ``MUJOCO_GL=egl`` or ``MUJOCO_GL=osmesa``.  It stores the exact
initial raw camera arrays and policy-processed image tensors in one compressed
NPZ sidecar, then executes the ordinary closed-loop policy from that same
observation.  A second script compares at least two replicates per backend.

The renderer is selected before importing this module; one process must never
attempt to compare two OpenGL context implementations in-process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# These must be selected before Torch initializes CUDA-backed libraries.  The
# launcher sets the same values; assignment here makes direct invocation safe.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import torch

from deterministic_eval import ACTION_TRACE_PROTOCOL, ExecutedActionTrace
from libero_hidden_state import (
    PROTOCOL as HIDDEN_STATE_PROTOCOL,
    integration_state_sha256,
    locate_robosuite_sim,
    model_xml_sha256,
)
from libero_duration_ablation_v2 import _structured_sha256
from libero_locked_eval_v2 import (
    _base_env,
    _determinism_metadata,
    _explicit_reset,
    _git_commit,
    _parse_task_ids,
    _scalar_bool,
    _seed_all,
    _sha256,
)


PROTOCOL = "libero_renderer_comparability_probe_v1"
ARRAY_PROTOCOL = "one_compressed_npz_exact_initial_camera_arrays_v1"

# Exact-task two-subgoal program.  Kept here, rather than imported from the
# claim evaluator, so a staged source snapshot contains every semantic input
# to the replay gate.
TWO_SUBGOAL_CLAUSES = {
    0: [
        "put the alphabet soup in the basket",
        "put the tomato sauce in the basket",
    ],
    4: [
        "put the white mug on the left plate",
        "put the yellow and white mug on the right plate",
    ],
}
EXPECTED_TASK_DESCRIPTIONS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    4: (
        "put the white mug on the left plate and put the yellow and white mug "
        "on the right plate"
    ),
}
FIXED_SWITCH_STEPS = {0: 139, 4: 107}


def _atomic_write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically publish one compressed array sidecar without clobbering."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication provides no-overwrite semantics on the shared
        # filesystem; the temporary inode is removed only after success.
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _flatten_arrays(value: Any, path: str = "") -> dict[str, np.ndarray]:
    """Return all array/tensor leaves under a nested mapping."""

    if isinstance(value, dict):
        flattened: dict[str, np.ndarray] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("observation mapping keys must be strings")
            child = f"{path}/{_json_pointer_token(key)}"
            flattened.update(_flatten_arrays(value[key], child))
        return flattened
    if torch.is_tensor(value):
        return {path or "/": value.detach().cpu().numpy().copy()}
    if isinstance(value, np.ndarray):
        return {path or "/": value.copy()}
    return {}


def _camera_arrays(value: Any, *, processed: bool) -> dict[str, np.ndarray]:
    """Select camera/image leaves and fail if an expected observation has none."""

    leaves = _flatten_arrays(value)
    selected: dict[str, np.ndarray] = {}
    for path, array in leaves.items():
        lower = path.lower()
        is_image_path = (
            "/pixels/" in lower
            or "image" in lower
            or "camera" in lower
        )
        if not is_image_path or array.ndim < 3:
            continue
        # The processed structure also contains masks in some policies.  They
        # are not rendered inputs and must not be mistaken for cameras.
        if processed and "mask" in lower:
            continue
        selected[path] = array
    if not selected:
        label = "processed" if processed else "raw"
        raise RuntimeError(f"no {label} camera arrays found in observation")
    return selected


def _array_key(task_id: int, state_id: int, kind: str, ordinal: int) -> str:
    return f"t{task_id:02d}_s{state_id:04d}_{kind}_{ordinal:02d}"


def _opengl_metadata() -> dict[str, str | None]:
    try:
        from OpenGL import GL

        def decode(name: int) -> str | None:
            value = GL.glGetString(name)
            return value.decode("utf-8", errors="replace") if value else None

        return {
            "vendor": decode(GL.GL_VENDOR),
            "renderer": decode(GL.GL_RENDERER),
            "version": decode(GL.GL_VERSION),
            "shading_language_version": decode(GL.GL_SHADING_LANGUAGE_VERSION),
        }
    except Exception as error:  # pragma: no cover - depends on cluster GL stack
        return {"query_error": f"{type(error).__name__}: {error}"}


def _episode_seed(seed_base: int, task_id: int, state_id: int) -> int:
    seed = seed_base + task_id * 1_000 + state_id
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError(f"derived environment seed is not uint32: {seed}")
    return seed


def _release_boundary(saw_closed: bool, previous_grip: float, grip: float) -> tuple[bool, bool]:
    """Return updated close history and whether an executed release occurred."""

    saw_closed = saw_closed or grip > 0
    return saw_closed, bool(saw_closed and previous_grip > 0 and grip <= 0)


def _clear_action_queue(policy: Any, action_key: str) -> None:
    """Drop commands generated under clause 1 before moving to clause 2."""

    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and action_key in queues:
        queues[action_key].clear()


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("mujoco", "robosuite", "libero", "lerobot"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def main() -> None:
    # LeRobot is imported only after the renderer and deterministic process
    # environment have been selected by the launcher.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task_ids", help="optional comma-separated task ids")
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states_per_task", type=int, default=10)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument(
        "--language_mode",
        choices=("compound", "fixed_clock", "event_clock", "duration_clock"),
        default="compound",
    )
    parser.add_argument("--replicate_index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arrays_out", required=True)
    args = parser.parse_args()
    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start nonnegative; states_per_task positive")
    if args.replicate_index < 0:
        raise ValueError("replicate_index must be nonnegative")
    backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    if backend not in {"egl", "osmesa"}:
        raise ValueError("MUJOCO_GL must be explicitly egl or osmesa")

    checkpoint = Path(args.ckpt).resolve()
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    output = Path(args.out).resolve()
    arrays_output = Path(args.arrays_out).resolve()
    for path in (config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or arrays_output.exists():
        raise FileExistsError("refusing to overwrite probe artifacts")

    _seed_all(args.seed_base)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    env_cfg = LiberoEnvCfg(
        task=args.suite,
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    suite_envs = make_env(env_cfg, n_envs=1)[args.suite]
    task_ids = _parse_task_ids(args.task_ids, list(suite_envs))
    if args.language_mode != "compound" and set(task_ids) != {0, 4}:
        raise ValueError(
            "two-subgoal language modes require exactly LIBERO suite tasks 0 and 4"
        )
    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    if args.language_mode == "duration_clock":
        if policy_cfg.type != "smolvla_spline" or not bool(
            getattr(policy.config, "predict_duration", False)
        ):
            raise ValueError(
                "duration_clock requires a C smolvla_spline checkpoint with predict_duration=true"
            )
        horizon_max = int(getattr(policy.config, "horizon_max", 0))
        min_seg = int(getattr(policy.config, "min_seg", 0))
        if not (0 < min_seg <= horizon_max):
            raise ValueError("duration_clock requires a valid [min_seg, horizon_max]")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    arrays: dict[str, np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    gl_metadata: dict[str, str | None] | None = None
    env_source: Path | None = None
    started = time.time()
    try:
        for task_id in task_ids:
            env = suite_envs[task_id]
            if env_source is None:
                env_source = Path(inspect.getfile(type(_base_env(env)))).resolve()
            max_steps = int(env.call("_max_episode_steps")[0])
            task_description = str(next(iter(env.call("task_description"))))
            expected_description = EXPECTED_TASK_DESCRIPTIONS.get(task_id)
            if expected_description is not None and task_description != expected_description:
                raise RuntimeError(
                    f"suite task {task_id} description mismatch: got {task_description!r}, "
                    f"expected {expected_description!r}"
                )
            clauses = TWO_SUBGOAL_CLAUSES.get(task_id, [])
            for state_id in state_ids:
                seed = _episode_seed(args.seed_base, task_id, state_id)
                observation, _, realized = _explicit_reset(env, state_id, seed, seed)
                simulator, simulator_path = locate_robosuite_sim(_base_env(env))
                if gl_metadata is None:
                    gl_metadata = _opengl_metadata()
                initial_simulator_hash = integration_state_sha256(simulator)
                initial_model_hash = model_xml_sha256(simulator)
                initial_raw_hash = _structured_sha256(observation)

                clause_index = 0
                saw_closed = False
                previous_grip = -1.0
                switch_steps: list[int] = []
                prompts_used: list[str] = []
                duration_chunks: list[dict[str, int]] = []
                duration_switch_trigger: dict[str, int] | None = None
                previous_duration_batch: Any | None = None
                prompt = task_description if args.language_mode == "compound" else clauses[0]
                processed = preprocess_observation(observation)
                processed["task"] = [prompt]
                processed = env_preprocessor(processed)
                processed = preprocessor(processed)
                initial_processed_hash = _structured_sha256(processed)
                raw_images = _camera_arrays(observation, processed=False)
                processed_images = _camera_arrays(processed, processed=True)
                raw_manifest: dict[str, Any] = {}
                processed_manifest: dict[str, Any] = {}
                for ordinal, (path, array) in enumerate(sorted(raw_images.items())):
                    key = _array_key(task_id, state_id, "raw", ordinal)
                    arrays[key] = array
                    raw_manifest[path] = {
                        "array_key": key,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    }
                for ordinal, (path, array) in enumerate(
                    sorted(processed_images.items())
                ):
                    key = _array_key(task_id, state_id, "processed", ordinal)
                    arrays[key] = array
                    processed_manifest[path] = {
                        "array_key": key,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    }

                policy.reset()
                trace = ExecutedActionTrace()
                success = False
                step = 0
                first_action_sha256 = None
                current_processed = processed
                while step < max_steps:
                    if (
                        args.language_mode == "fixed_clock"
                        and clause_index == 0
                        and step >= FIXED_SWITCH_STEPS[task_id]
                    ):
                        clause_index = 1
                        switch_steps.append(step)
                        _clear_action_queue(policy, ACTION)
                    prompt = task_description if args.language_mode == "compound" else clauses[clause_index]
                    if not prompts_used or prompts_used[-1] != prompt:
                        prompts_used.append(prompt)
                    if step:
                        current_processed = preprocess_observation(observation)
                        current_processed["task"] = [prompt]
                        current_processed = env_preprocessor(current_processed)
                        current_processed = preprocessor(current_processed)
                    chunks_before = int(getattr(policy, "n_chunks_generated", -1))
                    if chunks_before < 0:
                        raise RuntimeError("policy does not expose n_chunks_generated")
                    with torch.inference_mode():
                        action = policy.select_action(current_processed)
                    chunks_after = int(getattr(policy, "n_chunks_generated", -1))
                    if chunks_after - chunks_before not in (0, 1):
                        raise RuntimeError(
                            "unexpected chunk-generation delta from production select_action: "
                            f"{chunks_before}->{chunks_after}"
                        )
                    if args.language_mode == "duration_clock" and chunks_after == chunks_before + 1:
                        batch = getattr(policy, "last_predicted_T_batch", None)
                        if batch is None:
                            raise RuntimeError(
                                "duration_clock unavailable: select_action generated a chunk without "
                                "last_predicted_T_batch"
                            )
                        if batch is previous_duration_batch:
                            raise RuntimeError(
                                "duration_clock received a stale last_predicted_T_batch"
                            )
                        previous_duration_batch = batch
                        numel = getattr(batch, "numel", None)
                        median = getattr(batch, "median", None)
                        if not callable(numel) or int(numel()) != 1 or not callable(median):
                            raise RuntimeError(
                                "duration_clock requires a batch-1 tensor with median()"
                            )
                        value = median().item()
                        predicted_duration = int(value)
                        if predicted_duration != value or not min_seg <= predicted_duration <= horizon_max:
                            raise RuntimeError(
                                f"duration_clock received invalid predicted duration {value!r}"
                            )
                        duration_chunk = {
                            "chunk_index": len(duration_chunks),
                            "environment_step": step,
                            "predicted_duration": predicted_duration,
                        }
                        duration_chunks.append(duration_chunk)
                        if clause_index == 0 and predicted_duration < horizon_max:
                            clause_index = 1
                            # The triggering action remains under clause 1;
                            # discard only queued residual actions before the
                            # next control decision uses clause 2.
                            switch_steps.append(step + 1)
                            duration_switch_trigger = dict(duration_chunk)
                            _clear_action_queue(policy, ACTION)
                    action = postprocessor(action)
                    transition = env_postprocessor({ACTION: action})
                    action_numpy = transition[ACTION].detach().cpu().numpy()
                    if first_action_sha256 is None:
                        first_action_sha256 = hashlib.sha256(
                            action_numpy.tobytes(order="C")
                        ).hexdigest()
                    trace.update(action_numpy)
                    observation, reward, terminated, truncated, info = env.step(
                        action_numpy
                    )
                    step += 1
                    if args.language_mode == "event_clock" and clause_index == 0:
                        grip = float(np.asarray(action_numpy).reshape(-1)[6])
                        saw_closed, released = _release_boundary(
                            saw_closed, previous_grip, grip
                        )
                        if released:
                            clause_index = 1
                            switch_steps.append(step)
                            _clear_action_queue(policy, ACTION)
                        previous_grip = grip
                    success = success or bool(np.asarray(reward).max() >= 1.0)
                    if "is_success" in info:
                        success = success or _scalar_bool(info["is_success"])
                    final_info = info.get("final_info")
                    if isinstance(final_info, dict) and "is_success" in final_info:
                        success = success or _scalar_bool(final_info["is_success"])
                    if _scalar_bool(terminated) or _scalar_bool(truncated):
                        break
                if args.language_mode == "duration_clock" and not duration_chunks:
                    raise RuntimeError(
                        "duration_clock unavailable: episode ended without a generated duration chunk"
                    )
                episodes.append(
                    {
                        "task_id": task_id,
                        "task_description": task_description,
                        "language_mode": args.language_mode,
                        "reviewed_clauses": clauses,
                        "prompts_used": prompts_used,
                        "switch_steps": switch_steps,
                        "final_clause_index": clause_index,
                        "fixed_switch_step": FIXED_SWITCH_STEPS.get(task_id),
                        "duration_clock_horizon_max": (
                            horizon_max if args.language_mode == "duration_clock" else None
                        ),
                        "duration_clock_chunks": duration_chunks,
                        "duration_clock_switch_trigger": duration_switch_trigger,
                        "state_id": realized,
                        "env_seed_u32": seed,
                        "policy_seed_u64": seed,
                        "initial_raw_observation_sha256": initial_raw_hash,
                        "initial_processed_observation_sha256": initial_processed_hash,
                        "hidden_state_protocol": HIDDEN_STATE_PROTOCOL,
                        "simulator_path": simulator_path,
                        "initial_mujoco_integration_state_sha256": initial_simulator_hash,
                        "compiled_model_xml_sha256": initial_model_hash,
                        "raw_cameras": raw_manifest,
                        "processed_cameras": processed_manifest,
                        "success": success,
                        "steps": step,
                        "first_action_sha256": first_action_sha256,
                        "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
                        "executed_action_trace_sha256": trace.hexdigest(),
                    }
                )
                print(
                    f"renderer={backend} replicate={args.replicate_index} "
                    f"task={task_id} state={state_id} success={int(success)} "
                    f"steps={step} trace={trace.hexdigest()[:12]}",
                    flush=True,
                )
            env.close()
    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    if gl_metadata is None or env_source is None:
        raise RuntimeError("probe executed no episodes")
    _atomic_write_npz_new(arrays_output, arrays)
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    checkpoint_config = json.loads(config_path.read_text())
    stats_hash = None
    stats_name = checkpoint_config.get("spline_stats_file_v2")
    if stats_name:
        stats_path = policy_source.parent / stats_name
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"checkpoint references missing spline stats: {stats_path}"
            )
        stats_hash = _sha256(stats_path)
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "array_protocol": ARRAY_PROTOCOL,
        "renderer_backend": backend,
        "replicate_index": args.replicate_index,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "policy_type": checkpoint_config["type"],
        "n_action_steps": checkpoint_config.get("n_action_steps"),
        "suite": args.suite,
        "task_ids": task_ids,
        "state_ids": state_ids,
        "seed_base": args.seed_base,
        "control_frequency_hz": args.control_freq,
        "language_mode": args.language_mode,
        "n_episodes": len(episodes),
        "n_successes": sum(int(episode["success"]) for episode in episodes),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "episodes": episodes,
        "arrays_file": str(arrays_output),
        "arrays_sha256": _sha256(arrays_output),
        "opengl": gl_metadata,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "software_versions": _software_versions(),
            "determinism": _determinism_metadata(),
        },
        "source": {
            "collector_sha256": _sha256(Path(__file__).resolve()),
            "locked_reset_helper_sha256": _sha256(
                Path(inspect.getfile(_explicit_reset)).resolve()
            ),
            "structured_hash_helper_sha256": _sha256(
                Path(inspect.getfile(_structured_sha256)).resolve()
            ),
            "hidden_state_helper_sha256": _sha256(
                Path(inspect.getfile(integration_state_sha256)).resolve()
            ),
            "policy_source_sha256": _sha256(policy_source),
            "env_source_sha256": _sha256(env_source),
            "stats_sha256": stats_hash,
            "lerobot_commit": _git_commit(policy_source.parents[4]),
        },
        "wall_time_seconds": time.time() - started,
    }
    _atomic_write_json_new(output, payload)
    print(f"wrote {output} and {arrays_output}", flush=True)


if __name__ == "__main__":
    main()
