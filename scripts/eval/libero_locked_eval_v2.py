"""Deterministic, explicit-state LIBERO evaluation with action-level traces.

This is an isolated successor to ``libero_locked_eval.py``.  It strengthens the
existing outcome-independent initial-state protocol by configuring deterministic
Torch/CUDA execution, reseeding the policy RNG after every environment reset,
and hashing every action array actually passed to ``env.step``.

Only compact scalar records and SHA-256 traces are saved; no images or videos
are emitted.  Output publication is atomic and refuses to overwrite any prior
artifact.
"""

from __future__ import annotations

# These variables must be fixed before Torch (or anything importing Torch) is
# imported in this process.  Assignment, rather than setdefault, prevents an
# inherited incompatible cuBLAS workspace setting from weakening the protocol.
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import argparse
import hashlib
import inspect
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deterministic_eval import (
    ACTION_TRACE_PROTOCOL,
    ExecutedActionTrace,
    atomic_write_json_new,
)


def _configure_determinism() -> None:
    """Make nondeterministic Torch kernels fail instead of silently drifting."""

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


# Configure Torch before importing LeRobot, whose imports may transitively load
# CUDA-backed libraries. The CUDA environment variables above were set even
# earlier, before Torch itself was imported.
_configure_determinism()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _scalar_bool(value: Any) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _base_env(sync_vec_env: Any) -> Any:
    """Return the one underlying LiberoEnv and fail on another topology."""

    envs = getattr(sync_vec_env, "envs", None)
    if envs is None or len(envs) != 1:
        raise RuntimeError(
            "Locked evaluation requires a one-environment SyncVectorEnv; "
            f"got {type(sync_vec_env)!r} with envs={envs!r}"
        )
    return getattr(envs[0], "unwrapped", envs[0])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_policy_after_reset(seed: int) -> None:
    """Seed only Torch/CUDA for the stochastic policy, after simulator reset."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _explicit_reset(
    env: Any, state_id: int, env_seed: int, policy_seed: int
) -> tuple[Any, Any, int]:
    """Reset an explicit LIBERO state, then immediately seed the policy RNG."""

    base = _base_env(env)
    _seed_all(env_seed)
    base.init_state_id = state_id
    observation, info = env.reset(seed=env_seed)
    # Keep this immediately after env.reset: reset may consume Torch/CUDA RNG,
    # while policy flow sampling must start from a state independent of it.
    _seed_policy_after_reset(policy_seed)
    realized = int(base.init_state_id) - int(base._reset_stride)
    if realized != state_id:
        raise RuntimeError(f"requested init state {state_id}, realized {realized}")
    return observation, info, realized


def _parse_task_ids(value: str | None, available: list[int]) -> list[int]:
    if value is None:
        return sorted(available)
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("--task_ids must be a nonempty, duplicate-free CSV")
    missing = sorted(set(parsed) - set(available))
    if missing:
        raise ValueError(f"task ids unavailable in suite: {missing}")
    return sorted(parsed)


def _determinism_metadata() -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    try:
        nvidia_smi_devices = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        nvidia_smi_devices = []
    cuda_backend = torch.backends.cuda
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_smi_name_uuid_pci_driver": nvidia_smi_devices,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "cuda_devices": devices,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "flash_sdp_enabled": (
            cuda_backend.flash_sdp_enabled()
            if hasattr(cuda_backend, "flash_sdp_enabled")
            else None
        ),
        "memory_efficient_sdp_enabled": (
            cuda_backend.mem_efficient_sdp_enabled()
            if hasattr(cuda_backend, "mem_efficient_sdp_enabled")
            else None
        ),
        "math_sdp_enabled": (
            cuda_backend.math_sdp_enabled()
            if hasattr(cuda_backend, "math_sdp_enabled")
            else None
        ),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    # Import LeRobot only after the module-level determinism configuration has
    # run, guarding against import-time CUDA initialization in dependencies.
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
    parser.add_argument("--states_per_task", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start must be nonnegative and states_per_task positive")

    checkpoint = Path(args.ckpt).resolve()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_seed = args.seed_base
    _seed_all(bootstrap_seed)

    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    env_cfg = LiberoEnvCfg(
        task=args.suite,
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    envs_by_suite = make_env(env_cfg, n_envs=1)
    suite_envs = envs_by_suite[args.suite]
    task_ids = _parse_task_ids(args.task_ids, list(suite_envs))
    env_source = Path(
        inspect.getfile(type(_base_env(suite_envs[task_ids[0]])))
    ).resolve()

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    task_results: list[dict[str, Any]] = []
    started = time.time()

    try:
        for task_id in task_ids:
            env = suite_envs[task_id]
            max_steps = int(env.call("_max_episode_steps")[0])
            task_description = str(next(iter(env.call("task_description"))))
            episodes: list[dict[str, Any]] = []

            for state_id in state_ids:
                env_seed = args.seed_base + task_id * 1_000 + state_id
                policy_seed = env_seed
                observation, _, realized = _explicit_reset(
                    env, state_id, env_seed, policy_seed
                )
                policy.reset()
                success = False
                step = 0
                policy_calls = 0
                trace = ExecutedActionTrace()
                chunks_start = getattr(policy, "n_chunks_generated", None)

                while step < max_steps:
                    processed = preprocess_observation(observation)
                    processed["task"] = [task_description]
                    processed = env_preprocessor(processed)
                    processed = preprocessor(processed)
                    with torch.inference_mode():
                        action = policy.select_action(processed)
                    policy_calls += 1
                    action = postprocessor(action)
                    transition = env_postprocessor({ACTION: action})
                    action_numpy = transition[ACTION].detach().cpu().numpy()
                    trace.update(action_numpy)
                    observation, reward, terminated, truncated, info = env.step(
                        action_numpy
                    )
                    step += 1

                    success = success or bool(np.asarray(reward).max() >= 1.0)
                    if "is_success" in info:
                        success = success or _scalar_bool(info["is_success"])
                    if "final_info" in info:
                        final_info = info["final_info"]
                        if isinstance(final_info, dict) and "is_success" in final_info:
                            success = success or _scalar_bool(final_info["is_success"])
                    if _scalar_bool(terminated) or _scalar_bool(truncated):
                        break

                if trace.count != step or policy_calls != step:
                    raise RuntimeError(
                        "trace/call mismatch: "
                        f"trace={trace.count}, calls={policy_calls}, steps={step}"
                    )
                chunks_end = getattr(policy, "n_chunks_generated", None)
                chunk_delta = (
                    int(chunks_end) - int(chunks_start)
                    if isinstance(chunks_start, (int, np.integer))
                    and isinstance(chunks_end, (int, np.integer))
                    else None
                )
                episodes.append(
                    {
                        "task_id": task_id,
                        "state_id": realized,
                        "env_seed": env_seed,
                        "policy_seed": policy_seed,
                        "success": success,
                        "steps": step,
                        "policy_calls": policy_calls,
                        "policy_chunk_generations": chunk_delta,
                        "executed_action_count": trace.count,
                        "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
                        "executed_action_trace_sha256": trace.hexdigest(),
                    }
                )
                print(
                    f"task={task_id} state={realized} success={int(success)} "
                    f"steps={step} trace={trace.hexdigest()[:12]}",
                    flush=True,
                )

            successes = [episode["success"] for episode in episodes]
            task_results.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "success_rate": float(np.mean(successes)),
                    "episodes": episodes,
                }
            )
            env.close()

    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except (RuntimeError, OSError) as error:
                print(f"WARNING: environment close failed: {error}", file=sys.stderr)

    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    config = json.loads(config_path.read_text())
    stats_path = None
    stats_hash = None
    stats_name = config.get("spline_stats_file_v2")
    if stats_name:
        candidate = policy_source.parent / stats_name
        if candidate.exists():
            stats_path = str(candidate)
            stats_hash = _sha256(candidate)

    total_successes = sum(
        int(episode["success"])
        for task in task_results
        for episode in task["episodes"]
    )
    total_episodes = len(task_results) * len(state_ids)
    result = {
        "schema_version": 2,
        "protocol": "libero_explicit_init_state_deterministic_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "suite": args.suite,
        "task_ids": task_ids,
        "control_frequency_hz": args.control_freq,
        "state_ids": state_ids,
        "seed_base": args.seed_base,
        "bootstrap_seed": bootstrap_seed,
        "seed_protocol": {
            "env": "seed_all_before_explicit_reset",
            "policy": "torch_and_cuda_reseed_immediately_after_env_reset",
            "formula": "seed_base + task_id * 1000 + state_id",
        },
        "determinism": _determinism_metadata(),
        "n_tasks": len(task_results),
        "n_episodes": total_episodes,
        "n_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "policy_source": str(policy_source),
        "policy_source_sha256": _sha256(policy_source),
        "env_source": str(env_source),
        "env_source_sha256": _sha256(env_source),
        "lerobot_commit": _git_commit(policy_source.parents[4]),
        "stats_path": stats_path,
        "stats_sha256": stats_hash,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "helper_path": str(Path(atomic_write_json_new.__code__.co_filename).resolve()),
        "helper_sha256": _sha256(
            Path(atomic_write_json_new.__code__.co_filename).resolve()
        ),
        "tasks": task_results,
    }
    atomic_write_json_new(output, result)
    print(
        f"saved {output}: {total_successes}/{total_episodes} "
        f"({100 * result['success_rate']:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
