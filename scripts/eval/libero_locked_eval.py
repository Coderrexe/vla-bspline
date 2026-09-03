"""Evaluate a checkpoint on an explicit, outcome-independent LIBERO state manifest.

The deployed LeRobot LIBERO environment increments its ``init_state_id`` on
every reset and also resets internally after successful ``step`` calls.  A
naive sequence of rollouts therefore visits different initial states for two
policies as soon as their success outcomes differ.  This evaluator explicitly
sets the state id before every episode and records the realized id, so paired
comparisons remain paired regardless of prior outcomes.

Only compact scalar traces are saved: success, episode length, policy calls,
state id, and seed.  No videos or frame trees are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


SELF_PACED_CLAUSES = {
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

# Median causal switch frames from the exact source demonstrations selected by
# evaluator task text. A valid release follows >=48 consecutive closed commands,
# and the switch is the next frame. Half-integer medians are rounded upward.
# t0: 33 demonstrations, median 139.0. t4: 38, median 106.5 -> 107.
FIXED_SWITCH_STEPS = {0: 139, 4: 107}


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


def _release_boundary(saw_closed: bool, previous_grip: float, grip: float) -> tuple[bool, bool]:
    """Return updated close history and whether this command is a release."""

    saw_closed = saw_closed or grip > 0
    return saw_closed, bool(saw_closed and previous_grip > 0 and grip <= 0)


def _base_env(sync_vec_env: Any) -> Any:
    """Return the single underlying LiberoEnv and fail on another topology."""

    envs = getattr(sync_vec_env, "envs", None)
    if envs is None or len(envs) != 1:
        raise RuntimeError(
            "Locked evaluation requires a one-environment SyncVectorEnv; "
            f"got {type(sync_vec_env)!r} with envs={envs!r}"
        )
    return getattr(envs[0], "unwrapped", envs[0])


def _set_and_reset(env: Any, state_id: int, seed: int) -> tuple[Any, Any, int]:
    base = _base_env(env)
    base.init_state_id = state_id
    observation, info = env.reset(seed=seed)
    realized = int(base.init_state_id) - int(base._reset_stride)
    if realized != state_id:
        raise RuntimeError(f"requested init state {state_id}, realized {realized}")
    return observation, info, realized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task_id", type=int, action="append", default=None,
                        help="evaluate only this suite task id (repeatable)")
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states_per_task", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument(
        "--language_mode",
        choices=(
            "compound", "event_clock", "event_clock_swap",
            "fixed_clock", "fixed_clock_swap", "first_only",
            # Backward compatibility for the engineering-only v1 smoke.
            "selfpaced", "selfpaced_swap",
        ),
        default="compound",
        help="fixed benchmark prompt or gripper-release-steered two-clause program",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start must be nonnegative and states_per_task positive")

    checkpoint = Path(args.ckpt).resolve()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

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
    selected_task_ids = sorted(suite_envs) if not args.task_id else sorted(set(args.task_id))
    unknown = sorted(set(selected_task_ids) - set(suite_envs))
    if unknown:
        raise ValueError(f"unknown task ids for {args.suite}: {unknown}")
    if args.language_mode != "compound":
        unsupported = sorted(set(selected_task_ids) - set(SELF_PACED_CLAUSES))
        if unsupported:
            raise ValueError(
                f"language_mode={args.language_mode} has no reviewed clauses for tasks {unsupported}"
            )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    task_results: list[dict[str, Any]] = []
    started = time.time()

    try:
        for task_id in selected_task_ids:
            env = suite_envs[task_id]
            max_steps = int(env.call("_max_episode_steps")[0])
            task_description = str(list(env.call("task_description"))[0])
            expected_description = EXPECTED_TASK_DESCRIPTIONS.get(task_id)
            if expected_description is not None and task_description != expected_description:
                raise RuntimeError(
                    f"suite task {task_id} description mismatch: got "
                    f"{task_description!r}, expected {expected_description!r}"
                )
            clauses = list(SELF_PACED_CLAUSES.get(task_id, []))
            if args.language_mode.endswith("_swap"):
                clauses.reverse()
            episodes: list[dict[str, Any]] = []

            for state_id in state_ids:
                # Lock both simulator randomness and the policy's stochastic
                # flow-sampling path. Historical evals seeded neither policy
                # call-by-call nor the CUDA generator, so matching initial
                # states did not by itself make their outcomes paired.
                seed = args.seed_base + task_id * 1_000 + state_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                observation, _, realized = _set_and_reset(env, state_id, seed)
                policy.reset()
                success = False
                step = 0
                clause_index = 0
                saw_closed = False
                previous_grip = -1.0
                switch_steps: list[int] = []
                prompts_used: list[str] = []

                while step < max_steps:
                    if (
                        args.language_mode in ("fixed_clock", "fixed_clock_swap")
                        and clause_index == 0
                        and step >= FIXED_SWITCH_STEPS[task_id]
                    ):
                        clause_index = 1
                        switch_steps.append(step)
                        queues = getattr(policy, "_queues", None)
                        if isinstance(queues, dict) and ACTION in queues:
                            queues[ACTION].clear()
                    processed = preprocess_observation(observation)
                    if args.language_mode == "compound":
                        prompt = task_description
                    else:
                        prompt = clauses[clause_index]
                    if not prompts_used or prompts_used[-1] != prompt:
                        prompts_used.append(prompt)
                    processed["task"] = [prompt]
                    processed = env_preprocessor(processed)
                    processed = preprocessor(processed)
                    with torch.inference_mode():
                        action = policy.select_action(processed)
                    action = postprocessor(action)
                    transition = env_postprocessor({ACTION: action})
                    action_numpy = transition[ACTION].detach().cpu().numpy()
                    observation, reward, terminated, truncated, info = env.step(action_numpy)
                    step += 1

                    if args.language_mode in (
                        "event_clock", "event_clock_swap", "selfpaced", "selfpaced_swap"
                    ) and clause_index == 0:
                        grip = float(np.asarray(action_numpy).reshape(-1)[6])
                        saw_closed, released = _release_boundary(
                            saw_closed, previous_grip, grip
                        )
                        if released:
                            clause_index = 1
                            switch_steps.append(step)
                            # The release is the causal boundary. Discard any
                            # residual actions conditioned on clause 0 so the
                            # next control step is generated from clause 1.
                            queues = getattr(policy, "_queues", None)
                            if isinstance(queues, dict) and ACTION in queues:
                                queues[ACTION].clear()
                        previous_grip = grip

                    success = success or bool(np.asarray(reward).max() >= 1.0)
                    if "is_success" in info:
                        success = success or _scalar_bool(info["is_success"])
                    if "final_info" in info:
                        final_info = info["final_info"]
                        if isinstance(final_info, dict) and "is_success" in final_info:
                            success = success or _scalar_bool(final_info["is_success"])
                    if _scalar_bool(terminated) or _scalar_bool(truncated):
                        break

                episodes.append(
                    {
                        "state_id": realized,
                        "seed": seed,
                        "success": success,
                        "steps": step,
                        "policy_calls": int(getattr(policy, "n_chunks_generated", -1)),
                        "language_mode": args.language_mode,
                        "clauses": clauses,
                        "prompts_used": prompts_used,
                        "switch_steps": switch_steps,
                        "final_clause_index": clause_index,
                        "fixed_switch_step": FIXED_SWITCH_STEPS.get(task_id),
                    }
                )
                print(
                    f"task={task_id} state={realized} success={int(success)} "
                    f"steps={step}",
                    flush=True,
                )

            successes = [episode["success"] for episode in episodes]
            task_results.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "reviewed_clauses": clauses,
                    "success_rate": float(np.mean(successes)),
                    "episodes": episodes,
                }
            )
            env.close()

    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    env_source = Path(inspect.getfile(type(_base_env(next(iter(suite_envs.values())))))).resolve()
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
        "schema_version": 1,
        "protocol": "libero_explicit_init_state_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "suite": args.suite,
        "task_ids": selected_task_ids,
        "language_mode": args.language_mode,
        "control_frequency_hz": args.control_freq,
        "state_ids": state_ids,
        "seed_base": args.seed_base,
        "n_tasks": len(task_results),
        "n_episodes": total_episodes,
        "n_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "policy_source": str(policy_source),
        "policy_source_sha256": _sha256(policy_source),
        "env_source": str(env_source),
        "env_source_sha256": _sha256(env_source),
        "lerobot_commit": _git_commit(policy_source.parents[4]),
        "stats_path": stats_path,
        "stats_sha256": stats_hash,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "tasks": task_results,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
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
