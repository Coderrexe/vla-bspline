"""Locked-state LIBERO evaluation with decode-time duration interventions.

This is intentionally a separate evaluator: the canonical evaluator may be in
use by running jobs and must remain immutable.  Run ``predicted`` first, then
pass that result to ``shuffled`` so every shuffled episode preserves the
duration multiset predicted for the same task and initial state.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duration_intervention import (
    DurationIntervention,
    load_predicted_traces,
    load_task_duration_map,
    sha256,
)
from libero_locked_eval import _base_env, _git_commit, _scalar_bool, _set_and_reset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


def _validate_source(
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    suite: str,
    state_ids: list[int],
    seed_base: int,
) -> None:
    expected = {
        "model_sha256": sha256(checkpoint / "model.safetensors"),
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "suite": suite,
        "state_ids": state_ids,
        "seed_base": seed_base,
    }
    mismatches = {
        key: {"expected": value, "source": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"shuffle source is not a matched predicted run: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states_per_task", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument(
        "--duration_mode",
        required=True,
        choices=("predicted", "fixed", "fixed_task", "shuffled"),
    )
    parser.add_argument("--fixed_duration", type=int)
    parser.add_argument("--fixed_manifest")
    parser.add_argument("--task_duration_map")
    parser.add_argument("--shuffle_source")
    parser.add_argument("--shuffle_seed", type=int, default=73_921)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start must be nonnegative and states_per_task positive")
    if args.duration_mode == "fixed" and (args.fixed_duration is None or not args.fixed_manifest):
        raise ValueError("fixed mode requires --fixed_duration and --fixed_manifest")
    if args.duration_mode == "fixed_task" and not args.task_duration_map:
        raise ValueError("fixed_task mode requires --task_duration_map")
    if args.duration_mode == "shuffled" and not args.shuffle_source:
        raise ValueError("shuffled mode requires --shuffle_source from predicted mode")

    checkpoint = Path(args.ckpt).resolve()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint}")

    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    task_map: dict[str, int] | None = None
    task_map_path: Path | None = None
    source_traces: dict[tuple[int, int], list[int]] | None = None
    shuffle_payload: dict[str, Any] | None = None
    shuffle_path: Path | None = None
    fixed_manifest_path: Path | None = None
    if args.fixed_manifest:
        fixed_manifest_path = Path(args.fixed_manifest).resolve()
        fixed_payload = json.loads(fixed_manifest_path.read_text())
        if fixed_payload.get("complete_training_scan") is not True:
            raise ValueError("fixed-duration manifest must certify complete_training_scan=true")
        if fixed_payload.get("checkpoint_config_sha256") != sha256(config_path):
            raise ValueError("fixed-duration manifest was computed for a different checkpoint config")
        if int(fixed_payload.get("global_duration", -1)) != args.fixed_duration:
            raise ValueError("--fixed_duration does not match the source manifest")
    if args.task_duration_map:
        task_map_path = Path(args.task_duration_map).resolve()
        task_payload = json.loads(task_map_path.read_text())
        if task_payload.get("complete_training_scan") is not True:
            raise ValueError("task-duration map must certify complete_training_scan=true")
        if task_payload.get("checkpoint_config_sha256") != sha256(config_path):
            raise ValueError("task-duration map was computed for a different checkpoint config")
        task_map = load_task_duration_map(task_map_path)
    if args.shuffle_source:
        shuffle_path = Path(args.shuffle_source).resolve()
        source_traces, shuffle_payload = load_predicted_traces(shuffle_path)
        _validate_source(
            shuffle_payload,
            checkpoint=checkpoint,
            suite=args.suite,
            state_ids=state_ids,
            seed_base=args.seed_base,
        )

    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    if not bool(getattr(policy_cfg, "predict_duration", False)):
        raise ValueError("checkpoint is not a duration-enabled spline policy")
    # Isolate the basic time-allocation mechanism.  These optional controllers
    # also consume duration and would turn this into a bundled-system ablation.
    nonneutral = {
        "speedup_alpha": getattr(policy_cfg, "speedup_alpha", 1.0) != 1.0,
        "slowdown_alpha": getattr(policy_cfg, "slowdown_alpha", 1.0) != 1.0,
        "ease_out": getattr(policy_cfg, "ease_out", None) is not None,
        "profile_alpha": getattr(policy_cfg, "profile_alpha", None) is not None,
        "profile_slow_alpha": getattr(policy_cfg, "profile_slow_alpha", None) is not None,
        "replan_frac": getattr(policy_cfg, "replan_frac", None) is not None,
        "replan_margin": getattr(policy_cfg, "replan_margin", None) is not None,
        "feasibility_stretch": bool(getattr(policy_cfg, "feasibility_stretch", False)),
        "exec_rate_ratio": getattr(policy_cfg, "exec_rate_ratio", 1.0) != 1.0,
    }
    active = sorted(name for name, value in nonneutral.items() if value)
    if active:
        raise ValueError(f"duration mechanism ablation requires neutral decode knobs; active={active}")
    env_cfg = LiberoEnvCfg(
        task=args.suite,
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    envs_by_suite = make_env(env_cfg, n_envs=1)
    suite_envs = envs_by_suite[args.suite]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    controller = DurationIntervention(
        args.duration_mode,
        fixed_duration=args.fixed_duration,
        task_durations=task_map,
        source_traces=source_traces,
        shuffle_seed=args.shuffle_seed,
    )
    controller.attach(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    task_results: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task_id in sorted(suite_envs):
            env = suite_envs[task_id]
            max_steps = int(env.call("_max_episode_steps")[0])
            task_description = str(list(env.call("task_description"))[0])
            episodes: list[dict[str, Any]] = []
            for state_id in state_ids:
                seed = args.seed_base + task_id * 1_000 + state_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                observation, _, realized = _set_and_reset(env, state_id, seed)
                policy.reset()
                controller.begin_episode(task_id, state_id, task_description)
                success = False
                step = 0
                while step < max_steps:
                    processed = preprocess_observation(observation)
                    processed["task"] = [task_description]
                    processed = env_preprocessor(processed)
                    processed = preprocessor(processed)
                    with torch.inference_mode():
                        action = policy.select_action(processed)
                    action = postprocessor(action)
                    transition = env_postprocessor({ACTION: action})
                    action_numpy = transition[ACTION].detach().cpu().numpy()
                    observation, reward, terminated, truncated, info = env.step(action_numpy)
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
                duration_record = controller.end_episode()
                episodes.append(
                    {
                        "state_id": realized,
                        "seed": seed,
                        "success": success,
                        "steps": step,
                        "policy_calls": int(getattr(policy, "n_chunks_generated", -1)),
                        **duration_record,
                    }
                )
                print(
                    f"mode={args.duration_mode} task={task_id} state={realized} "
                    f"success={int(success)} steps={step} calls={duration_record['n_duration_calls']}",
                    flush=True,
                )
            task_results.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "success_rate": float(np.mean([ep["success"] for ep in episodes])),
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

    policy_source = Path(inspect.getfile(type(policy))).resolve()
    env_source = Path(inspect.getfile(type(_base_env(next(iter(suite_envs.values())))))).resolve()
    evaluator_path = Path(__file__).resolve()
    controller_path = Path(inspect.getfile(DurationIntervention)).resolve()
    config = json.loads(config_path.read_text())
    stats_path = None
    stats_hash = None
    if not config.get("embedded_spline_stats"):
        stats_name = config.get("spline_stats_file_v2")
        if stats_name:
            candidate = policy_source.parent / stats_name
            if candidate.exists():
                stats_path = str(candidate)
                stats_hash = sha256(candidate)
    total_successes = sum(int(ep["success"]) for task in task_results for ep in task["episodes"])
    total_episodes = sum(len(task["episodes"]) for task in task_results)
    all_executed = [
        duration
        for task in task_results
        for episode in task["episodes"]
        for duration in episode["executed_durations"]
    ]
    all_predicted = [
        duration
        for task in task_results
        for episode in task["episodes"]
        for duration in episode["predicted_durations"]
    ]
    all_absolute_interventions = [
        abs(executed - predicted)
        for task in task_results
        for episode in task["episodes"]
        for predicted, executed in zip(
            episode["predicted_durations"],
            episode["executed_durations"],
            strict=True,
        )
    ]
    intervened_calls = sum(value > 0 for value in all_absolute_interventions)
    reused_counterfactual_values = sum(
        episode["reused_counterfactual_values"]
        for task in task_results
        for episode in task["episodes"]
    )
    result = {
        "schema_version": 1,
        "protocol": "libero_duration_intervention_locked_state_v1",
        "duration_mode": args.duration_mode,
        "duration_estimand": "closed-loop total effect of observation-aligned duration at fixed shape-head checkpoint",
        "fixed_duration": args.fixed_duration,
        "fixed_manifest": str(fixed_manifest_path) if fixed_manifest_path else None,
        "fixed_manifest_sha256": sha256(fixed_manifest_path) if fixed_manifest_path else None,
        "task_duration_map": str(task_map_path) if task_map_path else None,
        "task_duration_map_sha256": sha256(task_map_path) if task_map_path else None,
        "shuffle_source": str(shuffle_path) if shuffle_path else None,
        "shuffle_source_sha256": sha256(shuffle_path) if shuffle_path else None,
        "shuffle_seed": args.shuffle_seed if args.duration_mode == "shuffled" else None,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256(config_path),
        "model_sha256": sha256(model_path),
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "suite": args.suite,
        "control_frequency_hz": args.control_freq,
        "state_ids": state_ids,
        "seed_base": args.seed_base,
        "n_tasks": len(task_results),
        "n_episodes": total_episodes,
        "n_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "n_duration_calls": len(all_executed),
        "n_intervened_duration_calls": intervened_calls,
        "intervention_call_fraction": intervened_calls / len(all_executed),
        "mean_absolute_duration_intervention": float(
            np.mean(all_absolute_interventions)
        ),
        "reused_counterfactual_values": reused_counterfactual_values,
        "mean_predicted_duration": float(np.mean(all_predicted)),
        "mean_executed_duration": float(np.mean(all_executed)),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "policy_source": str(policy_source),
        "policy_source_sha256": sha256(policy_source),
        "stats_embedded_in_config": bool(config.get("embedded_spline_stats")),
        "legacy_stats_path": stats_path,
        "legacy_stats_sha256": stats_hash,
        "env_source": str(env_source),
        "env_source_sha256": sha256(env_source),
        "lerobot_commit": _git_commit(policy_source.parents[4]),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256(evaluator_path),
        "controller_path": str(controller_path),
        "controller_sha256": sha256(controller_path),
        "tasks": task_results,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(f"saved {output}: {total_successes}/{total_episodes}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
