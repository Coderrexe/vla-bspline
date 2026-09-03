#!/usr/bin/env python3
"""Paired, replayable LIBERO t0/t4 clause-order steering evaluation.

The intervention changes only the order of the two reviewed language clauses.
For each explicit LIBERO initial state, the normal and reversed programs use the
same environment and policy seeds.  The program advances only when LIBERO's
official simulator predicate for the requested subgoal becomes true.  This is
an intentionally privileged transition oracle: the claim tested here is which
subgoal the language-conditioned policy completes first, not learned boundary
detection.

No installed package is modified.  Outputs are atomically published and refuse
overwrite; a Slurm launcher runs two fresh-process replicates and gates exact
initial state, action trace, predicate telemetry, and outcome replay.
"""

from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deterministic_eval import ACTION_TRACE_PROTOCOL, ExecutedActionTrace, atomic_write_json_new
from libero_duration_ablation_v2 import _structured_sha256
from libero_hidden_state import (
    PROTOCOL as HIDDEN_STATE_PROTOCOL,
    integration_state_sha256,
    locate_robosuite_sim,
    model_xml_sha256,
)
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


PROTOCOL = "libero_t04_paired_clause_order_predicate_oracle_v1"
EXPECTED_TASK_DESCRIPTIONS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    4: (
        "put the white mug on the left plate and put the yellow and white mug "
        "on the right plate"
    ),
}
CLAUSES = {
    0: [
        "put the alphabet soup in the basket",
        "put the tomato sauce in the basket",
    ],
    4: [
        "put the white mug on the left plate",
        "put the yellow and white mug on the right plate",
    ],
}


def _walk_wrappers(root: Any) -> list[tuple[Any, str]]:
    queue: list[tuple[Any, str]] = [(root, "root")]
    found: list[tuple[Any, str]] = []
    seen: set[int] = set()
    while queue:
        candidate, path = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        found.append((candidate, path))
        for attribute in ("_env", "env", "unwrapped"):
            try:
                child = getattr(candidate, attribute, None)
            except Exception:
                child = None
            if child is not None and child is not candidate:
                queue.append((child, f"{path}.{attribute}"))
    return found


def _locate_problem_env(environment: Any) -> tuple[Any, str]:
    """Locate the LIBERO problem object exposing official goal predicates."""

    for candidate, path in _walk_wrappers(environment):
        parsed = getattr(candidate, "parsed_problem", None)
        evaluator = getattr(candidate, "_eval_predicate", None)
        if isinstance(parsed, dict) and callable(evaluator):
            goals = parsed.get("goal_state")
            if isinstance(goals, list):
                return candidate, path
    raise RuntimeError("could not locate LIBERO parsed_problem/_eval_predicate")


def _canonical_goal(state: Any) -> list[str]:
    if not isinstance(state, (list, tuple)) or len(state) not in (2, 3):
        raise ValueError(f"unsupported LIBERO goal predicate: {state!r}")
    if not all(isinstance(item, str) and item for item in state):
        raise ValueError(f"malformed LIBERO goal predicate: {state!r}")
    return list(state)


def _goal_flags(problem_env: Any, goals: list[list[str]]) -> list[bool]:
    return [bool(problem_env._eval_predicate(goal)) for goal in goals]


def _clear_action_queue(policy: Any, action_key: str) -> None:
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and action_key in queues:
        queues[action_key].clear()


def _episode_seed(seed_base: int, task_id: int, state_id: int) -> int:
    seed = seed_base + task_id * 1_000 + state_id
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError(f"derived seed is not uint32: {seed}")
    return seed


def _first_completed(first_hit_steps: list[int | None]) -> list[int]:
    observed = [step for step in first_hit_steps if step is not None]
    if not observed:
        return []
    earliest = min(observed)
    return [index for index, step in enumerate(first_hit_steps) if step == earliest]


def _validate_task_goals(task_id: int, goals: list[list[str]]) -> None:
    """Bind reviewed clauses to the two official independent goal predicates."""

    expected = {
        0: [
            ["in", "alphabet_soup_1", "basket_1_contain_region"],
            ["in", "tomato_sauce_1", "basket_1_contain_region"],
        ],
        4: [
            ["on", "porcelain_mug_1", "plate_1"],
            ["on", "white_yellow_mug_1", "plate_2"],
        ],
    }
    if task_id not in expected or goals != expected[task_id]:
        raise RuntimeError(
            f"suite t{task_id} official goals changed: got {goals!r}, "
            f"expected {expected.get(task_id)!r}"
        )


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task_ids", default="0,4")
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states_per_task", type=int, default=2)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--replicate_index", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start nonnegative; states_per_task positive")
    if args.replicate_index < 0:
        raise ValueError("replicate_index must be nonnegative")
    if os.environ.get("MUJOCO_GL", "").lower() != "osmesa":
        raise ValueError("this claim evaluator requires MUJOCO_GL=osmesa")

    checkpoint = Path(args.ckpt).resolve()
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    output = Path(args.out).resolve()
    for path in (config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    _seed_all(args.seed_base)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    env_cfg = LiberoEnvCfg(
        task="libero_10", control_mode="relative", control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    suite_envs = make_env(env_cfg, n_envs=1)["libero_10"]
    task_ids = _parse_task_ids(args.task_ids, list(suite_envs))
    if set(task_ids) != {0, 4}:
        raise ValueError("clause-order evaluator requires exactly suite tasks 0 and 4")

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    states = list(range(args.state_start, args.state_start + args.states_per_task))
    episodes: list[dict[str, Any]] = []
    paired_initial_state_checks: list[dict[str, Any]] = []
    env_source: Path | None = None
    started = time.time()
    try:
        for task_id in task_ids:
            env = suite_envs[task_id]
            max_steps = int(env.call("_max_episode_steps")[0])
            description = str(next(iter(env.call("task_description"))))
            if description != EXPECTED_TASK_DESCRIPTIONS[task_id]:
                raise RuntimeError(f"suite t{task_id} description changed: {description!r}")
            if env_source is None:
                env_source = Path(inspect.getfile(type(_base_env(env)))).resolve()

            for state_id in states:
                condition_initials: dict[str, dict[str, str]] = {}
                for order_name, goal_order in (("normal", [0, 1]), ("reversed", [1, 0])):
                    seed = _episode_seed(args.seed_base, task_id, state_id)
                    observation, _, realized = _explicit_reset(env, state_id, seed, seed)
                    base = _base_env(env)
                    problem_env, problem_path = _locate_problem_env(base)
                    simulator, simulator_path = locate_robosuite_sim(base)
                    goals = [_canonical_goal(goal) for goal in problem_env.parsed_problem["goal_state"]]
                    _validate_task_goals(task_id, goals)
                    initial_flags = _goal_flags(problem_env, goals)
                    if any(initial_flags):
                        raise RuntimeError(
                            f"t{task_id}/s{state_id} starts with a completed goal: {initial_flags}"
                        )
                    initial = {
                        "raw_observation": _structured_sha256(observation),
                        "mujoco_integration_state": integration_state_sha256(simulator),
                        "compiled_model_xml": model_xml_sha256(simulator),
                    }
                    condition_initials[order_name] = initial
                    if len(condition_initials) == 2 and condition_initials["normal"] != initial:
                        raise RuntimeError(
                            f"normal/reversed reset mismatch t{task_id}/s{state_id}: "
                            f"{condition_initials!r}"
                        )

                    policy.reset()
                    trace = ExecutedActionTrace()
                    first_hit_steps: list[int | None] = [None, None]
                    switch_steps: list[int] = []
                    predicate_trace: list[dict[str, Any]] = [
                        {"step": 0, "flags": initial_flags}
                    ]
                    prompt_index = 0
                    current_flags = initial_flags
                    success = False
                    step = 0
                    first_action_sha256: str | None = None
                    while step < max_steps:
                        prompt_goal_index = goal_order[prompt_index]
                        processed = preprocess_observation(observation)
                        processed["task"] = [CLAUSES[task_id][prompt_goal_index]]
                        processed = env_preprocessor(processed)
                        processed = preprocessor(processed)
                        with torch.inference_mode():
                            action = policy.select_action(processed)
                        action = postprocessor(action)
                        transition = env_postprocessor({ACTION: action})
                        action_numpy = transition[ACTION].detach().cpu().numpy()
                        if first_action_sha256 is None:
                            first_action_sha256 = hashlib.sha256(
                                action_numpy.tobytes(order="C")
                            ).hexdigest()
                        trace.update(action_numpy)
                        observation, reward, terminated, truncated, info = env.step(action_numpy)
                        step += 1
                        step_success = bool(np.asarray(reward).max() >= 1.0)
                        if "is_success" in info:
                            step_success = step_success or _scalar_bool(info["is_success"])
                        final_info = info.get("final_info")
                        if isinstance(final_info, dict) and "is_success" in final_info:
                            step_success = step_success or _scalar_bool(final_info["is_success"])
                        # LeRobot's LIBERO wrapper resets inside step() on full
                        # success.  In that one case, its is_success value is
                        # the official conjunction evaluated before reset, so
                        # both goal flags were true at the terminal transition.
                        new_flags = (
                            [True] * len(goals)
                            if step_success
                            else _goal_flags(problem_env, goals)
                        )
                        if new_flags != current_flags:
                            predicate_trace.append({"step": step, "flags": new_flags})
                        for goal_index, (before, after) in enumerate(zip(current_flags, new_flags)):
                            if not before and after and first_hit_steps[goal_index] is None:
                                first_hit_steps[goal_index] = step
                        current_flags = new_flags
                        if prompt_index == 0 and current_flags[goal_order[0]]:
                            prompt_index = 1
                            switch_steps.append(step)
                            _clear_action_queue(policy, ACTION)
                        success = success or step_success
                        if _scalar_bool(terminated) or _scalar_bool(truncated):
                            break

                    final_flags = current_flags
                    official_success = success
                    if success and not all(final_flags):
                        raise RuntimeError(
                            f"reward/predicate success mismatch t{task_id}/s{state_id}/"
                            f"{order_name}: reward={success}, goals={final_flags}"
                        )
                    first_completed = _first_completed(first_hit_steps)
                    episodes.append(
                        {
                            "task_id": task_id,
                            "task_description": description,
                            "state_id": realized,
                            "condition": order_name,
                            "goal_order": goal_order,
                            "clause_order": [CLAUSES[task_id][index] for index in goal_order],
                            "official_goal_predicates": goals,
                            "initial_goal_flags": initial_flags,
                            "first_goal_hit_steps": first_hit_steps,
                            "first_completed_goal_indices": first_completed,
                            "first_completed_goal_predicates": [goals[index] for index in first_completed],
                            "requested_first_goal_completed_first": first_completed == [goal_order[0]],
                            "switch_steps": switch_steps,
                            "final_goal_flags": final_flags,
                            "predicate_trace": predicate_trace,
                            "success": official_success,
                            "steps": step,
                            "env_seed_u32": seed,
                            "policy_seed_u64": seed,
                            "initial_raw_observation_sha256": initial["raw_observation"],
                            "initial_mujoco_integration_state_sha256": initial["mujoco_integration_state"],
                            "compiled_model_xml_sha256": initial["compiled_model_xml"],
                            "hidden_state_protocol": HIDDEN_STATE_PROTOCOL,
                            "problem_env_path": problem_path,
                            "simulator_path": simulator_path,
                            "first_action_sha256": first_action_sha256,
                            "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
                            "executed_action_trace_sha256": trace.hexdigest(),
                        }
                    )
                    print(
                        f"t={task_id} s={state_id} order={order_name} "
                        f"first={first_completed} requested={int(first_completed == [goal_order[0]])} "
                        f"success={int(official_success)} steps={step}", flush=True
                    )
                paired_initial_state_checks.append(
                    {
                        "task_id": task_id,
                        "state_id": state_id,
                        "passed": condition_initials["normal"] == condition_initials["reversed"],
                        "hashes": condition_initials["normal"],
                    }
                )
            env.close()
    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    if env_source is None:
        raise RuntimeError("no episodes executed")
    episode_index = {
        (row["task_id"], row["state_id"], row["condition"]): row for row in episodes
    }
    paired_order_effects: list[dict[str, Any]] = []
    for task_id in task_ids:
        for state_id in states:
            normal = episode_index[(task_id, state_id, "normal")]
            reversed_order = episode_index[(task_id, state_id, "reversed")]
            normal_first = normal["first_completed_goal_indices"]
            reversed_first = reversed_order["first_completed_goal_indices"]
            paired_order_effects.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "normal_first_completed_goal_indices": normal_first,
                    "reversed_first_completed_goal_indices": reversed_first,
                    "requested_order_flip": normal_first == [0] and reversed_first == [1],
                    "unique_first_goal_changed": (
                        len(normal_first) == 1
                        and len(reversed_first) == 1
                        and normal_first != reversed_first
                    ),
                    "normal_success": normal["success"],
                    "reversed_success": reversed_order["success"],
                }
            )
    config = json.loads(config_path.read_text())
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "transition_oracle": "official_current_subgoal_predicate_true_then_clear_queue_v1",
        "transition_oracle_is_privileged": True,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "renderer_backend": "osmesa",
        "replicate_index": args.replicate_index,
        "suite": "libero_10",
        "task_ids": task_ids,
        "state_ids": states,
        "seed_base": args.seed_base,
        "control_frequency_hz": args.control_freq,
        "paired_initial_state_checks": paired_initial_state_checks,
        "paired_order_effects": paired_order_effects,
        "episodes": episodes,
        "n_episodes": len(episodes),
        "summary": {
            condition: {
                "n": sum(row["condition"] == condition for row in episodes),
                "requested_first_goal_completed_first": sum(
                    row["condition"] == condition and row["requested_first_goal_completed_first"]
                    for row in episodes
                ),
                "successes": sum(
                    row["condition"] == condition and row["success"] for row in episodes
                ),
            }
            for condition in ("normal", "reversed")
        }
        | {
            "paired": {
                "n": len(paired_order_effects),
                "requested_order_flips": sum(
                    row["requested_order_flip"] for row in paired_order_effects
                ),
                "unique_first_goal_changes": sum(
                    row["unique_first_goal_changed"] for row in paired_order_effects
                ),
                "both_orders_successful": sum(
                    row["normal_success"] and row["reversed_success"]
                    for row in paired_order_effects
                ),
            }
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "determinism": _determinism_metadata(),
            "wall_time_seconds": time.time() - started,
        },
        "source": {
            "collector_sha256": _sha256(Path(__file__).resolve()),
            "policy_source_sha256": _sha256(policy_source),
            "env_source_sha256": _sha256(env_source),
            "lerobot_commit": _git_commit(policy_source.parents[4]),
        },
    }
    atomic_write_json_new(output, payload)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
