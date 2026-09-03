#!/usr/bin/env python3
"""Exact paired evaluation of a same-scene held-out LIBERO composition.

The policy is trained only on t0 (soup+tomato) and t1 (cream-cheese+butter).
At evaluation, t0's Scene-2 simulator is reset exactly as usual, but its goal
conjunction is changed to soup+cream-cheese. Both objects already exist in the
scene at the same regions used by t0/t1. Whole and clause checkpoints receive
the same two-phase replanning/oracle structure; only their language differs.
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
    _scalar_bool,
    _seed_all,
    _sha256,
)


PROTOCOL = "libero_same_scene_compositional_holdout_v1"
SUITE_TASK_ID = 0
SOURCE_DESCRIPTION = "put both the alphabet soup and the tomato sauce in the basket"
HELD_OUT_DESCRIPTION = "put both the alphabet soup and the cream cheese box in the basket"
SOURCE_GOALS = [
    ["in", "alphabet_soup_1", "basket_1_contain_region"],
    ["in", "tomato_sauce_1", "basket_1_contain_region"],
]
HELD_OUT_GOALS = [
    ["in", "alphabet_soup_1", "basket_1_contain_region"],
    ["in", "cream_cheese_1", "basket_1_contain_region"],
]
HELD_OUT_CLAUSES = [
    "put the alphabet soup in the basket",
    "put the cream cheese box in the basket",
]
SEEN_T1_DESCRIPTION = "put both the cream cheese box and the butter in the basket"
SEEN_T1_GOALS = [
    ["in", "cream_cheese_1", "basket_1_contain_region"],
    ["in", "butter_1", "basket_1_contain_region"],
]
SEEN_T1_CLAUSES = [
    "put the cream cheese box in the basket",
    "put the butter in the basket",
]
TASK_VARIANTS = {
    "heldout": {
        "protocol": PROTOCOL,
        "source_task_id": 0,
        "source_description": SOURCE_DESCRIPTION,
        "source_goals": SOURCE_GOALS,
        "requested_description": HELD_OUT_DESCRIPTION,
        "requested_goals": HELD_OUT_GOALS,
        "clauses": HELD_OUT_CLAUSES,
        "scene_control": (
            "official_libero_long_t0_scene2; only terminal goal conjunction changed"
        ),
        "transition_oracle": "held_out_current_subgoal_predicate_true_then_clear_queue_v1",
    },
    "seen_t0": {
        "protocol": "libero_compositional_seen_competence_v1",
        "source_task_id": 0,
        "source_description": SOURCE_DESCRIPTION,
        "source_goals": SOURCE_GOALS,
        "requested_description": SOURCE_DESCRIPTION,
        "requested_goals": SOURCE_GOALS,
        "clauses": [
            "put the alphabet soup in the basket",
            "put the tomato sauce in the basket",
        ],
        "scene_control": "official_libero_long_t0_scene2; goal conjunction unchanged",
        "transition_oracle": "seen_current_subgoal_predicate_true_then_clear_queue_v1",
    },
    "seen_t1": {
        "protocol": "libero_compositional_seen_competence_v1",
        "source_task_id": 1,
        "source_description": SEEN_T1_DESCRIPTION,
        "source_goals": SEEN_T1_GOALS,
        "requested_description": SEEN_T1_DESCRIPTION,
        "requested_goals": SEEN_T1_GOALS,
        "clauses": SEEN_T1_CLAUSES,
        "scene_control": "official_libero_long_t1_scene2; goal conjunction unchanged",
        "transition_oracle": "seen_current_subgoal_predicate_true_then_clear_queue_v1",
    },
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
    for candidate, path in _walk_wrappers(environment):
        parsed = getattr(candidate, "parsed_problem", None)
        evaluator = getattr(candidate, "_eval_predicate", None)
        if isinstance(parsed, dict) and callable(evaluator):
            if isinstance(parsed.get("goal_state"), list):
                return candidate, path
    raise RuntimeError("could not locate LIBERO parsed_problem/_eval_predicate")


def _canonical_goals(states: Any) -> list[list[str]]:
    if not isinstance(states, list):
        raise ValueError(f"goal state must be a list, got {states!r}")
    goals = []
    for state in states:
        if (
            not isinstance(state, (list, tuple))
            or len(state) not in (2, 3)
            or not all(isinstance(item, str) and item for item in state)
        ):
            raise ValueError(f"unsupported LIBERO goal predicate: {state!r}")
        goals.append(list(state))
    return goals


def validate_source_and_holdout_goals(source: Any, held_out: Any) -> None:
    if _canonical_goals(source) != SOURCE_GOALS:
        raise RuntimeError(f"suite t0 goals changed: {source!r}")
    if _canonical_goals(held_out) != HELD_OUT_GOALS:
        raise RuntimeError(f"held-out goal specification changed: {held_out!r}")
    if source[0] != held_out[0] or source[1] == held_out[1]:
        raise RuntimeError("holdout must retain goal 0 and replace only goal 1")


def _goal_flags(problem_env: Any, goals: list[list[str]]) -> list[bool]:
    return [bool(problem_env._eval_predicate(goal)) for goal in goals]


def _clear_action_queue(policy: Any, action_key: str) -> None:
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and action_key in queues:
        queues[action_key].clear()


def _first_completed(first_hit_steps: list[int | None]) -> list[int]:
    observed = [step for step in first_hit_steps if step is not None]
    if not observed:
        return []
    earliest = min(observed)
    return [index for index, step in enumerate(first_hit_steps) if step == earliest]


def _prompt(
    language_mode: str,
    requested_goal: int,
    description: str = HELD_OUT_DESCRIPTION,
    clauses: list[str] = HELD_OUT_CLAUSES,
) -> str:
    if language_mode == "whole":
        return description
    if language_mode == "clause":
        return clauses[requested_goal]
    raise ValueError(language_mode)


def _video_frame(observation: dict[str, Any]) -> np.ndarray:
    candidates: list[tuple[int, str, np.ndarray]] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}/{key}", child)
            return
        array = np.asarray(value)
        if array.ndim not in (3, 4) or not any(
            token in path.lower() for token in ("image", "pixel", "camera")
        ):
            return
        wrist = int(any(token in path.lower() for token in ("wrist", "image2", "eye_in_hand")))
        candidates.append((wrist, path, array))

    visit("observation", observation)
    if not candidates:
        raise RuntimeError("no camera-valued observation leaf available for video")
    frame = min(candidates, key=lambda item: (item[0], item[1]))[2]
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise RuntimeError(f"video recorder requires batch size one, got {frame.shape}")
        frame = frame[0]
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame = frame * 255.0 if float(np.nanmax(frame)) <= 1.0 else frame
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--language_mode", choices=("whole", "clause"), required=True)
    parser.add_argument(
        "--task_variant",
        choices=tuple(TASK_VARIANTS),
        default="heldout",
        help="held-out recombination or an in-distribution competence control",
    )
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--seed_base", type=int, default=200_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--replicate_index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video_dir", type=Path)
    args = parser.parse_args()
    if args.state_start < 0 or args.states <= 0:
        raise ValueError("state_start must be nonnegative and states positive")
    if args.replicate_index < 0:
        raise ValueError("replicate_index must be nonnegative")
    if os.environ.get("MUJOCO_GL", "").lower() != "osmesa":
        raise ValueError("claim evaluation requires MUJOCO_GL=osmesa")
    variant = TASK_VARIANTS[args.task_variant]

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
        task="libero_10",
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    suite_envs = make_env(env_cfg, n_envs=1)["libero_10"]
    env = suite_envs[variant["source_task_id"]]
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

    states = list(range(args.state_start, args.state_start + args.states))
    episodes: list[dict[str, Any]] = []
    paired_initials: list[dict[str, Any]] = []
    max_steps = int(env.call("_max_episode_steps")[0])
    description = str(next(iter(env.call("task_description"))))
    if description != variant["source_description"]:
        raise RuntimeError(
            f"suite t{variant['source_task_id']} description changed: {description!r}"
        )
    env_source = Path(inspect.getfile(type(_base_env(env)))).resolve()
    started = time.time()
    try:
        for state_id in states:
            condition_initials: dict[str, dict[str, str]] = {}
            for condition, goal_order in (("normal", [0, 1]), ("reversed", [1, 0])):
                seed = args.seed_base + state_id
                observation, _, realized = _explicit_reset(env, state_id, seed, seed)
                base = _base_env(env)
                problem_env, problem_path = _locate_problem_env(base)
                simulator, simulator_path = locate_robosuite_sim(base)
                original_goals = _canonical_goals(problem_env.parsed_problem["goal_state"])
                if original_goals != variant["source_goals"]:
                    raise RuntimeError(
                        f"suite t{variant['source_task_id']} goals changed: {original_goals!r}"
                    )
                if args.task_variant == "heldout":
                    validate_source_and_holdout_goals(
                        original_goals, variant["requested_goals"]
                    )
                initial_flags = _goal_flags(problem_env, variant["requested_goals"])
                if any(initial_flags):
                    raise RuntimeError(f"state {state_id} starts with a requested goal true")
                initial = {
                    "raw_observation": _structured_sha256(observation),
                    "mujoco_integration_state": integration_state_sha256(simulator),
                    "compiled_model_xml": model_xml_sha256(simulator),
                }
                condition_initials[condition] = initial
                if len(condition_initials) == 2 and condition_initials["normal"] != initial:
                    raise RuntimeError(f"normal/reversed reset mismatch at state {state_id}")

                # Make the counterfactual conjunction the environment's own
                # terminal predicate so wrapper reward/autoreset semantics remain valid.
                problem_env.parsed_problem["goal_state"] = [
                    list(goal) for goal in variant["requested_goals"]
                ]
                policy.reset()
                video_frames = [_video_frame(observation)] if args.video_dir else []
                trace = ExecutedActionTrace()
                prompts_used: list[str] = []
                first_hit_steps: list[int | None] = [None, None]
                ever_false_after_hit = [False, False]
                predicate_trace = [{"step": 0, "flags": initial_flags}]
                switch_steps: list[int] = []
                prompt_index = 0
                current_flags = initial_flags
                success = False
                step = 0
                first_action_sha256: str | None = None
                try:
                    while step < max_steps:
                        requested_goal = goal_order[prompt_index]
                        prompt = _prompt(
                            args.language_mode,
                            requested_goal,
                            variant["requested_description"],
                            variant["clauses"],
                        )
                        prompts_used.append(prompt)
                        processed = preprocess_observation(observation)
                        processed["task"] = [prompt]
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
                        if args.video_dir:
                            final_observation = info.get("final_observation")
                            video_frames.append(
                                _video_frame(
                                    final_observation
                                    if isinstance(final_observation, dict)
                                    else observation
                                )
                            )
                        new_flags = [True, True] if step_success else _goal_flags(
                            problem_env, variant["requested_goals"]
                        )
                        if new_flags != current_flags:
                            predicate_trace.append({"step": step, "flags": new_flags})
                        for index, (before, after) in enumerate(zip(current_flags, new_flags)):
                            if not before and after and first_hit_steps[index] is None:
                                first_hit_steps[index] = step
                            if first_hit_steps[index] is not None and not after:
                                ever_false_after_hit[index] = True
                        current_flags = new_flags
                        if prompt_index == 0 and current_flags[goal_order[0]]:
                            prompt_index = 1
                            switch_steps.append(step)
                            _clear_action_queue(policy, ACTION)
                        success = success or step_success or all(current_flags)
                        if success or _scalar_bool(terminated) or _scalar_bool(truncated):
                            break
                finally:
                    problem_env.parsed_problem["goal_state"] = original_goals

                first_completed = _first_completed(first_hit_steps)
                video_path = None
                if args.video_dir:
                    import imageio.v2 as imageio

                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = args.video_dir / (
                        f"state_{state_id:02d}_{condition}_{args.language_mode}.mp4"
                    )
                    imageio.mimwrite(
                        video_path,
                        video_frames,
                        fps=args.control_freq,
                        codec="libx264",
                        quality=7,
                    )
                episodes.append(
                    {
                        "source_suite_task_id": variant["source_task_id"],
                        "source_task_description": description,
                        "requested_task_description": variant["requested_description"],
                        "task_variant": args.task_variant,
                        "state_id": realized,
                        "condition": condition,
                        "language_mode": args.language_mode,
                        "goal_order": goal_order,
                        "prompts_used_unique_ordered": list(dict.fromkeys(prompts_used)),
                        "source_goal_predicates": variant["source_goals"],
                        "held_out_goal_predicates": variant["requested_goals"],
                        "initial_goal_flags": initial_flags,
                        "first_goal_hit_steps": first_hit_steps,
                        "first_completed_goal_indices": first_completed,
                        "requested_first_goal_completed_first": first_completed == [goal_order[0]],
                        "switch_steps": switch_steps,
                        "final_goal_flags": current_flags,
                        "first_goal_retained_final": bool(current_flags[goal_order[0]]),
                        "goal_ever_false_after_first_hit": ever_false_after_hit,
                        "predicate_trace": predicate_trace,
                        "success": bool(success),
                        "steps": step,
                        "env_seed_u32": seed,
                        "policy_seed_u64": seed,
                        "initial_raw_observation_sha256": initial["raw_observation"],
                        "initial_mujoco_integration_state_sha256": initial[
                            "mujoco_integration_state"
                        ],
                        "compiled_model_xml_sha256": initial["compiled_model_xml"],
                        "hidden_state_protocol": HIDDEN_STATE_PROTOCOL,
                        "problem_env_path": problem_path,
                        "simulator_path": simulator_path,
                        "first_action_sha256": first_action_sha256,
                        "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
                        "executed_action_trace_sha256": trace.hexdigest(),
                        "diagnostic_video": str(video_path) if video_path else None,
                    }
                )
                print(
                    f"state={state_id} condition={condition} first={first_completed} "
                    f"success={int(success)} steps={step}",
                    flush=True,
                )
            paired_initials.append(
                {
                    "state_id": state_id,
                    "passed": condition_initials["normal"] == condition_initials["reversed"],
                    "hashes": condition_initials["normal"],
                }
            )
    finally:
        for candidate in suite_envs.values():
            try:
                candidate.close()
            except Exception:
                pass

    config = json.loads(config_path.read_text())
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    summary = {
        condition: {
            "n": sum(row["condition"] == condition for row in episodes),
            "requested_first": sum(
                row["condition"] == condition and row["requested_first_goal_completed_first"]
                for row in episodes
            ),
            "successes": sum(
                row["condition"] == condition and row["success"] for row in episodes
            ),
        }
        for condition in ("normal", "reversed")
    }
    payload = {
        "schema_version": 1,
        "protocol": variant["protocol"],
        "transition_oracle": variant["transition_oracle"],
        "transition_oracle_is_privileged": True,
        "scene_control": variant["scene_control"],
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "language_mode": args.language_mode,
        "task_variant": args.task_variant,
        "renderer_backend": "osmesa",
        "replicate_index": args.replicate_index,
        "suite": "libero_10",
        "source_task_id": variant["source_task_id"],
        "state_ids": states,
        "seed_base": args.seed_base,
        "control_frequency_hz": args.control_freq,
        "paired_initial_state_checks": paired_initials,
        "episodes": episodes,
        "n_episodes": len(episodes),
        "summary": summary,
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
