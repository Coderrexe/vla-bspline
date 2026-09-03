"""Failure-directed RoboCasa365 evaluation with task-specific progress traces.

This evaluator complements binary benchmark success.  It records the physical
predicates that comprise four atomic-adjacent composite tasks so a zero-success
checkpoint can be distinguished from one that never reaches, grasps, places,
or actuates the relevant fixture.  It is intentionally isolated from the
benchmark environment and never changes rewards or simulator state.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import re
import tempfile
from pathlib import Path
from typing import Any

# Required by deterministic CUDA matrix multiplication.  It must be set before
# the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from robocasa_video import video_frame
from robocasa_language_programs import (
    KETTLE_OFFICIAL_PROGRAM,
    RINSE_OFFICIAL_PROGRAM,
    RINSE_REGION_PROGRAM,
    kettle_fixed_program_pointer,
    kettle_burner_instruction,
    kettle_goal_program,
    kettle_goal_program_advance,
    kettle_program_advance,
    rinse_official_fixed_pointer,
    rinse_official_program_advance,
    rinse_program_advance,
)


SUPPORTED_TASKS = {
    "KettleBoiling",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "StackBowlsCabinet",
}
ORACLE_TASKS = {"KettleBoiling", "ScrubCuttingBoard"}
PROPRIO_TASKS = {"KettleBoiling"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_sha256(observation: dict[str, Any]) -> str:
    """Hash the exact reset observation to audit paired environment identity."""

    digest = hashlib.sha256()

    def update(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                update(f"{prefix}/{key}", value[key])
            return
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError(f"object-valued observation leaf at {prefix}")
        digest.update(prefix.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())

    update("observation", observation)
    return digest.hexdigest()


def _vector_env(envs: Any) -> Any:
    current = envs
    while isinstance(current, dict):
        current = next(iter(current.values()))
    return current


def _lock_robocasa_constructor(seed: int) -> None:
    """Make RoboCasa's one-time scene construction reproducible.

    LeRobot's RoboCasa wrapper constructs ``RoboCasaGymEnv`` lazily on the
    first reset, but does not pass that reset's seed into the constructor.
    RoboCasa samples layout/style/object instances during construction, before
    ``reset(seed=...)`` replaces ``env.rng``.  Consequently identical episode
    seeds in separate jobs start from different scenes.  This process-local
    patch reproduces LeRobot's method exactly while supplying a fixed
    constructor seed; it never modifies the installed package.
    """

    from lerobot.envs.robocasa import RoboCasaEnv as LeRobotRoboCasaEnv
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    def _ensure_env(instance: Any) -> None:
        if instance._env is not None:
            return
        instance._env = RoboCasaGymEnv(
            env_name=instance.task,
            camera_widths=instance.observation_width,
            camera_heights=instance.observation_height,
            split=instance.split if instance.split is not None else "all",
            obj_registries=instance.obj_registries,
            seed=int(seed) + int(instance.episode_index),
        )
        ep_meta = instance._env.env.get_ep_meta()
        instance.task_description = ep_meta.get("lang", instance.task)

    LeRobotRoboCasaEnv._ensure_env = _ensure_env


def _split_sentences(text: str) -> list[str]:
    """Match the sentence normalization used by the granular-data evaluator."""

    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    normalized = []
    for part in parts:
        part = re.sub(
            r"^(Then|Next|Finally|After that|Afterwards)[, ]+",
            "",
            part,
            flags=re.I,
        )
        normalized.append(part[0].upper() + part[1:] if part else part)
    return normalized


def _kitchen_env(envs: Any) -> Any:
    """Resolve dict -> SyncVectorEnv -> LeRobot wrapper -> RoboCasa kitchen."""

    vector = _vector_env(envs)
    if not hasattr(vector, "envs") or len(vector.envs) != 1:
        raise RuntimeError("progress evaluation requires one synchronous environment")
    wrapper = vector.envs[0]
    gym_wrapper = getattr(wrapper, "_env", None)
    kitchen = getattr(gym_wrapper, "env", None)
    if kitchen is None:
        raise RuntimeError("could not resolve underlying RoboCasa Kitchen environment")
    return kitchen


def _eef_position(env: Any) -> np.ndarray:
    site = env.robots[0].eef_site_id
    if isinstance(site, dict):
        site = site.get("right", next(iter(site.values())))
    return np.asarray(env.sim.data.site_xpos[int(site)], dtype=np.float64)


def _object_position(env: Any, name: str) -> np.ndarray:
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]], dtype=np.float64)


def _snapshot(task: str, env: Any, initial: dict[str, Any]) -> dict[str, Any]:
    """Read progress predicates without calling the task success checker.

    ``RinseSinkBasin._check_success`` mutates ``washed_loc``.  Calling it an
    extra time from the evaluator would therefore change benchmark state.  We
    only read the underlying predicates here and obtain terminal success from
    the environment's reward / info after ``step``.
    """

    from robocasa.utils import object_utils as ou

    if task == "KettleBoiling":
        obj = _object_position(env, "obj")
        eef = _eef_position(env)
        burner_distances: list[tuple[float, str]] = []
        burners_on = []
        knob_state = env.stove.get_knobs_state(env=env)
        for location, site in env.stove.burner_sites.items():
            if site is None:
                continue
            burner = np.asarray(env.sim.data.get_site_xpos(site.get("name")))
            burner_distances.append(
                (float(np.linalg.norm(burner[:2] - obj[:2])), str(location))
            )
            burners_on.append(
                bool(env.stove.is_burner_on(env=env, burner_loc=location))
                if location in knob_state
                else False
            )
        minimum, nearest_location = min(
            burner_distances, default=(float("inf"), "")
        )
        return {
            "eef_object_distance": float(np.linalg.norm(eef - obj)),
            "object_lift": float(obj[2] - initial["obj_z"]),
            "object_grasped": bool(ou.check_obj_grasped(env, "obj")),
            "object_stove_contact": bool(
                ou.check_obj_fixture_contact(env, "obj", env.stove)
            ),
            "minimum_burner_xy_distance": minimum,
            "nearest_burner_location": nearest_location,
            "kettle_near_burner": minimum < 0.15,
            "any_burner_on": any(burners_on),
            "gripper_object_far": bool(ou.gripper_obj_far(env)),
        }

    if task == "RinseSinkBasin":
        handle = env.sink.get_handle_state(env=env)
        washed = [bool(value) for value in env.washed_loc]
        return {
            "water_on": bool(handle["water_on"]),
            "washed_count": int(sum(washed)),
            "washed_left": washed[0],
            "washed_center": washed[1],
            "washed_right": washed[2],
            "spout_orientation": str(handle["spout_ori"]),
        }

    if task == "ScrubCuttingBoard":
        sponge = _object_position(env, "sponge")
        eef = _eef_position(env)
        positions = np.asarray(env.board_contact_positions)
        sweep = 0.0
        if positions.size:
            sweep = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        return {
            "eef_sponge_distance": float(np.linalg.norm(eef - sponge)),
            "sponge_lift": float(sponge[2] - initial["sponge_z"]),
            "sponge_grasped": bool(ou.check_obj_grasped(env, "sponge")),
            "board_contact_count": int(env.board_contact_timer),
            "board_sweep_range": sweep,
            "sponge_released_far": bool(
                ou.gripper_obj_far(env, "sponge", th=0.15)
            ),
        }

    if task == "StackBowlsCabinet":
        bowl1 = _object_position(env, "bowl1")
        bowl2 = _object_position(env, "bowl2")
        eef = _eef_position(env)
        return {
            "eef_bowl1_distance": float(np.linalg.norm(eef - bowl1)),
            "eef_bowl2_distance": float(np.linalg.norm(eef - bowl2)),
            "bowl1_lift": float(bowl1[2] - initial["bowl1_z"]),
            "bowl2_lift": float(bowl2[2] - initial["bowl2_z"]),
            "bowl1_grasped": bool(ou.check_obj_grasped(env, "bowl1")),
            "bowl2_grasped": bool(ou.check_obj_grasped(env, "bowl2")),
            "bowl1_in_cabinet": bool(ou.obj_inside_of(env, "bowl1", env.cabinet)),
            "bowl2_in_cabinet": bool(ou.obj_inside_of(env, "bowl2", env.cabinet)),
            "bowls_stacked": (
                bool(ou.check_obj_in_receptacle(env, "bowl2", "bowl1"))
                or bool(ou.check_obj_in_receptacle(env, "bowl1", "bowl2"))
            ),
        }
    raise ValueError(task)


def _initial_state(task: str, env: Any) -> dict[str, float]:
    if task == "KettleBoiling":
        return {"obj_z": float(_object_position(env, "obj")[2])}
    if task == "ScrubCuttingBoard":
        return {"sponge_z": float(_object_position(env, "sponge")[2])}
    if task == "StackBowlsCabinet":
        return {
            "bowl1_z": float(_object_position(env, "bowl1")[2]),
            "bowl2_z": float(_object_position(env, "bowl2")[2]),
        }
    return {}


def _available_kettle_burners(env: Any) -> list[str]:
    available = []
    for location, site in env.stove.burner_sites.items():
        if site is not None and env.stove.knob_joints.get(location) is not None:
            available.append(str(location).replace("_", "-"))
    return available


def _select_kettle_goal_burner(env: Any, requested: str) -> str:
    available = set(_available_kettle_burners(env))
    if requested != "auto":
        canonical = requested.replace("_", "-")
        if canonical not in available:
            raise RuntimeError(
                f"requested Kettle burner {canonical!r} is unavailable; "
                f"available={sorted(available)}"
            )
        kettle_burner_instruction(canonical)
        return canonical
    # Frozen ordering follows official training-label frequency. This uses
    # static fixture availability only, never rollout progress.
    for location in (
        "front-left", "front-right", "left", "rear-left", "rear-right",
        "center", "front-center",
    ):
        if location in available:
            return location
    raise RuntimeError(
        f"scene has no burner represented by official labels: {sorted(available)}"
    )


def _kettle_goal_sample(env: Any, location: str) -> dict[str, Any]:
    internal = location.replace("-", "_")
    site = env.stove.burner_sites.get(internal)
    if site is None:
        raise RuntimeError(f"selected burner site disappeared: {location}")
    burner = np.asarray(env.sim.data.get_site_xpos(site.get("name")))
    obj = _object_position(env, "obj")
    return {
        "selected_burner_location": location,
        "selected_burner_xy_distance": float(np.linalg.norm(burner[:2] - obj[:2])),
        "selected_burner_on": bool(
            env.stove.is_burner_on(env=env, burner_loc=internal)
        ),
    }


def _gripper_width(observation: dict[str, Any]) -> float:
    """Return Panda finger separation from deployable proprioception.

    RoboCasa's LeRobot wrapper exposes ``agent_pos`` as
    ``base(7) + ee(7) + gripper_qpos(2)``.  The two finger joints have
    opposite signs, so ``qpos[0] - qpos[1]`` is their separation.  This is a
    robot observation available at deployment, not simulator object state.
    """

    for key in ("agent_pos", "observation.state"):
        if key in observation:
            state = np.asarray(observation[key], dtype=np.float64)
            if state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            if state.ndim != 1 or state.shape[0] != 16:
                raise RuntimeError(
                    f"expected RoboCasa state shape (16,), got {state.shape}"
                )
            return float(state[-2] - state[-1])
    raise RuntimeError(
        f"RoboCasa observation has no deployable state key: {sorted(observation)}"
    )


def _update_extrema(extrema: dict[str, Any], sample: dict[str, Any]) -> None:
    for key, value in sample.items():
        if isinstance(value, bool):
            extrema[key] = bool(extrema.get(key, False) or value)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            if (
                key.startswith(("eef_", "minimum_")) and key.endswith("distance")
            ) or key == "selected_burner_xy_distance":
                extrema[key] = min(float(extrema.get(key, float("inf"))), float(value))
            else:
                extrema[key] = max(float(extrema.get(key, -float("inf"))), float(value))
        elif key not in extrema:
            extrema[key] = value


def _update_first_progress(
    first_progress: dict[str, int], sample: dict[str, Any], step: int
) -> None:
    """Record when each discrete predicate is first attained.

    Positive integer counters are included (for example ``washed_count`` and
    ``board_contact_count``); continuous values remain summarized by extrema.
    """

    for key, value in sample.items():
        attained = bool(value) if isinstance(value, bool) else (
            isinstance(value, (int, np.integer)) and int(value) > 0
        )
        if attained and key not in first_progress:
            first_progress[key] = int(step)


def _oracle_switch_ready(task: str, sample: dict[str, Any]) -> bool:
    """Return the benchmark-predicate boundary for temporal subtask 1.

    This is a diagnostic upper bound, not a deployable scheduler.  Only tasks
    whose two sentences are genuinely sequential are supported.
    """

    if task == "KettleBoiling":
        return bool(
            sample["object_stove_contact"]
            and sample["kettle_near_burner"]
            and sample["gripper_object_far"]
        )
    if task == "ScrubCuttingBoard":
        return bool(
            sample["board_contact_count"] >= 5
            and sample["board_sweep_range"] >= 0.1
        )
    raise ValueError(f"oracle switching is not defined for {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video_dir", type=Path)
    parser.add_argument("--video_stride", type=int, default=5)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument(
        "--instr_mode",
        default="env",
        help=(
            "env, selfpaced, proprio, oracle, rinse_oracle_regions, "
            "kettle_official_oracle, kettle_official_fixed, "
            "kettle_official_oracle_slowplace, kettle_official_fixed_slowplace, "
            "kettle_goal_oracle, kettle_goal_fixed, "
            "rinse_official_oracle, rinse_official_fixed, "
            "fixedk:<positive steps>, or cyclek:<positive steps>"
        ),
    )
    parser.add_argument("--switch_debounce", type=int, default=25)
    parser.add_argument("--proprio_width_threshold", type=float, default=0.009)
    parser.add_argument("--proprio_close_streak", type=int, default=25)
    parser.add_argument("--constructor_seed", type=int, default=20260821)
    parser.add_argument(
        "--kettle_goal_burner",
        default="auto",
        help="goal-consistent Kettle burner, or auto for the frozen availability rule",
    )
    parser.add_argument(
        "--split",
        choices=("target", "pretrain", "all"),
        default="target",
        help="RoboCasa scene split; paper benchmark evaluation uses target",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    renderer = os.environ.get("MUJOCO_GL")
    if renderer not in {"egl", "osmesa"}:
        raise RuntimeError(
            "MUJOCO_GL must be explicitly set to 'egl' or 'osmesa' for "
            "renderer-controlled RoboCasa evaluation"
        )

    clock_mode = args.instr_mode.startswith(("fixedk:", "cyclek:"))
    if args.instr_mode not in {
        "env", "selfpaced", "proprio", "oracle", "rinse_oracle_regions",
        "kettle_official_oracle", "kettle_official_fixed",
        "kettle_official_oracle_slowplace", "kettle_official_fixed_slowplace",
        "kettle_goal_oracle", "kettle_goal_fixed",
        "rinse_official_oracle", "rinse_official_fixed",
    } and not (
        clock_mode
        and args.instr_mode.split(":", 1)[1].isdigit()
        and int(args.instr_mode.split(":", 1)[1]) > 0
    ):
        raise ValueError(f"unsupported instruction mode: {args.instr_mode}")
    if args.instr_mode == "oracle" and args.task not in ORACLE_TASKS:
        raise ValueError(f"oracle switching is not defined for {args.task}")
    if args.instr_mode == "proprio" and args.task not in PROPRIO_TASKS:
        raise ValueError(f"proprio switching is not defined for {args.task}")
    if args.instr_mode == "rinse_oracle_regions" and args.task != "RinseSinkBasin":
        raise ValueError("rinse_oracle_regions is only defined for RinseSinkBasin")
    if args.instr_mode.startswith("rinse_official_") and args.task != "RinseSinkBasin":
        raise ValueError("official Rinse programs are only defined for RinseSinkBasin")
    if args.instr_mode.startswith("kettle_official_") and args.task != "KettleBoiling":
        raise ValueError("official Kettle programs are only defined for KettleBoiling")
    if args.instr_mode.startswith("kettle_goal_") and args.task != "KettleBoiling":
        raise ValueError("goal-consistent Kettle programs are only defined for KettleBoiling")
    if args.proprio_width_threshold <= 0:
        raise ValueError("proprio_width_threshold must be positive")
    if args.proprio_close_streak < 1:
        raise ValueError("proprio_close_streak must be positive")

    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    random.seed(args.seed_base)
    np.random.seed(args.seed_base)
    torch.manual_seed(args.seed_base)
    torch.cuda.manual_seed_all(args.seed_base)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _lock_robocasa_constructor(args.constructor_seed)

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    env_cfg = RoboCasaEnvCfg(
        task=args.task,
        episode_length=args.max_steps + 20,
        obj_registries=["lightwheel", "aigen"],
        split=None if args.split == "all" else args.split,
    )
    envs = make_env(env_cfg, n_envs=1)
    vector = _vector_env(envs)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    if args.video_stride <= 0:
        raise ValueError("--video_stride must be positive")
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    video_artifacts: list[dict[str, Any]] = []
    environment_provenance: dict[str, str] | None = None
    for episode_index in range(args.episodes):
        seed = args.seed_base + episode_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        observation, _ = vector.reset(seed=seed)
        video_frames = [video_frame(observation)] if args.video_dir is not None else []
        initial_observation_sha256 = _observation_sha256(observation)
        policy.reset()
        environment_instruction = str(vector.call("task_description")[0])
        kitchen = _kitchen_env(envs)
        kettle_goal_location: str | None = None
        if args.instr_mode == "env":
            sentences = [environment_instruction]
        elif args.instr_mode == "rinse_oracle_regions":
            sentences = list(RINSE_REGION_PROGRAM)
        elif args.instr_mode.startswith("rinse_official_"):
            sentences = list(RINSE_OFFICIAL_PROGRAM)
        elif args.instr_mode.startswith("kettle_official_"):
            sentences = list(KETTLE_OFFICIAL_PROGRAM)
        elif args.instr_mode.startswith("kettle_goal_"):
            kettle_goal_location = _select_kettle_goal_burner(
                kitchen, args.kettle_goal_burner
            )
            sentences = kettle_goal_program(kettle_goal_location)
        else:
            sentences = _split_sentences(environment_instruction)
        instruction_pointer = 0
        switch_events: list[dict[str, Any]] = []
        fixed_k = (
            int(args.instr_mode.split(":", 1)[1])
            if args.instr_mode.startswith("fixedk:")
            else None
        )
        cycle_k = (
            int(args.instr_mode.split(":", 1)[1])
            if args.instr_mode.startswith("cyclek:")
            else None
        )
        last_switch = -10**9
        if environment_provenance is None:
            class_file = Path(inspect.getfile(type(kitchen))).resolve()
            environment_provenance = {
                "class": f"{type(kitchen).__module__}.{type(kitchen).__qualname__}",
                "class_file": str(class_file),
                "class_file_sha256": _sha256(class_file),
            }
        initial = _initial_state(args.task, kitchen)
        extrema: dict[str, Any] = {}
        first_progress: dict[str, int] = {}
        transitions = 0
        previous_gripper = 0.0
        positive_gripper_streak = 0
        release_candidates: list[dict[str, Any]] = []
        success = False
        action_trace = hashlib.sha256()
        action_prefix_hashes: list[dict[str, Any]] = []
        first_action: list[float] | None = None
        step = 0
        while step < args.max_steps and not success:
            sample = _snapshot(args.task, kitchen, initial)
            if kettle_goal_location is not None:
                sample.update(_kettle_goal_sample(kitchen, kettle_goal_location))
            _update_extrema(extrema, sample)
            _update_first_progress(first_progress, sample, step)
            if args.instr_mode == "rinse_oracle_regions":
                advanced = rinse_program_advance(instruction_pointer, sample)
                if advanced != instruction_pointer:
                    previous_pointer = instruction_pointer
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "rinse_region_predicate",
                            "from": previous_pointer,
                            "to": instruction_pointer,
                        }
                    )
                    policy.reset()
            elif args.instr_mode == "rinse_official_oracle":
                advanced = rinse_official_program_advance(instruction_pointer, sample)
                if advanced != instruction_pointer:
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {"step": step, "reason": "official_rinse_water_on", "to": advanced}
                    )
                    policy.reset()
            elif args.instr_mode == "rinse_official_fixed":
                advanced = rinse_official_fixed_pointer(step)
                if advanced != instruction_pointer:
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {"step": step, "reason": "official_rinse_demo_median", "to": advanced}
                    )
                    policy.reset()
            elif args.instr_mode in {
                "kettle_official_oracle", "kettle_official_oracle_slowplace"
            }:
                advanced = kettle_program_advance(instruction_pointer, sample)
                if advanced != instruction_pointer:
                    previous_pointer = instruction_pointer
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "official_kettle_predicate",
                            "from": previous_pointer,
                            "to": instruction_pointer,
                        }
                    )
                    if instruction_pointer == 2:
                        sentences[2] = kettle_burner_instruction(
                            sample["nearest_burner_location"]
                        )
                    policy.reset()
            elif args.instr_mode in {
                "kettle_official_fixed", "kettle_official_fixed_slowplace"
            }:
                advanced = kettle_fixed_program_pointer(
                    step,
                    slow_place=args.instr_mode.endswith("_slowplace"),
                )
                if advanced != instruction_pointer:
                    previous_pointer = instruction_pointer
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "official_kettle_fixed_demo_median",
                            "from": previous_pointer,
                            "to": instruction_pointer,
                        }
                    )
                    if instruction_pointer == 2:
                        sentences[2] = kettle_burner_instruction(
                            sample["nearest_burner_location"]
                        )
                    policy.reset()
            elif args.instr_mode == "kettle_goal_oracle":
                advanced = kettle_goal_program_advance(instruction_pointer, sample)
                if advanced != instruction_pointer:
                    previous_pointer = instruction_pointer
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "goal_consistent_kettle_predicate",
                            "from": previous_pointer,
                            "to": instruction_pointer,
                        }
                    )
                    policy.reset()
            elif args.instr_mode == "kettle_goal_fixed":
                advanced = kettle_fixed_program_pointer(step)
                if advanced != instruction_pointer:
                    previous_pointer = instruction_pointer
                    instruction_pointer = advanced
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "goal_consistent_kettle_fixed_demo_median",
                            "from": previous_pointer,
                            "to": instruction_pointer,
                        }
                    )
                    policy.reset()
            elif (
                args.instr_mode == "oracle"
                and instruction_pointer < len(sentences) - 1
                and _oracle_switch_ready(args.task, sample)
            ):
                instruction_pointer += 1
                last_switch = step
                switch_events.append({"step": step, "reason": "oracle_predicate"})
                policy.reset()
            elif (
                fixed_k is not None
                and instruction_pointer < len(sentences) - 1
                and step > 0
                and step % fixed_k == 0
            ):
                instruction_pointer += 1
                last_switch = step
                switch_events.append({"step": step, "reason": f"fixedk:{fixed_k}"})
                policy.reset()
            elif cycle_k is not None and step > 0 and step % cycle_k == 0:
                instruction_pointer = (instruction_pointer + 1) % len(sentences)
                last_switch = step
                switch_events.append(
                    {
                        "step": step,
                        "reason": f"cyclek:{cycle_k}",
                        "instruction_pointer": instruction_pointer,
                    }
                )
                policy.reset()
            if args.instr_mode.endswith("_slowplace"):
                # Phase-conditioned decode control: the place phase is executed
                # at half the demonstration control rate; pick/burner stay 1x.
                # Pointer changes reset the queue above, so every generated
                # chunk consumes the ratio belonging to its current phase.
                policy.config.exec_rate_ratio = 2.0 if instruction_pointer == 1 else 1.0
            processed = preprocess_observation(observation)
            processed["task"] = [sentences[instruction_pointer]]
            processed = preprocessor(env_pre(processed))
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocessor(action)
            action_array = env_post({ACTION: action})[ACTION].cpu().numpy()[0]
            if action_array.shape != (12,) or not np.isfinite(action_array).all():
                raise RuntimeError(
                    f"expected a finite RoboCasa-12 action, got {action_array.shape}"
                )
            if first_action is None:
                first_action = action_array.astype(np.float64).tolist()
            action_trace.update(np.asarray(action_array, dtype="<f4").tobytes())
            if step < 5 or (step + 1) % 50 == 0:
                action_prefix_hashes.append(
                    {"steps": step + 1, "sha256": action_trace.copy().hexdigest()}
                )
            gripper = float(action_array[11])
            if np.sign(gripper) != np.sign(previous_gripper):
                transitions += 1
            gripper_width = _gripper_width(observation)
            is_release = previous_gripper > 0 >= np.sign(gripper)
            if is_release:
                accepted = bool(
                    args.instr_mode == "proprio"
                    and instruction_pointer < len(sentences) - 1
                    and positive_gripper_streak >= args.proprio_close_streak
                    and gripper_width >= args.proprio_width_threshold
                    and step - last_switch >= args.switch_debounce
                )
                release_candidates.append(
                    {
                        "step": step,
                        "gripper_width": gripper_width,
                        "positive_command_streak": positive_gripper_streak,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    instruction_pointer += 1
                    last_switch = step
                    switch_events.append(
                        {
                            "step": step,
                            "reason": "proprio_grasp_release",
                            "gripper_width": gripper_width,
                            "positive_command_streak": positive_gripper_streak,
                        }
                    )
                    policy.reset()
            if (
                args.instr_mode == "selfpaced"
                and instruction_pointer < len(sentences) - 1
                and is_release
                and step - last_switch >= args.switch_debounce
            ):
                instruction_pointer += 1
                last_switch = step
                switch_events.append({"step": step, "reason": "gripper_release"})
                policy.reset()
            previous_gripper = gripper
            positive_gripper_streak = (
                positive_gripper_streak + 1 if gripper > 0 else 0
            )
            observation, reward, terminated, truncated, info = vector.step(action_array[None, :])
            success = bool(
                np.asarray(reward).max() >= 1.0
                or np.any(np.asarray(info.get("is_success", False)))
            )
            step += 1
            if args.video_dir is not None and step % args.video_stride == 0:
                video_frames.append(video_frame(observation))
            done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
            if done and not success:
                break
            kitchen = _kitchen_env(envs)
        extrema["success"] = bool(success or extrema.get("success", False))
        video_record = None
        if args.video_dir is not None:
            import imageio.v2 as imageio

            video_path = args.video_dir / f"episode_{episode_index:03d}_seed_{seed}.mp4"
            if video_path.exists():
                raise FileExistsError(f"refusing video overwrite: {video_path}")
            imageio.mimwrite(
                video_path,
                video_frames,
                fps=max(1, 20 // args.video_stride),
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
            video_record = {
                "path": str(video_path.resolve()),
                "sha256": _sha256(video_path),
                "frames": len(video_frames),
                "stride": args.video_stride,
            }
            video_artifacts.append(video_record)
        episodes.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "initial_observation_sha256": initial_observation_sha256,
                "first_action": first_action,
                "action_trace_sha256": action_trace.hexdigest(),
                "action_prefix_hashes": action_prefix_hashes,
                "steps": step,
                "environment_instruction": environment_instruction,
                "instruction_sentences": sentences,
                "kettle_goal_location": kettle_goal_location,
                "instruction_switches": switch_events,
                "final_instruction_pointer": instruction_pointer,
                "gripper_sign_transitions": transitions,
                "gripper_release_candidates": release_candidates,
                "first_progress_steps": first_progress,
                "extrema": extrema,
                "video": video_record,
            }
        )
        print(
            f"episode={episode_index} success={extrema['success']} steps={step} "
            f"progress={extrema}",
            flush=True,
        )

    ckpt = Path(args.ckpt).resolve()
    result = {
        "schema": "robocasa_progress_eval_v16_goal_consistent_kettle",
        "task": args.task,
        "checkpoint": str(ckpt),
        "checkpoint_model_sha256": _sha256(ckpt / "model.safetensors"),
        "checkpoint_config_sha256": _sha256(ckpt / "config.json"),
        "episodes": episodes,
        "video_artifacts": video_artifacts,
        "successes": sum(bool(ep["extrema"]["success"]) for ep in episodes),
        "n_episodes": len(episodes),
        "seed_base": args.seed_base,
        "max_steps": args.max_steps,
        "instruction_mode": args.instr_mode,
        "kettle_goal_burner_request": args.kettle_goal_burner,
        "phase_exec_rate_ratios": (
            [1.0, 2.0, 1.0] if args.instr_mode.endswith("_slowplace") else [1.0]
        ),
        "switch_debounce": args.switch_debounce,
        "proprio_width_threshold": args.proprio_width_threshold,
        "proprio_close_streak": args.proprio_close_streak,
        "constructor_seed": args.constructor_seed,
        "environment_split": args.split,
        "environment": environment_provenance,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "renderer": renderer,
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "source_sha256": _sha256(__file__),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = stream.name
    os.link(temporary, destination)
    os.unlink(temporary)
    print(f"saved {destination}", flush=True)
    vector.close()


if __name__ == "__main__":
    main()
