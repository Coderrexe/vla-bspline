"""Stochastic, counterbalanced paired-block LIBERO duration experiment.

Exact trajectory replay is known to fail despite matched inputs and strict
deterministic-kernel settings. Each task/state/repeat block therefore runs a
trace-only predicted donor, a deterministically counterbalanced permutation of
the three efficacy arms, and a predicted closure drift sentinel. The donor's
in-process duration trace supplies the shuffled arm but is never used as the
efficacy predicted outcome.
"""

from __future__ import annotations

# Import v2 before Torch: v2 imports canonical deterministic setup first.
from libero_duration_ablation_v2 import (
    STRUCTURED_HASH_PROTOCOL,
    _active_duration_consumers,
    _single_action_sha256,
    _structured_sha256,
)
from libero_locked_eval_v2 import (
    _base_env,
    _determinism_metadata,
    _explicit_reset,
    _git_commit,
    _parse_task_ids,
    _seed_all,
    _sha256,
)

import argparse
import inspect
import json
import os
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
from duration_intervention_v2 import (
    load_complete_fixed_manifest,
)
from duration_intervention_v3 import (
    PROTOCOL as INTERVENTION_PROTOCOL,
    DurationInterventionV3,
)
from libero_hidden_state import (
    PROTOCOL as HIDDEN_STATE_PROTOCOL,
    integration_state_sha256,
    locate_robosuite_sim,
    model_xml_sha256,
)
from stochastic_duration_v3 import (
    BLOCK_SEED_PROTOCOL,
    COUNTERBALANCE_PROTOCOL,
    EFFICACY_PERMUTATIONS,
    WATERMARK,
    aggregate_predicted_pairs,
    block_seeds,
    canonical_json_sha256,
    observation_component_hashes,
    OBSERVATION_COMPONENT_PROTOCOL,
    predicted_pair_diagnostic,
    validate_embedded_sha256,
    materialize_task_order_manifest,
)


EVALUATION_PROTOCOL = "libero_duration_stochastic_paired_block_v3"
ARM_TO_MODE = {
    "predicted_donor": "predicted",
    "predicted_eval": "predicted",
    "fixed": "fixed",
    "shuffled": "shuffled",
    "predicted_closure": "predicted",
}


def _append_component_step(
    traces: dict[str, list[str]], components: dict[str, str], step: int
) -> None:
    """Append one schema-stable set of compact component hashes."""

    if step == 0:
        if traces:
            raise RuntimeError("component trace must be empty at step zero")
        traces.update({path: [digest] for path, digest in components.items()})
        return
    if set(traces) != set(components):
        raise RuntimeError(
            "observation component schema changed within episode: "
            f"step={step}, prior={sorted(traces)}, current={sorted(components)}"
        )
    if any(len(values) != step for values in traces.values()):
        raise RuntimeError("observation component trace lengths disagree")
    for path, digest in components.items():
        traces[path].append(digest)


def _run_episode(
    *,
    arm_label: str,
    env: Any,
    task_id: int,
    state_id: int,
    repeat_index: int,
    env_seed: int,
    policy_seed: int,
    task_description: str,
    max_steps: int,
    policy: Any,
    controller: DurationInterventionV3,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    action_key: str,
    preprocess_observation: Any,
    shuffle_source: list[int] | None = None,
    shuffle_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    duration_mode = ARM_TO_MODE[arm_label]
    observation, _, realized = _explicit_reset(
        env, state_id, env_seed, policy_seed
    )
    # `env` is the one-member SyncVectorEnv. Start traversal from its actual
    # LeRobot LiberoEnv so the private `_env` -> LIBERO -> robosuite chain is
    # reachable; the vector wrapper itself does not expose that chain.
    simulator, simulator_path = locate_robosuite_sim(_base_env(env))
    compiled_model_xml_sha256 = model_xml_sha256(simulator)
    policy.reset()
    controller.begin_episode(
        duration_mode,
        task_id,
        state_id,
        repeat_index,
        shuffle_source=shuffle_source,
    )
    success = False
    step = 0
    select_action_calls = 0
    action_trace = ExecutedActionTrace()
    action_step_hashes: list[str] = []
    action_prefix_hashes: list[str] = []
    raw_observation_hashes: list[str] = []
    processed_observation_hashes: list[str] = []
    raw_observation_component_hashes: dict[str, list[str]] = {}
    processed_observation_component_hashes: dict[str, list[str]] = {}
    simulator_state_hashes: list[str] = []
    replan_inputs: list[dict[str, Any]] = []

    while step < max_steps:
        simulator_hash = integration_state_sha256(simulator)
        raw_hash = _structured_sha256(observation)
        raw_components = observation_component_hashes(
            observation, _structured_sha256
        )
        processed = preprocess_observation(observation)
        processed["task"] = [task_description]
        processed = env_preprocessor(processed)
        processed = preprocessor(processed)
        processed_hash = _structured_sha256(processed)
        processed_components = observation_component_hashes(
            processed, _structured_sha256
        )
        raw_observation_hashes.append(raw_hash)
        processed_observation_hashes.append(processed_hash)
        _append_component_step(
            raw_observation_component_hashes, raw_components, step
        )
        _append_component_step(
            processed_observation_component_hashes, processed_components, step
        )
        simulator_state_hashes.append(simulator_hash)

        chunks_before = int(getattr(policy, "n_chunks_generated", -1))
        with torch.inference_mode():
            action = policy.select_action(processed)
        select_action_calls += 1
        chunks_after = int(getattr(policy, "n_chunks_generated", -1))
        if chunks_before < 0 or chunks_after < 0:
            raise RuntimeError("policy does not expose integer n_chunks_generated")
        if chunks_after - chunks_before not in (0, 1):
            raise RuntimeError(
                f"unexpected chunks generated in one select_action: "
                f"before={chunks_before}, after={chunks_after}"
            )
        if chunks_after == chunks_before + 1:
            replan_inputs.append(
                {
                    "chunk_index": len(replan_inputs),
                    "environment_step": step,
                    "raw_observation_sha256": raw_hash,
                    "processed_observation_sha256": processed_hash,
                    "mujoco_integration_state_sha256": simulator_hash,
                }
            )

        action = postprocessor(action)
        transition = env_postprocessor({action_key: action})
        action_numpy = transition[action_key].detach().cpu().numpy()
        action_step_hashes.append(_single_action_sha256(action_numpy))
        action_trace.update(action_numpy)
        action_prefix_hashes.append(action_trace.hexdigest())
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        step += 1
        success = success or bool(np.asarray(reward).max() >= 1.0)
        if "is_success" in info:
            success = success or bool(np.asarray(info["is_success"]).reshape(-1)[0])
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = success or bool(
                    np.asarray(final_info["is_success"]).reshape(-1)[0]
                )
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break

    duration_record = controller.end_episode()
    chunk_generations = int(getattr(policy, "n_chunks_generated", -1))
    if not (
        action_trace.count
        == step
        == select_action_calls
        == len(action_step_hashes)
        == len(action_prefix_hashes)
        == len(raw_observation_hashes)
        == len(processed_observation_hashes)
        == len(simulator_state_hashes)
    ):
        raise RuntimeError("step/call/action/observation trace lengths disagree")
    if not raw_observation_component_hashes or not (
        all(len(values) == step for values in raw_observation_component_hashes.values())
        and all(
            len(values) == step
            for values in processed_observation_component_hashes.values()
        )
    ):
        raise RuntimeError("component observation trace lengths disagree")
    if duration_record["n_duration_calls"] != chunk_generations:
        raise RuntimeError(
            f"duration calls {duration_record['n_duration_calls']} != "
            f"chunks {chunk_generations}"
        )
    if len(replan_inputs) != chunk_generations:
        raise RuntimeError(
            f"replan input count {len(replan_inputs)} != chunks {chunk_generations}"
        )
    if realized != state_id:
        raise RuntimeError(f"requested state {state_id}, realized {realized}")
    for replan, duration_chunk in zip(
        replan_inputs, duration_record["duration_chunks"], strict=True
    ):
        replan["predicted_duration"] = duration_chunk["predicted_duration"]
        replan["executed_duration"] = duration_chunk["executed_duration"]
        replan["canonical_replan_record_sha256"] = canonical_json_sha256(replan)

    source_length = len(shuffle_source) if shuffle_source is not None else None
    unused_source_values = (
        max(source_length - duration_record["n_duration_calls"], 0)
        if source_length is not None
        else None
    )
    equal_source_multiset = (
        sorted(duration_record["executed_durations"]) == sorted(shuffle_source)
        if shuffle_source is not None
        else None
    )
    record = {
        "arm_label": arm_label,
        "duration_mode": duration_mode,
        "task_id": task_id,
        "state_id": realized,
        "repeat_index": repeat_index,
        "env_seed_u32": env_seed,
        "policy_seed_u64": policy_seed,
        "success": success,
        "steps": step,
        "select_action_calls": select_action_calls,
        "chunk_generations": chunk_generations,
        "observation_hash_protocol": STRUCTURED_HASH_PROTOCOL,
        "observation_component_hash_protocol": OBSERVATION_COMPONENT_PROTOCOL,
        "initial_raw_observation_sha256": raw_observation_hashes[0],
        "initial_processed_observation_sha256": processed_observation_hashes[0],
        "hidden_state_protocol": HIDDEN_STATE_PROTOCOL,
        "simulator_path": simulator_path,
        "compiled_model_xml_sha256": compiled_model_xml_sha256,
        "initial_mujoco_integration_state_sha256": simulator_state_hashes[0],
        "raw_observation_step_sha256": raw_observation_hashes,
        "processed_observation_step_sha256": processed_observation_hashes,
        "raw_observation_component_step_sha256": (
            raw_observation_component_hashes
        ),
        "processed_observation_component_step_sha256": (
            processed_observation_component_hashes
        ),
        "mujoco_integration_state_step_sha256": simulator_state_hashes,
        "policy_replan_inputs": replan_inputs,
        "executed_action_count": action_trace.count,
        "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
        "executed_action_trace_sha256": action_trace.hexdigest(),
        "executed_action_step_sha256": action_step_hashes,
        "executed_action_prefix_sha256": action_prefix_hashes,
        "shuffle_source_manifest_sha256": shuffle_source_manifest_sha256,
        "unused_counterfactual_source_values": unused_source_values,
        "equal_counterfactual_source_multiset": equal_source_multiset,
        **duration_record,
    }
    print(
        f"arm={arm_label} task={task_id} state={state_id} repeat={repeat_index} "
        f"success={int(success)} steps={step} chunks={chunk_generations} "
        f"intervention={duration_record['intervention_call_fraction']:.3f} "
        f"trace={action_trace.hexdigest()[:12]}",
        flush=True,
    )
    return record


def _initial_pairing(donor: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_label": episode["arm_label"],
        "seed_agreement": (
            donor["env_seed_u32"] == episode["env_seed_u32"]
            and donor["policy_seed_u64"] == episode["policy_seed_u64"]
        ),
        "raw_initial_observation_agreement": (
            donor["initial_raw_observation_sha256"]
            == episode["initial_raw_observation_sha256"]
        ),
        "processed_initial_observation_agreement": (
            donor["initial_processed_observation_sha256"]
            == episode["initial_processed_observation_sha256"]
        ),
        "mujoco_integration_initial_state_agreement": (
            donor["initial_mujoco_integration_state_sha256"]
            == episode["initial_mujoco_integration_state_sha256"]
        ),
        "compiled_model_xml_agreement": (
            donor["compiled_model_xml_sha256"]
            == episode["compiled_model_xml_sha256"]
        ),
    }


def _arm_summary(blocks: list[dict[str, Any]], arm_label: str) -> dict[str, Any]:
    episodes = [block["episodes"][arm_label] for block in blocks]
    duration_calls = sum(episode["n_duration_calls"] for episode in episodes)
    intervened_calls = sum(
        episode["n_intervened_duration_calls"] for episode in episodes
    )
    absolute_sum = sum(
        chunk["absolute_duration_intervention"]
        for episode in episodes
        for chunk in episode["duration_chunks"]
    )
    equal_multiset_values = [
        episode["equal_counterfactual_source_multiset"]
        for episode in episodes
        if episode["equal_counterfactual_source_multiset"] is not None
    ]
    return {
        "arm_label": arm_label,
        "duration_mode": ARM_TO_MODE[arm_label],
        "role": (
            "efficacy"
            if arm_label in ("predicted_eval", "fixed", "shuffled")
            else "trace_only_drift_diagnostic"
        ),
        "n_episodes": len(episodes),
        "n_successes": sum(int(episode["success"]) for episode in episodes),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "n_steps": sum(episode["steps"] for episode in episodes),
        "n_duration_calls": duration_calls,
        "n_intervened_duration_calls": intervened_calls,
        "intervention_call_fraction": intervened_calls / duration_calls,
        "mean_absolute_duration_intervention": absolute_sum / duration_calls,
        "unused_counterfactual_source_values": sum(
            episode["unused_counterfactual_source_values"] or 0
            for episode in episodes
        ),
        "reused_counterfactual_values": sum(
            episode["reused_counterfactual_values"] for episode in episodes
        ),
        "n_equal_counterfactual_source_multisets": (
            sum(equal_multiset_values) if equal_multiset_values else None
        ),
        "equal_counterfactual_source_multiset_fraction": (
            sum(equal_multiset_values) / len(equal_multiset_values)
            if equal_multiset_values
            else None
        ),
    }


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fixed_manifest", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task_ids", help="optional comma-separated task ids")
    parser.add_argument("--state_start", type=int, default=0)
    parser.add_argument("--states_per_task", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--order_seed", type=int, default=20_260_821)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--shuffle_seed", type=int, default=73_921)
    parser.add_argument("--shuffle_strength_threshold", type=float, default=0.5)
    parser.add_argument(
        "--repeatability_equivalence_margin", type=float, default=0.10
    )
    parser.add_argument(
        "--require_exact_predicted_replay",
        action="store_true",
        help="publish diagnostics then fail unless all three predicted pairs replay exactly",
    )
    parser.add_argument("--block_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.state_start < 0 or args.states_per_task <= 0 or args.repeats <= 0:
        raise ValueError("state_start nonnegative; states_per_task/repeats positive")
    if not 0.0 <= args.shuffle_strength_threshold <= 1.0:
        raise ValueError("shuffle_strength_threshold must be in [0,1]")
    if not 0.0 < args.repeatability_equivalence_margin < 1.0:
        raise ValueError("repeatability_equivalence_margin must be in (0,1)")

    checkpoint = Path(args.ckpt).resolve()
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint}")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    block_directory = Path(args.block_dir)
    block_directory.mkdir(parents=True, exist_ok=True)
    fixed_manifest_path = Path(args.fixed_manifest).resolve()
    fixed_duration, fixed_manifest_payload = load_complete_fixed_manifest(
        fixed_manifest_path, config_path
    )

    _seed_all(args.seed_base)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    if not bool(getattr(policy_cfg, "predict_duration", False)):
        raise ValueError("checkpoint is not duration-enabled")
    active_consumers = _active_duration_consumers(policy_cfg)
    if active_consumers:
        raise ValueError(
            "duration ablation requires neutral optional decode controllers; "
            f"active={active_consumers}"
        )
    if not int(policy_cfg.min_seg) <= fixed_duration <= int(policy_cfg.horizon_max):
        raise ValueError("fixed manifest duration is outside checkpoint range")

    env_cfg = LiberoEnvCfg(
        task=args.suite,
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    suite_envs = make_env(env_cfg, n_envs=1)[args.suite]
    task_ids = _parse_task_ids(args.task_ids, list(suite_envs))
    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    env_source = Path(
        inspect.getfile(type(_base_env(suite_envs[task_ids[0]])))
    ).resolve()
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    controller = DurationInterventionV3(
        fixed_duration=fixed_duration, shuffle_seed=args.shuffle_seed
    )
    controller.attach(policy)
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
    episode_kwargs = {
        "policy": policy,
        "controller": controller,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
        "action_key": ACTION,
        "preprocess_observation": preprocess_observation,
    }

    model_hash = _sha256(model_path)
    config_hash = _sha256(config_path)
    config = json.loads(config_path.read_text())
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    evaluator_path = Path(__file__).resolve()
    controller_path = Path(inspect.getfile(DurationInterventionV3)).resolve()
    stochastic_helper_path = Path(inspect.getfile(block_seeds)).resolve()
    v2_hashing_path = Path(inspect.getfile(_structured_sha256)).resolve()
    hidden_state_helper_path = Path(
        inspect.getfile(integration_state_sha256)
    ).resolve()
    base_evaluator_path = Path(inspect.getfile(_explicit_reset)).resolve()
    action_helper_path = Path(atomic_write_json_new.__code__.co_filename).resolve()
    stats_path = None
    stats_hash = None
    stats_name = config.get("spline_stats_file_v2")
    if stats_name:
        candidate = policy_source.parent / stats_name
        if candidate.exists():
            stats_path = str(candidate)
            stats_hash = _sha256(candidate)
    source_hashes = {
        "evaluator_sha256": _sha256(evaluator_path),
        "controller_sha256": _sha256(controller_path),
        "stochastic_helper_sha256": _sha256(stochastic_helper_path),
        "v2_hashing_sha256": _sha256(v2_hashing_path),
        "hidden_state_helper_sha256": _sha256(hidden_state_helper_path),
        "base_evaluator_sha256": _sha256(base_evaluator_path),
        "action_helper_sha256": _sha256(action_helper_path),
        "policy_source_sha256": _sha256(policy_source),
        "env_source_sha256": _sha256(env_source),
        "stats_sha256": stats_hash,
        "lerobot_commit": _git_commit(policy_source.parents[4]),
    }
    resume_identity = {
        "protocol": "duration_v3_atomic_block_resume_identity",
        "checkpoint": str(checkpoint),
        "model_sha256": model_hash,
        "checkpoint_config_sha256": config_hash,
        "fixed_manifest_sha256": _sha256(fixed_manifest_path),
        "suite": args.suite,
        "task_ids": task_ids,
        "state_ids": state_ids,
        "repeats": args.repeats,
        "seed_base": args.seed_base,
        "order_seed": args.order_seed,
        "control_frequency_hz": args.control_freq,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_strength_threshold": args.shuffle_strength_threshold,
        "repeatability_equivalence_margin": (
            args.repeatability_equivalence_margin
        ),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "require_exact_predicted_replay": args.require_exact_predicted_replay,
        **source_hashes,
    }
    resume_identity_sha256 = canonical_json_sha256(resume_identity)
    process_runtime_manifest = {
        "protocol": "duration_v3_process_runtime_manifest",
        "determinism": _determinism_metadata(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    process_runtime_sha256 = canonical_json_sha256(process_runtime_manifest)
    execution_runtime_manifests = {
        process_runtime_sha256: process_runtime_manifest
    }
    task_order_manifests = {
        task_id: materialize_task_order_manifest(
            suite=args.suite,
            task_id=task_id,
            state_ids=state_ids,
            repeats=args.repeats,
            seed_base=args.seed_base,
            order_seed=args.order_seed,
        )
        for task_id in task_ids
    }

    all_blocks: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen_env_seeds: dict[int, tuple[int, int, int]] = {}
    seen_policy_seeds: dict[int, tuple[int, int, int]] = {}
    permutation_counts = {"/".join(order): 0 for order in EFFICACY_PERMUTATIONS}
    for task_id, manifest in task_order_manifests.items():
        for order_name, count in manifest["permutation_counts"].items():
            permutation_counts[order_name] += count
        values = list(manifest["permutation_counts"].values())
        if max(values) - min(values) > 1:
            raise RuntimeError(f"task {task_id} order quotas are not balanced")
        for block_spec in manifest["blocks"]:
            coordinates = (
                task_id,
                block_spec["state_id"],
                block_spec["repeat_index"],
            )
            env_seed = block_spec["env_seed_u32"]
            policy_seed = block_spec["policy_seed_u64"]
            if not 0 <= env_seed <= 0xFFFFFFFF:
                raise RuntimeError(f"invalid u32 environment seed {env_seed}")
            if not 0 <= policy_seed <= 0xFFFFFFFFFFFFFFFF:
                raise RuntimeError(f"invalid u64 policy seed {policy_seed}")
            if env_seed in seen_env_seeds:
                raise RuntimeError(
                    f"environment seed collision: {coordinates} and "
                    f"{seen_env_seeds[env_seed]}"
                )
            if policy_seed in seen_policy_seeds:
                raise RuntimeError(
                    f"policy seed collision: {coordinates} and "
                    f"{seen_policy_seeds[policy_seed]}"
                )
            seen_env_seeds[env_seed] = coordinates
            seen_policy_seeds[policy_seed] = coordinates

    started = time.time()
    n_blocks_resumed = 0
    n_blocks_executed = 0
    try:
        for task_ordinal, task_id in enumerate(task_ids):
            env = suite_envs[task_id]
            max_steps = int(env.call("_max_episode_steps")[0])
            task_description = str(next(iter(env.call("task_description"))))
            task_blocks: list[dict[str, Any]] = []
            task_manifest = task_order_manifests[task_id]
            task_manifest_hash = task_manifest["manifest_sha256"]
            for block_spec in task_manifest["blocks"]:
                state_id = int(block_spec["state_id"])
                repeat_index = int(block_spec["repeat_index"])
                env_seed = int(block_spec["env_seed_u32"])
                policy_seed = int(block_spec["policy_seed_u64"])
                efficacy_permutation = tuple(block_spec["efficacy_permutation"])
                arm_order = list(block_spec["arm_order"])
                block_path = block_directory / (
                    f"task{task_id:02d}_state{state_id:04d}_"
                    f"repeat{repeat_index:03d}.json"
                )
                if block_path.exists():
                    block_artifact = json.loads(block_path.read_text())
                    expected = {
                        "schema_version": 1,
                        "protocol": "duration_v3_atomic_complete_block",
                        "resume_identity_sha256": resume_identity_sha256,
                        "task_order_manifest_sha256": task_manifest_hash,
                        "task_id": task_id,
                        "state_id": state_id,
                        "repeat_index": repeat_index,
                    }
                    mismatches = {
                        key: {"expected": value, "found": block_artifact.get(key)}
                        for key, value in expected.items()
                        if block_artifact.get(key) != value
                    }
                    if mismatches:
                        raise ValueError(
                            f"refusing incompatible block resume {block_path}: "
                            f"{mismatches}"
                        )
                    block = block_artifact["block"]
                    if block_artifact.get("block_sha256") != canonical_json_sha256(
                        block
                    ):
                        raise ValueError(f"resumed block hash mismatch: {block_path}")
                    runtime_manifest = block_artifact.get(
                        "execution_runtime_manifest"
                    )
                    runtime_sha256 = block_artifact.get(
                        "execution_runtime_sha256"
                    )
                    if not isinstance(runtime_manifest, dict) or (
                        runtime_sha256 != canonical_json_sha256(runtime_manifest)
                    ):
                        raise ValueError(
                            f"resumed block runtime provenance mismatch: {block_path}"
                        )
                    if block.get("execution_runtime_sha256") != runtime_sha256:
                        raise ValueError(
                            f"resumed block runtime binding mismatch: {block_path}"
                        )
                    execution_runtime_manifests[runtime_sha256] = runtime_manifest
                    if block.get("arm_order") != arm_order or set(
                        block.get("episodes", {})
                    ) != set(ARM_TO_MODE):
                        raise ValueError(f"resumed block is incomplete: {block_path}")
                    shuffle_manifest = block.get("shuffle_source_manifest")
                    if not isinstance(shuffle_manifest, dict):
                        raise ValueError(
                            f"resumed block lacks shuffle manifest: {block_path}"
                        )
                    validate_embedded_sha256(shuffle_manifest)
                    shuffled_episode = block["episodes"]["shuffled"]
                    if shuffled_episode.get(
                        "shuffle_source_manifest_sha256"
                    ) != shuffle_manifest["manifest_sha256"]:
                        raise ValueError(
                            f"resumed shuffle source binding mismatch: {block_path}"
                        )
                    n_blocks_resumed += 1
                    task_blocks.append(block)
                    all_blocks.append(block)
                    continue

                # The following five rollouts are the indivisible resume unit.
                # Nothing is published until the closure and all diagnostics
                # are complete.
                donor = _run_episode(
                    arm_label="predicted_donor",
                    env=env,
                    task_id=task_id,
                    state_id=state_id,
                    repeat_index=repeat_index,
                    env_seed=env_seed,
                    policy_seed=policy_seed,
                    task_description=task_description,
                    max_steps=max_steps,
                    **episode_kwargs,
                )
                source_payload = {
                    "protocol": "in_process_predicted_donor_shuffle_source_v3",
                    "resume_identity_sha256": resume_identity_sha256,
                    "checkpoint": str(checkpoint),
                    "model_sha256": model_hash,
                    "checkpoint_config_sha256": config_hash,
                    "suite": args.suite,
                    "configured_task_ids": task_ids,
                    "configured_state_ids": state_ids,
                    "configured_repeats": args.repeats,
                    "task_order_manifest_sha256": task_manifest_hash,
                    "task_id": task_id,
                    "state_id": state_id,
                    "repeat_index": repeat_index,
                    "env_seed_u32": env_seed,
                    "policy_seed_u64": policy_seed,
                    "control_frequency_hz": args.control_freq,
                    "shuffle_seed": args.shuffle_seed,
                    "intervention_protocol": INTERVENTION_PROTOCOL,
                    "predicted_durations": donor["predicted_durations"],
                    "donor_action_trace_sha256": donor[
                        "executed_action_trace_sha256"
                    ],
                }
                source_payload["manifest_sha256"] = canonical_json_sha256(
                    source_payload
                )
                validate_embedded_sha256(source_payload)
                episodes: dict[str, dict[str, Any]] = {
                    "predicted_donor": donor
                }
                for arm_label in efficacy_permutation:
                    episodes[arm_label] = _run_episode(
                        arm_label=arm_label,
                        env=env,
                        task_id=task_id,
                        state_id=state_id,
                        repeat_index=repeat_index,
                        env_seed=env_seed,
                        policy_seed=policy_seed,
                        task_description=task_description,
                        max_steps=max_steps,
                        shuffle_source=(
                            donor["predicted_durations"]
                            if arm_label == "shuffled"
                            else None
                        ),
                        shuffle_source_manifest_sha256=(
                            source_payload["manifest_sha256"]
                            if arm_label == "shuffled"
                            else None
                        ),
                        **episode_kwargs,
                    )
                closure = _run_episode(
                    arm_label="predicted_closure",
                    env=env,
                    task_id=task_id,
                    state_id=state_id,
                    repeat_index=repeat_index,
                    env_seed=env_seed,
                    policy_seed=policy_seed,
                    task_description=task_description,
                    max_steps=max_steps,
                    **episode_kwargs,
                )
                episodes["predicted_closure"] = closure
                shuffled_episode = episodes["shuffled"]
                if (
                    shuffled_episode["shuffle_source_manifest_sha256"]
                    != source_payload["manifest_sha256"]
                    or shuffled_episode["counterfactual_source_length"]
                    != len(donor["predicted_durations"])
                ):
                    raise RuntimeError("shuffled episode/source binding failed")
                predicted_repeatability = {
                    "donor_vs_eval": predicted_pair_diagnostic(
                        donor, episodes["predicted_eval"]
                    ),
                    "donor_vs_closure": predicted_pair_diagnostic(donor, closure),
                    "eval_vs_closure": predicted_pair_diagnostic(
                        episodes["predicted_eval"], closure
                    ),
                }
                block = {
                    "execution_runtime_sha256": process_runtime_sha256,
                    "task_block_ordinal": block_spec["task_block_ordinal"],
                    "task_order_manifest_sha256": task_manifest_hash,
                    "task_id": task_id,
                    "state_id": state_id,
                    "repeat_index": repeat_index,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
                    "env_seed_u32": env_seed,
                    "policy_seed_u64": policy_seed,
                    "arm_order": arm_order,
                    "efficacy_permutation": list(efficacy_permutation),
                    "shuffle_source_manifest": source_payload,
                    "initial_pairing_by_arm": {
                        label: _initial_pairing(donor, episodes[label])
                        for label in arm_order[1:]
                    },
                    "predicted_repeatability": predicted_repeatability,
                    "episodes": episodes,
                }
                block_artifact = {
                    "schema_version": 1,
                    "protocol": "duration_v3_atomic_complete_block",
                    "watermark": WATERMARK,
                    "resume_identity_sha256": resume_identity_sha256,
                    "task_order_manifest_sha256": task_manifest_hash,
                    "task_id": task_id,
                    "state_id": state_id,
                    "repeat_index": repeat_index,
                    "execution_runtime_sha256": process_runtime_sha256,
                    "execution_runtime_manifest": process_runtime_manifest,
                    "block_sha256": canonical_json_sha256(block),
                    "block": block,
                }
                atomic_write_json_new(block_path, block_artifact)
                n_blocks_executed += 1
                task_blocks.append(block)
                all_blocks.append(block)
            tasks.append(
                {
                    "task_id": task_id,
                    "task_ordinal": task_ordinal,
                    "task_description": task_description,
                    "task_order_manifest": task_manifest,
                    "n_blocks": len(task_blocks),
                    "blocks": task_blocks,
                }
            )
    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except (RuntimeError, OSError) as error:
                print(f"WARNING: environment close failed: {error}", file=sys.stderr)

    expected_blocks = len(task_ids) * len(state_ids) * args.repeats
    if len(all_blocks) != expected_blocks:
        raise RuntimeError(f"blocks={len(all_blocks)} != expected={expected_blocks}")
    arm_labels = (
        "predicted_donor",
        "predicted_eval",
        "fixed",
        "shuffled",
        "predicted_closure",
    )
    arm_summaries = {
        label: _arm_summary(all_blocks, label) for label in arm_labels
    }
    predicted_pair_names = (
        "donor_vs_eval",
        "donor_vs_closure",
        "eval_vs_closure",
    )
    predicted_repeatability = {
        pair_name: aggregate_predicted_pairs(
            [
                block["predicted_repeatability"][pair_name]
                for block in all_blocks
            ]
        )
        for pair_name in predicted_pair_names
    }
    exact_predicted_replay_all_pairs = all(
        summary["n_exact_action_trace_agreements"] == len(all_blocks)
        and summary["n_exact_predicted_duration_trace_agreements"]
        == len(all_blocks)
        and summary["first_raw_observation_divergence_step"]["n"] == 0
        and summary["first_processed_observation_divergence_step"]["n"] == 0
        and summary["first_mujoco_integration_state_divergence_step"]["n"] == 0
        for summary in predicted_repeatability.values()
    )
    predicted_repeatability_by_task = {
        str(task["task_id"]): {
            pair_name: aggregate_predicted_pairs(
                [
                    block["predicted_repeatability"][pair_name]
                    for block in task["blocks"]
                ]
            )
            for pair_name in predicted_pair_names
        }
        for task in tasks
    }
    initial_pairings = [
        pairing
        for block in all_blocks
        for pairing in block["initial_pairing_by_arm"].values()
    ]
    pairing_summary = {
        "n_arm_episode_pairs": len(initial_pairings),
        "n_seed_agreements": sum(pairing["seed_agreement"] for pairing in initial_pairings),
        "n_raw_initial_observation_agreements": sum(
            pairing["raw_initial_observation_agreement"]
            for pairing in initial_pairings
        ),
        "n_processed_initial_observation_agreements": sum(
            pairing["processed_initial_observation_agreement"]
            for pairing in initial_pairings
        ),
        "n_mujoco_integration_initial_state_agreements": sum(
            pairing["mujoco_integration_initial_state_agreement"]
            for pairing in initial_pairings
        ),
        "n_compiled_model_xml_agreements": sum(
            pairing["compiled_model_xml_agreement"]
            for pairing in initial_pairings
        ),
    }

    predicted_rates = {
        label: arm_summaries[label]["success_rate"]
        for label in (
            "predicted_donor",
            "predicted_eval",
            "predicted_closure",
        )
    }
    predicted_rate_spread = max(predicted_rates.values()) - min(
        predicted_rates.values()
    )
    taskwise_predicted_rates = {
        str(task["task_id"]): {
            label: _arm_summary(task["blocks"], label)["success_rate"]
            for label in (
                "predicted_donor",
                "predicted_eval",
                "predicted_closure",
            )
        }
        for task in tasks
    }
    taskwise_predicted_spreads = {
        task_id: max(rates.values()) - min(rates.values())
        for task_id, rates in taskwise_predicted_rates.items()
    }
    repeatability_gate = {
        "protocol": "predicted_position_marginal_equivalence_target_v3",
        "equivalence_margin": args.repeatability_equivalence_margin,
        "predicted_success_rates": predicted_rates,
        "observed_max_minus_min": predicted_rate_spread,
        "pilot_observed_within_margin": (
            predicted_rate_spread <= args.repeatability_equivalence_margin
        ),
        "taskwise_predicted_success_rates": taskwise_predicted_rates,
        "taskwise_observed_max_minus_min": taskwise_predicted_spreads,
        "pilot_all_tasks_within_margin": all(
            spread <= args.repeatability_equivalence_margin
            for spread in taskwise_predicted_spreads.values()
        ),
        "confirmatory_requirement": (
            "hierarchical paired confidence intervals for all predicted-position "
            "contrasts must lie inside +/-0.05 at 95% confidence; the configured "
            "pilot equivalence_margin is descriptive only"
        ),
        "confirmatory_95pct_ci_margin": 0.05,
    }
    order_gate = {
        "protocol": "per_task_six_permutation_quota_balance_v3",
        "passed": all(
            max(task["task_order_manifest"]["permutation_counts"].values())
            - min(task["task_order_manifest"]["permutation_counts"].values())
            <= 1
            for task in tasks
        ),
        "per_task_counts": {
            str(task["task_id"]): task["task_order_manifest"]["permutation_counts"]
            for task in tasks
        },
    }
    shuffled_summary = arm_summaries["shuffled"]
    shuffled_taskwise = {
        str(task["task_id"]): _arm_summary(task["blocks"], "shuffled")
        for task in tasks
    }
    shuffle_passed_overall = (
        shuffled_summary["intervention_call_fraction"]
        >= args.shuffle_strength_threshold
    )
    shuffle_all_tasks_passed = all(
        summary["intervention_call_fraction"]
        >= args.shuffle_strength_threshold
        for summary in shuffled_taskwise.values()
    )
    shuffle_strength_gate = {
        "protocol": "shuffle_effective_changed_call_fraction_v3",
        "target_minimum": args.shuffle_strength_threshold,
        "observed_changed_call_fraction": shuffled_summary[
            "intervention_call_fraction"
        ],
        "passed": shuffle_passed_overall and shuffle_all_tasks_passed,
        "passed_overall": shuffle_passed_overall,
        "all_tasks_passed": shuffle_all_tasks_passed,
        "unused_counterfactual_source_values": shuffled_summary[
            "unused_counterfactual_source_values"
        ],
        "reused_counterfactual_values": shuffled_summary[
            "reused_counterfactual_values"
        ],
        "equal_source_multiset_fraction": shuffled_summary[
            "equal_counterfactual_source_multiset_fraction"
        ],
        "taskwise": shuffled_taskwise,
    }
    initial_pairing_complete = all(
        pairing_summary[name] == pairing_summary["n_arm_episode_pairs"]
        for name in (
            "n_seed_agreements",
            "n_raw_initial_observation_agreements",
            "n_processed_initial_observation_agreements",
            "n_mujoco_integration_initial_state_agreements",
            "n_compiled_model_xml_agreements",
        )
    )
    scientific_pilot_eligible = (
        repeatability_gate["pilot_observed_within_margin"]
        and repeatability_gate["pilot_all_tasks_within_margin"]
        and order_gate["passed"]
        and shuffle_strength_gate["passed"]
        and initial_pairing_complete
    )

    result = {
        "schema_version": 3,
        "protocol": EVALUATION_PROTOCOL,
        "watermark": WATERMARK,
        "stochastic_result": True,
        "bitwise_replay_assumed": False,
        "exact_predicted_replay_required": args.require_exact_predicted_replay,
        "exact_predicted_replay_all_pairs": exact_predicted_replay_all_pairs,
        "known_replay_failure": (
            "same-process t2/state0 smoke 2319737: identical initial hashes and "
            "actions through step 119, first action divergence step 120"
        ),
        "scientific_arms": ["predicted_eval", "fixed", "shuffled"],
        "trace_only_arms": ["predicted_donor", "predicted_closure"],
        "primary_predicted_arm": "predicted_eval",
        "execution_scope": (
            "each indivisible five-rollout block uses one process and one loaded "
            "policy/environment set; atomic resume may combine completed blocks "
            "from multiple process allocations"
        ),
        "block_execution": (
            "predicted donor; one of all six permutations of predicted_eval, "
            "fixed, shuffled; predicted closure"
        ),
        "counterbalance_protocol": COUNTERBALANCE_PROTOCOL,
        "order_seed": args.order_seed,
        "efficacy_permutations": [list(order) for order in EFFICACY_PERMUTATIONS],
        "efficacy_permutation_counts": permutation_counts,
        "block_seed_protocol": BLOCK_SEED_PROTOCOL,
        "duration_prediction_source": (
            "bound original production policy._predicted_T(tokens) called once "
            "inside the common wrapper in every arm"
        ),
        "intervention_protocol": INTERVENTION_PROTOCOL,
        "inference_contract": (
            "No episode-level iid inference. Primary efficacy uses predicted_eval "
            "only and must model task/state/repeat hierarchy and arm-order period; "
            "all donor/eval/closure pairs quantify drift only. A paper claim "
            "requires independent checkpoint seeds and hierarchical state/repeat "
            "inference."
        ),
        "pilot_default": {"states_per_task": 12, "repeats": 1},
        "fixed_duration": fixed_duration,
        "fixed_manifest": str(fixed_manifest_path),
        "fixed_manifest_sha256": _sha256(fixed_manifest_path),
        "fixed_manifest_protocol": fixed_manifest_payload.get("protocol"),
        "fixed_manifest_complete_training_scan": fixed_manifest_payload[
            "complete_training_scan"
        ],
        "shuffle_seed": args.shuffle_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": config_hash,
        "model_sha256": model_hash,
        "policy_type": config["type"],
        "n_action_steps": config.get("n_action_steps"),
        "suite": args.suite,
        "task_ids": task_ids,
        "state_ids": state_ids,
        "repeats": args.repeats,
        "seed_base": args.seed_base,
        "control_frequency_hz": args.control_freq,
        "n_blocks": len(all_blocks),
        "n_blocks_executed_this_process": n_blocks_executed,
        "n_blocks_resumed_atomically": n_blocks_resumed,
        "block_directory": str(block_directory.resolve()),
        "resume_identity": resume_identity,
        "resume_identity_sha256": resume_identity_sha256,
        "n_episodes": len(all_blocks) * len(arm_labels),
        "arm_summaries": arm_summaries,
        "predicted_repeatability_all_pairs": predicted_repeatability,
        "predicted_repeatability_by_task": predicted_repeatability_by_task,
        "repeatability_equivalence_gate": repeatability_gate,
        "order_balance_gate": order_gate,
        "shuffle_strength_gate": shuffle_strength_gate,
        "initial_pairing_complete": initial_pairing_complete,
        "scientific_pilot_eligible": scientific_pilot_eligible,
        "initial_pairing_summary": pairing_summary,
        "current_execution_runtime_sha256": process_runtime_sha256,
        "execution_runtime_manifests": execution_runtime_manifests,
        "determinism": process_runtime_manifest["determinism"],
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "policy_source": str(policy_source),
        "policy_source_sha256": source_hashes["policy_source_sha256"],
        "env_source": str(env_source),
        "env_source_sha256": source_hashes["env_source_sha256"],
        "lerobot_commit": source_hashes["lerobot_commit"],
        "stats_path": stats_path,
        "stats_sha256": stats_hash,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": source_hashes["evaluator_sha256"],
        "controller_path": str(controller_path),
        "controller_sha256": source_hashes["controller_sha256"],
        "stochastic_helper_path": str(stochastic_helper_path),
        "stochastic_helper_sha256": source_hashes["stochastic_helper_sha256"],
        "v2_hashing_path": str(v2_hashing_path),
        "v2_hashing_sha256": source_hashes["v2_hashing_sha256"],
        "hidden_state_helper_path": str(hidden_state_helper_path),
        "hidden_state_helper_sha256": source_hashes[
            "hidden_state_helper_sha256"
        ],
        "base_evaluator_path": str(base_evaluator_path),
        "base_evaluator_sha256": source_hashes["base_evaluator_sha256"],
        "action_helper_path": str(action_helper_path),
        "action_helper_sha256": source_hashes["action_helper_sha256"],
        "tasks": tasks,
    }
    atomic_write_json_new(output, result)
    print(
        f"saved stochastic paired-block result {output}: "
        f"blocks={len(all_blocks)} episodes={result['n_episodes']} "
        f"pilot_eligible={scientific_pilot_eligible} "
        f"shuffle_strength={shuffle_strength_gate['observed_changed_call_fraction']:.3f}",
        flush=True,
    )
    if args.require_exact_predicted_replay and not exact_predicted_replay_all_pairs:
        raise RuntimeError(
            "exact predicted replay was required but at least one predicted pair "
            "diverged; diagnostic result was published"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
