"""Single-process deterministic LIBERO duration-mechanism experiment.

One loaded policy and one set of environments execute, in order: predicted A,
predicted B, fixed, and shuffled. Predicted A/B must be exactly replay-identical
before either intervention runs. The output retains initial observation hashes,
per-step action hashes, cumulative action-prefix hashes, and per-chunk duration
traces, making any failed determinism gate localizable.
"""

from __future__ import annotations

# Canonical v2 sets CUDA environment variables before importing Torch and then
# enables the same fail-closed deterministic-kernel configuration used here.
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

import argparse
import hashlib
import inspect
import json
import os
import struct
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
    PROTOCOL as INTERVENTION_PROTOCOL,
    DurationInterventionV2,
    canonical_json_sha256,
    load_complete_fixed_manifest,
)


EVALUATION_PROTOCOL = "libero_duration_intervention_deterministic_v2"
STRUCTURED_HASH_PROTOCOL = "typed_recursive_value_sha256_v1"


def _hash_frame(digest: Any, tag: str, payload: bytes = b"") -> None:
    tag_bytes = tag.encode("ascii")
    digest.update(struct.pack("<Q", len(tag_bytes)))
    digest.update(tag_bytes)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _update_structured_hash(digest: Any, value: Any) -> None:
    """Hash nested observations with dtype/shape/type framing."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "source_device": str(value.device),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        # Viewing raw storage as uint8 also supports bfloat16, which NumPy's
        # dtype bridge does not represent on every installed version.
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _hash_frame(digest, "torch_header", header)
        _hash_frame(digest, "torch_payload", payload)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("cannot hash object-dtype observation arrays")
        array = np.ascontiguousarray(value)
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        _hash_frame(digest, "array_header", header)
        _hash_frame(digest, "array_payload", array.tobytes(order="C"))
    elif isinstance(value, dict):
        _hash_frame(digest, "dict_length", struct.pack("<Q", len(value)))
        if not all(isinstance(key, str) for key in value):
            raise TypeError("observation dictionaries must have string keys")
        for key in sorted(value):
            _hash_frame(digest, "dict_key", key.encode("utf-8"))
            _update_structured_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _hash_frame(
            digest,
            "tuple_length" if isinstance(value, tuple) else "list_length",
            struct.pack("<Q", len(value)),
        )
        for item in value:
            _update_structured_hash(digest, item)
    elif value is None:
        _hash_frame(digest, "none")
    elif isinstance(value, (str, bool, int, float, np.generic)):
        scalar = value.item() if isinstance(value, np.generic) else value
        payload = json.dumps(
            scalar, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _hash_frame(digest, f"scalar_{type(scalar).__name__}", payload)
    else:
        raise TypeError(f"unsupported structured-hash value {type(value)!r}")


def _structured_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(STRUCTURED_HASH_PROTOCOL.encode("ascii") + b"\0")
    _update_structured_hash(digest, value)
    return digest.hexdigest()


def _single_action_sha256(action: Any) -> str:
    trace = ExecutedActionTrace()
    trace.update(action)
    return trace.hexdigest()


def _active_duration_consumers(config: Any) -> list[str]:
    checks = {
        "speedup_alpha": getattr(config, "speedup_alpha", 1.0) != 1.0,
        "slowdown_alpha": getattr(config, "slowdown_alpha", 1.0) != 1.0,
        "ease_out": getattr(config, "ease_out", None) is not None,
        "profile_alpha": getattr(config, "profile_alpha", None) is not None,
        "profile_slow_alpha": getattr(config, "profile_slow_alpha", None)
        is not None,
        "replan_frac": getattr(config, "replan_frac", None) is not None,
        "replan_margin": getattr(config, "replan_margin", None) is not None,
        "chain_velocity": bool(getattr(config, "chain_velocity", False)),
        "feasibility_stretch": bool(
            getattr(config, "feasibility_stretch", False)
        ),
        "exec_rate_ratio": getattr(config, "exec_rate_ratio", 1.0) != 1.0,
    }
    return sorted(name for name, active in checks.items() if active)


def _run_arm(
    *,
    arm_label: str,
    duration_mode: str,
    suite_envs: dict[int, Any],
    task_ids: list[int],
    state_ids: list[int],
    seed_base: int,
    policy: Any,
    controller: DurationInterventionV2,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    action_key: str,
    preprocess_observation: Any,
    shuffle_sources: dict[tuple[int, int], list[int]] | None = None,
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    arm_trace = ExecutedActionTrace()
    started = time.time()

    for task_id in task_ids:
        env = suite_envs[task_id]
        max_steps = int(env.call("_max_episode_steps")[0])
        task_description = str(next(iter(env.call("task_description"))))
        episodes: list[dict[str, Any]] = []
        for state_id in state_ids:
            env_seed = seed_base + task_id * 1_000 + state_id
            policy_seed = env_seed
            observation, _, realized = _explicit_reset(
                env, state_id, env_seed, policy_seed
            )
            raw_observation_sha256 = _structured_sha256(observation)
            policy.reset()
            source = (
                shuffle_sources[(task_id, state_id)]
                if shuffle_sources is not None
                else None
            )
            controller.begin_episode(
                duration_mode,
                task_id,
                state_id,
                shuffle_source=source,
            )
            success = False
            step = 0
            select_action_calls = 0
            episode_trace = ExecutedActionTrace()
            step_hashes: list[str] = []
            prefix_hashes: list[str] = []
            processed_observation_sha256: str | None = None

            while step < max_steps:
                processed = preprocess_observation(observation)
                processed["task"] = [task_description]
                processed = env_preprocessor(processed)
                processed = preprocessor(processed)
                if processed_observation_sha256 is None:
                    processed_observation_sha256 = _structured_sha256(processed)
                with torch.inference_mode():
                    action = policy.select_action(processed)
                select_action_calls += 1
                action = postprocessor(action)
                transition = env_postprocessor({action_key: action})
                action_numpy = transition[action_key].detach().cpu().numpy()
                step_hashes.append(_single_action_sha256(action_numpy))
                episode_trace.update(action_numpy)
                arm_trace.update(action_numpy)
                prefix_hashes.append(episode_trace.hexdigest())
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

            duration_record = controller.end_episode()
            chunk_generations = int(getattr(policy, "n_chunks_generated", -1))
            if episode_trace.count != step or select_action_calls != step:
                raise RuntimeError(
                    "trace/call mismatch: "
                    f"trace={episode_trace.count}, calls={select_action_calls}, "
                    f"steps={step}"
                )
            if duration_record["n_duration_calls"] != chunk_generations:
                raise RuntimeError(
                    "duration/chunk mismatch: "
                    f"duration_calls={duration_record['n_duration_calls']}, "
                    f"chunks={chunk_generations}"
                )
            if len(step_hashes) != step or len(prefix_hashes) != step:
                raise RuntimeError("per-step action hash count does not match steps")
            if processed_observation_sha256 is None:
                raise RuntimeError("episode executed no processed observation")

            episode = {
                "task_id": task_id,
                "state_id": realized,
                "env_seed": env_seed,
                "policy_seed": policy_seed,
                "success": success,
                "steps": step,
                "select_action_calls": select_action_calls,
                "chunk_generations": chunk_generations,
                "initial_raw_observation_hash_protocol": STRUCTURED_HASH_PROTOCOL,
                "initial_raw_observation_sha256": raw_observation_sha256,
                "initial_processed_observation_hash_protocol": STRUCTURED_HASH_PROTOCOL,
                "initial_processed_observation_sha256": processed_observation_sha256,
                "executed_action_count": episode_trace.count,
                "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
                "executed_action_trace_sha256": episode_trace.hexdigest(),
                "executed_action_step_sha256": step_hashes,
                "executed_action_prefix_sha256": prefix_hashes,
                **duration_record,
            }
            episodes.append(episode)
            print(
                f"arm={arm_label} mode={duration_mode} task={task_id} "
                f"state={realized} success={int(success)} steps={step} "
                f"chunks={chunk_generations} "
                f"intervention={duration_record['intervention_call_fraction']:.3f} "
                f"trace={episode_trace.hexdigest()[:12]}",
                flush=True,
            )

        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "success_rate": float(
                    np.mean([episode["success"] for episode in episodes])
                ),
                "episodes": episodes,
            }
        )

    flat = [episode for task in task_results for episode in task["episodes"]]
    chunks = [
        chunk for episode in flat for chunk in episode["duration_chunks"]
    ]
    if not chunks:
        raise RuntimeError(f"arm {arm_label} produced no duration calls")
    total_steps = sum(int(episode["steps"]) for episode in flat)
    if arm_trace.count != total_steps:
        raise RuntimeError(
            f"arm trace count {arm_trace.count} != total steps {total_steps}"
        )
    changes = [chunk["absolute_duration_intervention"] for chunk in chunks]
    return {
        "arm_label": arm_label,
        "duration_mode": duration_mode,
        "n_tasks": len(task_results),
        "n_episodes": len(flat),
        "n_successes": sum(int(episode["success"]) for episode in flat),
        "success_rate": float(np.mean([episode["success"] for episode in flat])),
        "n_steps": total_steps,
        "n_duration_calls": len(chunks),
        "n_intervened_duration_calls": sum(value > 0 for value in changes),
        "intervention_call_fraction": sum(value > 0 for value in changes)
        / len(chunks),
        "mean_absolute_duration_intervention": float(np.mean(changes)),
        "mean_predicted_duration": float(
            np.mean([chunk["predicted_duration"] for chunk in chunks])
        ),
        "mean_executed_duration": float(
            np.mean([chunk["executed_duration"] for chunk in chunks])
        ),
        "reused_counterfactual_values": sum(
            episode["reused_counterfactual_values"] for episode in flat
        ),
        "executed_action_count": arm_trace.count,
        "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL,
        "executed_action_trace_sha256": arm_trace.hexdigest(),
        "elapsed_seconds": time.time() - started,
        "tasks": task_results,
    }


def _episode_index(arm: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(task["task_id"]), int(episode["state_id"])): episode
        for task in arm["tasks"]
        for episode in task["episodes"]
    }


def _first_difference(left: list[Any], right: list[Any]) -> int | None:
    for index, (first, second) in enumerate(zip(left, right)):
        if first != second:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _predicted_repeat_gate(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    left = _episode_index(first)
    right = _episode_index(second)
    if set(left) != set(right):
        raise RuntimeError("predicted repeats have different task/state manifests")
    fields = (
        "env_seed",
        "policy_seed",
        "success",
        "steps",
        "select_action_calls",
        "chunk_generations",
        "initial_raw_observation_sha256",
        "initial_processed_observation_sha256",
        "executed_action_count",
        "executed_action_trace_sha256",
        "predicted_durations",
        "executed_durations",
    )
    diagnostics: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for key in sorted(left):
        mismatched_fields = [
            field for field in fields if left[key][field] != right[key][field]
        ]
        step_difference = _first_difference(
            left[key]["executed_action_step_sha256"],
            right[key]["executed_action_step_sha256"],
        )
        prefix_difference = _first_difference(
            left[key]["executed_action_prefix_sha256"],
            right[key]["executed_action_prefix_sha256"],
        )
        identity_payload = {
            field: left[key][field] for field in fields
        } | {
            "executed_action_step_sha256": left[key][
                "executed_action_step_sha256"
            ],
            "executed_action_prefix_sha256": left[key][
                "executed_action_prefix_sha256"
            ],
        }
        identities[f"{key[0]}:{key[1]}"] = canonical_json_sha256(identity_payload)
        if mismatched_fields or step_difference is not None or prefix_difference is not None:
            diagnostics.append(
                {
                    "task_id": key[0],
                    "state_id": key[1],
                    "mismatched_summary_fields": mismatched_fields,
                    "first_different_action_step": step_difference,
                    "first_different_action_prefix": prefix_difference,
                    "raw_observation_exact": (
                        left[key]["initial_raw_observation_sha256"]
                        == right[key]["initial_raw_observation_sha256"]
                    ),
                    "processed_observation_exact": (
                        left[key]["initial_processed_observation_sha256"]
                        == right[key]["initial_processed_observation_sha256"]
                    ),
                    "left_steps": left[key]["steps"],
                    "right_steps": right[key]["steps"],
                    "left_action_trace_sha256": left[key][
                        "executed_action_trace_sha256"
                    ],
                    "right_action_trace_sha256": right[key][
                        "executed_action_trace_sha256"
                    ],
                }
            )
    exact = not diagnostics
    return {
        "protocol": "same_process_loaded_policy_predicted_repeat_gate_v2",
        "exact": exact,
        "n_episodes": len(left),
        "n_divergent_episodes": len(diagnostics),
        "episode_identity_sha256_from_first_repeat": identities,
        "diagnostics": diagnostics,
    }


def _initial_state_gate(
    reference: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    left = _episode_index(reference)
    right = _episode_index(comparison)
    if set(left) != set(right):
        raise RuntimeError("intervention arm has a different task/state manifest")
    fields = (
        "env_seed",
        "policy_seed",
        "initial_raw_observation_sha256",
        "initial_processed_observation_sha256",
    )
    diagnostics = [
        {
            "task_id": key[0],
            "state_id": key[1],
            "mismatched_fields": [
                field for field in fields if left[key][field] != right[key][field]
            ],
        }
        for key in sorted(left)
        if any(left[key][field] != right[key][field] for field in fields)
    ]
    return {
        "protocol": "matched_initial_observation_gate_v2",
        "reference_arm": reference["arm_label"],
        "comparison_arm": comparison["arm_label"],
        "exact": not diagnostics,
        "n_episodes": len(left),
        "n_divergent_episodes": len(diagnostics),
        "diagnostics": diagnostics,
    }


def _predicted_sources(
    arm: dict[str, Any], *, identity: dict[str, Any]
) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any]]:
    traces: dict[tuple[int, int], list[int]] = {}
    trace_records: list[dict[str, Any]] = []
    for key, episode in sorted(_episode_index(arm).items()):
        trace = [int(value) for value in episode["predicted_durations"]]
        if not trace or episode["executed_durations"] != trace:
            raise RuntimeError(f"predicted source arm is not identity for {key}")
        traces[key] = trace
        trace_records.append(
            {
                "task_id": key[0],
                "state_id": key[1],
                "env_seed": episode["env_seed"],
                "policy_seed": episode["policy_seed"],
                "predicted_durations": trace,
                "predicted_action_trace_sha256": episode[
                    "executed_action_trace_sha256"
                ],
            }
        )
    manifest = {
        "protocol": "in_process_predicted_duration_shuffle_source_v2",
        **identity,
        "predicted_traces": trace_records,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return traces, manifest


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
    parser.add_argument("--states_per_task", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=100_000)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--shuffle_seed", type=int, default=73_921)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.state_start < 0 or args.states_per_task <= 0:
        raise ValueError("state_start must be nonnegative and states_per_task positive")

    checkpoint = Path(args.ckpt).resolve()
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint}")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_manifest_path = Path(args.fixed_manifest).resolve()
    fixed_duration, fixed_manifest_payload = load_complete_fixed_manifest(
        fixed_manifest_path, config_path
    )

    bootstrap_seed = args.seed_base
    _seed_all(bootstrap_seed)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    if not bool(getattr(policy_cfg, "predict_duration", False)):
        raise ValueError("checkpoint is not a duration-enabled spline policy")
    active_consumers = _active_duration_consumers(policy_cfg)
    if active_consumers:
        raise ValueError(
            "duration ablation requires neutral optional decode controllers; "
            f"active={active_consumers}"
        )
    minimum = int(getattr(policy_cfg, "min_seg"))
    maximum = int(getattr(policy_cfg, "horizon_max"))
    if not minimum <= fixed_duration <= maximum:
        raise ValueError(
            f"fixed manifest duration {fixed_duration} outside [{minimum}, {maximum}]"
        )

    env_cfg = LiberoEnvCfg(
        task=args.suite,
        control_mode="relative",
        control_freq=args.control_freq,
        max_parallel_tasks=1,
    )
    envs_by_suite = make_env(env_cfg, n_envs=1)
    suite_envs = envs_by_suite[args.suite]
    task_ids = _parse_task_ids(args.task_ids, list(suite_envs))
    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    env_source = Path(
        inspect.getfile(type(_base_env(suite_envs[task_ids[0]])))
    ).resolve()

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    controller = DurationInterventionV2(
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
    run_kwargs = {
        "suite_envs": suite_envs,
        "task_ids": task_ids,
        "state_ids": state_ids,
        "seed_base": args.seed_base,
        "policy": policy,
        "controller": controller,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
        "action_key": ACTION,
        "preprocess_observation": preprocess_observation,
    }

    started = time.time()
    arms: dict[str, dict[str, Any]] = {}
    gate: dict[str, Any]
    intervention_initial_state_gates: dict[str, dict[str, Any]] = {}
    shuffle_manifest: dict[str, Any] | None = None
    try:
        arms["predicted"] = _run_arm(
            arm_label="predicted",
            duration_mode="predicted",
            **run_kwargs,
        )
        arms["predicted_repeat"] = _run_arm(
            arm_label="predicted_repeat",
            duration_mode="predicted",
            **run_kwargs,
        )
        gate = _predicted_repeat_gate(
            arms["predicted"], arms["predicted_repeat"]
        )
        if gate["exact"]:
            identity = {
                "checkpoint": str(checkpoint),
                "model_sha256": _sha256(model_path),
                "checkpoint_config_sha256": _sha256(config_path),
                "suite": args.suite,
                "task_ids": task_ids,
                "state_ids": state_ids,
                "seed_base": args.seed_base,
                "control_frequency_hz": args.control_freq,
                "seed_formula": "seed_base + task_id * 1000 + state_id",
            }
            shuffle_sources, shuffle_manifest = _predicted_sources(
                arms["predicted"], identity=identity
            )
            arms["fixed"] = _run_arm(
                arm_label="fixed",
                duration_mode="fixed",
                **run_kwargs,
            )
            intervention_initial_state_gates["fixed"] = _initial_state_gate(
                arms["predicted"], arms["fixed"]
            )
            if intervention_initial_state_gates["fixed"]["exact"]:
                arms["shuffled"] = _run_arm(
                    arm_label="shuffled",
                    duration_mode="shuffled",
                    shuffle_sources=shuffle_sources,
                    **run_kwargs,
                )
                intervention_initial_state_gates["shuffled"] = _initial_state_gate(
                    arms["predicted"], arms["shuffled"]
                )
    finally:
        for env in suite_envs.values():
            try:
                env.close()
            except (RuntimeError, OSError) as error:
                print(f"WARNING: environment close failed: {error}", file=sys.stderr)

    config = json.loads(config_path.read_text())
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    controller_source = Path(inspect.getfile(DurationInterventionV2)).resolve()
    evaluator_path = Path(__file__).resolve()
    base_evaluator_path = Path(inspect.getfile(_explicit_reset)).resolve()
    helper_path = Path(atomic_write_json_new.__code__.co_filename).resolve()
    stats_path = None
    stats_hash = None
    stats_name = config.get("spline_stats_file_v2")
    if stats_name:
        candidate = policy_source.parent / stats_name
        if candidate.exists():
            stats_path = str(candidate)
            stats_hash = _sha256(candidate)

    complete = (
        gate["exact"]
        and set(intervention_initial_state_gates) == {"fixed", "shuffled"}
        and all(
            arm_gate["exact"]
            for arm_gate in intervention_initial_state_gates.values()
        )
    )
    status = (
        "complete"
        if complete
        else (
            "determinism_gate_failed"
            if not gate["exact"]
            else "intervention_initial_state_gate_failed"
        )
    )
    result = {
        "schema_version": 2,
        "protocol": EVALUATION_PROTOCOL,
        "status": status,
        "intervention_protocol": INTERVENTION_PROTOCOL,
        "duration_modes": ["predicted", "fixed", "shuffled"],
        "execution_order": [
            "predicted",
            "predicted_repeat",
            "fixed",
            "shuffled",
        ],
        "execution_scope": (
            "one process, one loaded policy, one loaded environment set; "
            "explicit reset and reseed per task/state/arm"
        ),
        "duration_estimand": (
            "closed-loop total effect of observation-aligned duration at fixed "
            "duration-enabled shape checkpoint"
        ),
        "duration_prediction_source": (
            "median of batch-1 policy.last_predicted_T_batch computed once by "
            "production _get_action_chunk"
        ),
        "determinism_gate": gate,
        "intervention_initial_state_gates": intervention_initial_state_gates,
        "fixed_duration": fixed_duration,
        "fixed_manifest": str(fixed_manifest_path),
        "fixed_manifest_sha256": _sha256(fixed_manifest_path),
        "fixed_manifest_protocol": fixed_manifest_payload.get("protocol"),
        "fixed_manifest_complete_training_scan": fixed_manifest_payload[
            "complete_training_scan"
        ],
        "shuffle_source_manifest": shuffle_manifest,
        "shuffle_seed": args.shuffle_seed,
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
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "base_evaluator_path": str(base_evaluator_path),
        "base_evaluator_sha256": _sha256(base_evaluator_path),
        "controller_path": str(controller_source),
        "controller_sha256": _sha256(controller_source),
        "helper_path": str(helper_path),
        "helper_sha256": _sha256(helper_path),
        "arms": arms,
    }
    atomic_write_json_new(output, result)
    print(
        f"saved {output}: status={result['status']} "
        f"gate_divergences={gate['n_divergent_episodes']}",
        flush=True,
    )
    if not complete:
        raise RuntimeError(
            f"duration-v2 validity gate failed with status={status}; "
            "only validity-gated arms were run"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
