#!/usr/bin/env python3
"""Claim-safe analysis for LIBERO duration-ablation v3 artifacts.

The evaluator writes one combined JSON per checkpoint.  This analyzer treats
those JSON files as untrusted scientific inputs: it validates their embedded
identities and complete paired blocks before computing any result.  A
single-checkpoint result is always labelled exploratory; training-seed
intervals are unavailable until at least three distinct model hashes are
provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
PROTOCOL = "duration_v3_claim_safe_analysis_v1"
EVALUATION_PROTOCOL = "libero_duration_stochastic_paired_block_v3"
WATERMARK_PREFIX = "STOCHASTIC_PAIRED_BLOCK_RESULT"
ARMS = (
    "predicted_donor",
    "predicted_eval",
    "fixed",
    "shuffled",
    "predicted_closure",
)
EFFICACY_ARMS = ("predicted_eval", "fixed", "shuffled")
EFFICACY_PERMUTATIONS = (
    ("predicted_eval", "fixed", "shuffled"),
    ("predicted_eval", "shuffled", "fixed"),
    ("fixed", "predicted_eval", "shuffled"),
    ("fixed", "shuffled", "predicted_eval"),
    ("shuffled", "predicted_eval", "fixed"),
    ("shuffled", "fixed", "predicted_eval"),
)
PREDICTED_ARMS = (
    "predicted_donor",
    "predicted_eval",
    "predicted_closure",
)
PRIMARY_CONTROLS = ("fixed", "shuffled")
PAIR_NAMES = {
    "donor_vs_eval": ("predicted_donor", "predicted_eval"),
    "donor_vs_closure": ("predicted_donor", "predicted_closure"),
    "eval_vs_closure": ("predicted_eval", "predicted_closure"),
}
SOURCE_IDENTITY_FIELDS = (
    "evaluator_sha256",
    "controller_sha256",
    "stochastic_helper_sha256",
    "v2_hashing_sha256",
    "hidden_state_helper_sha256",
    "base_evaluator_sha256",
    "action_helper_sha256",
    "policy_source_sha256",
    "env_source_sha256",
    "stats_sha256",
    "lerobot_commit",
)
PAIRING_FIELDS = (
    "seed_agreement",
    "raw_initial_observation_agreement",
    "processed_initial_observation_agreement",
    "mujoco_integration_initial_state_agreement",
    "compiled_model_xml_agreement",
)
EPISODE_IDENTITY_FIELDS = (
    "task_id",
    "state_id",
    "repeat_index",
    "env_seed_u32",
    "policy_seed_u64",
)


class ValidationError(ValueError):
    """Raised before analysis when a result is incomplete or inconsistent."""


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_sha256(value: Any, context: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    valid = isinstance(value, str) and len(value) == 64
    if valid:
        try:
            bytes.fromhex(value)
        except ValueError:
            valid = False
    _require(valid, f"{context}: expected a SHA-256 hex digest")


def _embedded_hash(payload: dict[str, Any], field: str, context: str) -> None:
    unhashed = dict(payload)
    reported = unhashed.pop(field, None)
    _require(
        reported == canonical_json_sha256(unhashed),
        f"{context}: invalid embedded {field}",
    )


def _block_seeds(
    seed_base: int, task_id: int, state_id: int, repeat_index: int
) -> tuple[int, int]:
    payload = (
        f"duration_stochastic_v3|{seed_base}|{task_id}|{state_id}|{repeat_index}"
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:12], "little"),
    )


def _close(left: Any, right: float, context: str, tolerance: float = 1e-12) -> None:
    _require(
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=tolerance, abs_tol=tolerance),
        f"{context}: expected {right}, found {left}",
    )


def _episode_exact(first: dict[str, Any], second: dict[str, Any]) -> bool:
    exact_fields = (
        "success",
        "steps",
        "initial_raw_observation_sha256",
        "initial_processed_observation_sha256",
        "initial_mujoco_integration_state_sha256",
        "compiled_model_xml_sha256",
        "raw_observation_step_sha256",
        "processed_observation_step_sha256",
        "mujoco_integration_state_step_sha256",
        "executed_action_step_sha256",
        "executed_action_prefix_sha256",
        "executed_action_trace_sha256",
        "predicted_durations",
    )
    return all(first.get(field) == second.get(field) for field in exact_fields)


def _validate_duration_record(episode: dict[str, Any], context: str) -> None:
    chunks = episode.get("duration_chunks")
    _require(isinstance(chunks, list) and chunks, f"{context}: empty duration trace")
    predicted = episode.get("predicted_durations")
    executed = episode.get("executed_durations")
    _require(
        isinstance(predicted, list) and isinstance(executed, list),
        f"{context}: missing duration arrays",
    )
    n_calls = len(chunks)
    _require(
        len(predicted) == len(executed) == n_calls,
        f"{context}: duration trace lengths disagree",
    )
    _require(
        episode.get("n_duration_calls") == n_calls,
        f"{context}: n_duration_calls mismatch",
    )
    changed = 0
    absolute_sum = 0.0
    for index, (chunk, pred, used) in enumerate(zip(chunks, predicted, executed)):
        _require(isinstance(chunk, dict), f"{context}: chunk {index} is not an object")
        _require(
            chunk.get("predicted_duration") == pred
            and chunk.get("executed_duration") == used,
            f"{context}: chunk {index} does not bind duration arrays",
        )
        difference = abs(int(pred) - int(used))
        _require(
            chunk.get("absolute_duration_intervention") == difference,
            f"{context}: chunk {index} intervention mismatch",
        )
        changed += int(pred != used)
        absolute_sum += difference
    _require(
        episode.get("n_intervened_duration_calls") == changed,
        f"{context}: changed-call count mismatch",
    )
    _close(
        episode.get("intervention_call_fraction"),
        changed / n_calls,
        f"{context}: intervention fraction",
    )
    _close(
        episode.get("mean_absolute_duration_intervention"),
        absolute_sum / n_calls,
        f"{context}: mean absolute intervention",
    )


def _episode_initial_agreement(
    donor: dict[str, Any], episode: dict[str, Any]
) -> dict[str, bool]:
    return {
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


def _validate_block(
    block: dict[str, Any],
    *,
    artifact: dict[str, Any],
    task_manifest: dict[str, Any],
    block_spec: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    coordinates = (
        int(block_spec["task_id"]),
        int(block_spec["state_id"]),
        int(block_spec["repeat_index"]),
    )
    task_id, state_id, repeat_index = coordinates
    _require(
        (block.get("task_id"), block.get("state_id"), block.get("repeat_index"))
        == coordinates,
        f"{context}: block coordinates mismatch",
    )
    _require(
        block.get("task_block_ordinal") == block_spec.get("task_block_ordinal"),
        f"{context}: task block ordinal mismatch",
    )
    expected_order = list(block_spec["arm_order"])
    _require(block.get("arm_order") == expected_order, f"{context}: arm order mismatch")
    expected_permutation = list(block_spec["efficacy_permutation"])
    _require(
        block.get("efficacy_permutation") == expected_permutation,
        f"{context}: efficacy permutation mismatch",
    )
    _require(
        expected_order == ["predicted_donor", *expected_permutation, "predicted_closure"],
        f"{context}: malformed five-arm order",
    )
    episodes = block.get("episodes")
    _require(
        isinstance(episodes, dict) and set(episodes) == set(ARMS),
        f"{context}: block must contain exactly five arms",
    )
    _require(
        block.get("task_order_manifest_sha256") == task_manifest["manifest_sha256"],
        f"{context}: task-order binding mismatch",
    )
    runtime_sha = block.get("execution_runtime_sha256")
    runtimes = artifact.get("execution_runtime_manifests")
    _require(
        isinstance(runtimes, dict) and runtime_sha in runtimes,
        f"{context}: missing execution runtime manifest",
    )
    _require(
        runtime_sha == canonical_json_sha256(runtimes[runtime_sha]),
        f"{context}: execution runtime hash mismatch",
    )

    expected_identity = {
        "task_id": task_id,
        "state_id": state_id,
        "repeat_index": repeat_index,
        "env_seed_u32": int(block_spec["env_seed_u32"]),
        "policy_seed_u64": int(block_spec["policy_seed_u64"]),
    }
    for arm in ARMS:
        episode = episodes[arm]
        _require(isinstance(episode, dict), f"{context}/{arm}: episode is not an object")
        _require(episode.get("arm_label") == arm, f"{context}/{arm}: label mismatch")
        expected_mode = "predicted" if arm.startswith("predicted_") else arm
        _require(
            episode.get("duration_mode") == expected_mode,
            f"{context}/{arm}: duration mode mismatch",
        )
        for field, value in expected_identity.items():
            _require(
                episode.get(field) == value,
                f"{context}/{arm}: {field} mismatch",
            )
        _validate_duration_record(episode, f"{context}/{arm}")

    donor = episodes["predicted_donor"]
    pairings = block.get("initial_pairing_by_arm")
    _require(
        isinstance(pairings, dict) and set(pairings) == set(expected_order[1:]),
        f"{context}: initial-pairing coverage mismatch",
    )
    for arm in expected_order[1:]:
        actual = _episode_initial_agreement(donor, episodes[arm])
        reported = pairings[arm]
        for field in PAIRING_FIELDS:
            _require(
                reported.get(field) is actual[field],
                f"{context}/{arm}: reported {field} mismatch",
            )
            _require(actual[field], f"{context}/{arm}: initial pairing failed: {field}")

    shuffle_manifest = block.get("shuffle_source_manifest")
    _require(isinstance(shuffle_manifest, dict), f"{context}: missing shuffle manifest")
    _embedded_hash(shuffle_manifest, "manifest_sha256", f"{context}/shuffle_manifest")
    source_expected = {
        "resume_identity_sha256": artifact["resume_identity_sha256"],
        "checkpoint": artifact["checkpoint"],
        "model_sha256": artifact["model_sha256"],
        "checkpoint_config_sha256": artifact["checkpoint_config_sha256"],
        "suite": artifact["suite"],
        "configured_task_ids": artifact["task_ids"],
        "configured_state_ids": artifact["state_ids"],
        "configured_repeats": artifact["repeats"],
        "task_order_manifest_sha256": task_manifest["manifest_sha256"],
        **expected_identity,
        "control_frequency_hz": artifact["control_frequency_hz"],
        "shuffle_seed": artifact["shuffle_seed"],
        "predicted_durations": donor["predicted_durations"],
        "donor_action_trace_sha256": donor["executed_action_trace_sha256"],
    }
    for field, value in source_expected.items():
        _require(
            shuffle_manifest.get(field) == value,
            f"{context}: shuffle manifest {field} mismatch",
        )
    shuffled = episodes["shuffled"]
    _require(
        shuffled.get("shuffle_source_manifest_sha256")
        == shuffle_manifest["manifest_sha256"],
        f"{context}: shuffled episode is not bound to donor manifest",
    )
    _require(
        shuffled.get("counterfactual_source_length")
        == len(donor["predicted_durations"]),
        f"{context}: shuffled source length mismatch",
    )
    return {
        "task_id": task_id,
        "state_id": state_id,
        "repeat_index": repeat_index,
        "arm_order": expected_order,
        "efficacy_permutation": expected_permutation,
        "episodes": episodes,
        "predicted_exact": {
            name: _episode_exact(episodes[first], episodes[second])
            for name, (first, second) in PAIR_NAMES.items()
        },
    }


def validate_artifact(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"{path}: cannot read JSON: {error}") from error
    context = str(path)
    _require(isinstance(artifact, dict), f"{context}: root must be an object")
    _require(artifact.get("schema_version") == 3, f"{context}: unsupported schema")
    _require(
        artifact.get("protocol") == EVALUATION_PROTOCOL,
        f"{context}: wrong evaluator protocol",
    )
    _require(
        str(artifact.get("watermark", "")).startswith(WATERMARK_PREFIX),
        f"{context}: missing stochastic-result watermark",
    )
    _require(artifact.get("scientific_arms") == list(EFFICACY_ARMS), f"{context}: scientific arms changed")
    _require(artifact.get("trace_only_arms") == ["predicted_donor", "predicted_closure"], f"{context}: trace-only arms changed")
    _require(artifact.get("primary_predicted_arm") == "predicted_eval", f"{context}: primary arm changed")

    resume = artifact.get("resume_identity")
    _require(isinstance(resume, dict), f"{context}: missing resume identity")
    _require(
        artifact.get("resume_identity_sha256") == canonical_json_sha256(resume),
        f"{context}: resume identity hash mismatch",
    )
    top_resume_pairs = {
        "checkpoint": "checkpoint",
        "model_sha256": "model_sha256",
        "checkpoint_config_sha256": "checkpoint_config_sha256",
        "fixed_manifest_sha256": "fixed_manifest_sha256",
        "suite": "suite",
        "task_ids": "task_ids",
        "state_ids": "state_ids",
        "repeats": "repeats",
        "seed_base": "seed_base",
        "order_seed": "order_seed",
        "control_frequency_hz": "control_frequency_hz",
        "shuffle_seed": "shuffle_seed",
    }
    for top, embedded in top_resume_pairs.items():
        _require(
            artifact.get(top) == resume.get(embedded),
            f"{context}: top-level {top} disagrees with resume identity",
        )
    for field in SOURCE_IDENTITY_FIELDS:
        _require(
            artifact.get(field) == resume.get(field),
            f"{context}: source identity mismatch for {field}",
        )
        if field.endswith("_sha256"):
            _require_sha256(
                artifact.get(field), context + f": {field}", allow_none=True
            )
    for field in (
        "model_sha256",
        "checkpoint_config_sha256",
        "fixed_manifest_sha256",
        "resume_identity_sha256",
    ):
        _require_sha256(artifact.get(field), context + f": {field}")
    _require(
        artifact.get("fixed_manifest_complete_training_scan") is True,
        f"{context}: fixed prior is not a complete training scan",
    )
    _require(
        artifact.get("fixed_manifest_protocol")
        == "production_event_duration_training_prior_v2_parquet",
        f"{context}: unexpected fixed-prior protocol",
    )
    top_exact_requirement = artifact.get("exact_predicted_replay_required")
    resume_exact_requirement = resume.get("require_exact_predicted_replay")
    if top_exact_requirement is None and resume_exact_requirement is None:
        exact_replay_required = False  # legacy diagnostic artifact
    else:
        _require(
            isinstance(top_exact_requirement, bool)
            and isinstance(resume_exact_requirement, bool)
            and top_exact_requirement is resume_exact_requirement,
            f"{context}: exact-replay requirement disagrees with resume identity",
        )
        exact_replay_required = top_exact_requirement
    task_ids = artifact.get("task_ids")
    state_ids = artifact.get("state_ids")
    repeats = artifact.get("repeats")
    _require(
        isinstance(task_ids, list) and task_ids and len(task_ids) == len(set(task_ids)),
        f"{context}: task ids must be nonempty and unique",
    )
    _require(
        isinstance(state_ids, list) and state_ids and len(state_ids) == len(set(state_ids)),
        f"{context}: state ids must be nonempty and unique",
    )
    _require(isinstance(repeats, int) and repeats > 0, f"{context}: invalid repeats")

    runtimes = artifact.get("execution_runtime_manifests")
    _require(isinstance(runtimes, dict) and runtimes, f"{context}: no runtime manifests")
    for digest, manifest in runtimes.items():
        _require(digest == canonical_json_sha256(manifest), f"{context}: runtime manifest hash mismatch")

    tasks = artifact.get("tasks")
    _require(isinstance(tasks, list), f"{context}: tasks must be a list")
    _require(
        [task.get("task_id") for task in tasks] == task_ids,
        f"{context}: task records do not match configured task ids/order",
    )
    normalized_blocks: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    seen_env_seeds: set[int] = set()
    seen_policy_seeds: set[int] = set()
    task_manifest_hashes: dict[str, str] = {}
    aggregate_permutation_counts = {
        "/".join(permutation): 0 for permutation in EFFICACY_PERMUTATIONS
    }
    expected_coordinates = {
        (int(task_id), int(state_id), repeat_index)
        for task_id in task_ids
        for state_id in state_ids
        for repeat_index in range(repeats)
    }
    for task in tasks:
        task_id = int(task["task_id"])
        manifest = task.get("task_order_manifest")
        _require(isinstance(manifest, dict), f"{context}/task{task_id}: no order manifest")
        _embedded_hash(manifest, "manifest_sha256", f"{context}/task{task_id}/manifest")
        expected_manifest_identity = {
            "suite": artifact["suite"],
            "task_id": task_id,
            "state_ids": state_ids,
            "repeats": repeats,
            "seed_base": artifact["seed_base"],
            "order_seed": artifact["order_seed"],
        }
        for field, value in expected_manifest_identity.items():
            _require(manifest.get(field) == value, f"{context}/task{task_id}: manifest {field} mismatch")
        specs = manifest.get("blocks")
        _require(isinstance(specs, list), f"{context}/task{task_id}: manifest blocks missing")
        specs_by_coord: dict[tuple[int, int, int], dict[str, Any]] = {}
        manifest_counts = {
            "/".join(permutation): 0 for permutation in EFFICACY_PERMUTATIONS
        }
        for ordinal, spec in enumerate(specs):
            coord = (task_id, int(spec["state_id"]), int(spec["repeat_index"]))
            _require(coord not in specs_by_coord, f"{context}: duplicate manifest coordinate {coord}")
            permutation = tuple(spec.get("efficacy_permutation", ()))
            _require(
                permutation in EFFICACY_PERMUTATIONS,
                f"{context}: invalid efficacy permutation at {coord}",
            )
            _require(
                spec.get("task_block_ordinal") == ordinal,
                f"{context}: task block ordinal mismatch at {coord}",
            )
            expected_env_seed, expected_policy_seed = _block_seeds(
                int(artifact["seed_base"]), *coord
            )
            _require(
                spec.get("env_seed_u32") == expected_env_seed
                and spec.get("policy_seed_u64") == expected_policy_seed,
                f"{context}: derived block seeds mismatch at {coord}",
            )
            _require(
                expected_env_seed not in seen_env_seeds
                and expected_policy_seed not in seen_policy_seeds,
                f"{context}: block seed collision at {coord}",
            )
            seen_env_seeds.add(expected_env_seed)
            seen_policy_seeds.add(expected_policy_seed)
            name = "/".join(permutation)
            manifest_counts[name] += 1
            aggregate_permutation_counts[name] += 1
            specs_by_coord[coord] = {"task_id": task_id, **spec}
        _require(
            manifest.get("permutation_counts") == manifest_counts,
            f"{context}/task{task_id}: permutation counts mismatch",
        )
        _require(
            max(manifest_counts.values()) - min(manifest_counts.values()) <= 1,
            f"{context}/task{task_id}: permutation quotas are unbalanced",
        )
        task_blocks = task.get("blocks")
        _require(isinstance(task_blocks, list), f"{context}/task{task_id}: blocks missing")
        _require(task.get("n_blocks") == len(task_blocks), f"{context}/task{task_id}: n_blocks mismatch")
        _require(set(specs_by_coord) == {(task_id, int(s), r) for s in state_ids for r in range(repeats)}, f"{context}/task{task_id}: incomplete manifest grid")
        for block in task_blocks:
            coord = (int(block.get("task_id")), int(block.get("state_id")), int(block.get("repeat_index")))
            _require(coord in specs_by_coord, f"{context}: block {coord} absent from manifest")
            _require(coord not in seen, f"{context}: duplicate block coordinate {coord}")
            seen.add(coord)
            normalized_blocks.append(
                _validate_block(
                    block,
                    artifact=artifact,
                    task_manifest=manifest,
                    block_spec=specs_by_coord[coord],
                    context=f"{context}/task{coord[0]}/state{coord[1]}/repeat{coord[2]}",
                )
            )
        task_manifest_hashes[str(task_id)] = manifest["manifest_sha256"]
    _require(seen == expected_coordinates, f"{context}: block grid is incomplete or extraneous")
    _require(artifact.get("n_blocks") == len(normalized_blocks), f"{context}: n_blocks mismatch")
    _require(artifact.get("n_episodes") == len(normalized_blocks) * len(ARMS), f"{context}: n_episodes mismatch")
    _require(
        artifact.get("efficacy_permutation_counts") == aggregate_permutation_counts,
        f"{context}: aggregate permutation counts mismatch",
    )
    order_gate = artifact.get("order_balance_gate", {})
    _require(order_gate.get("passed") is True, f"{context}: order-balance gate failed")

    exact_pairs = {
        name: all(block["predicted_exact"][name] for block in normalized_blocks)
        for name in PAIR_NAMES
    }
    exact_all = all(exact_pairs.values())
    reported_exact = artifact.get("exact_predicted_replay_all_pairs")
    if reported_exact is not None:
        _require(
            reported_exact is exact_all,
            f"{context}: exact replay summary disagrees with traces",
        )
    if exact_replay_required:
        _require(exact_all, f"{context}: required exact predicted replay failed")

    pairing_summary = artifact.get("initial_pairing_summary", {})
    n_pairings = len(normalized_blocks) * 4
    _require(artifact.get("initial_pairing_complete") is True, f"{context}: initial-pairing gate failed")
    _require(pairing_summary.get("n_arm_episode_pairs") == n_pairings, f"{context}: pairing count mismatch")
    for field in (
        "n_seed_agreements",
        "n_raw_initial_observation_agreements",
        "n_processed_initial_observation_agreements",
        "n_mujoco_integration_initial_state_agreements",
        "n_compiled_model_xml_agreements",
    ):
        _require(pairing_summary.get(field) == n_pairings, f"{context}: incomplete {field}")

    threshold = float(resume["shuffle_strength_threshold"])
    shuffled_episodes = [block["episodes"]["shuffled"] for block in normalized_blocks]
    changed = sum(ep["n_intervened_duration_calls"] for ep in shuffled_episodes)
    calls = sum(ep["n_duration_calls"] for ep in shuffled_episodes)
    observed = changed / calls
    gate = artifact.get("shuffle_strength_gate", {})
    _close(gate.get("observed_changed_call_fraction"), observed, f"{context}: shuffle gate")
    _require(gate.get("target_minimum") == threshold, f"{context}: shuffle threshold mismatch")
    task_strength: dict[str, float] = {}
    for task_id in task_ids:
        selected = [ep for block in normalized_blocks if block["task_id"] == task_id for ep in [block["episodes"]["shuffled"]]]
        task_changed = sum(ep["n_intervened_duration_calls"] for ep in selected)
        task_calls = sum(ep["n_duration_calls"] for ep in selected)
        task_strength[str(task_id)] = task_changed / task_calls
    passed = observed >= threshold and all(value >= threshold for value in task_strength.values())
    _require(gate.get("passed") is passed, f"{context}: shuffle pass flag mismatch")
    _require(passed, f"{context}: shuffle strength gate failed")

    return {
        "path": str(path.resolve()),
        "artifact_sha256": file_sha256(path),
        "artifact": artifact,
        "model_sha256": artifact["model_sha256"],
        "checkpoint_config_sha256": artifact["checkpoint_config_sha256"],
        "fixed_manifest_sha256": artifact["fixed_manifest_sha256"],
        "source_identity": {field: artifact.get(field) for field in SOURCE_IDENTITY_FIELDS},
        "task_manifest_hashes": task_manifest_hashes,
        "blocks": normalized_blocks,
        "exact_predicted_replay": exact_all,
        "exact_predicted_replay_by_pair": exact_pairs,
        "shuffle_strength": observed,
        "shuffle_strength_by_task": task_strength,
        "mujoco_gl_values": sorted(
            {str(manifest.get("mujoco_gl")) for manifest in runtimes.values()}
        ),
    }


def _cross_validate(validated: list[dict[str, Any]]) -> None:
    _require(validated, "at least one input is required")
    reference = validated[0]
    reference_artifact = reference["artifact"]
    fields = (
        "suite",
        "task_ids",
        "state_ids",
        "repeats",
        "seed_base",
        "order_seed",
        "control_frequency_hz",
        "shuffle_seed",
        "fixed_duration",
        "policy_type",
        "n_action_steps",
    )
    for current in validated[1:]:
        artifact = current["artifact"]
        for field in fields:
            _require(
                artifact.get(field) == reference_artifact.get(field),
                f"cross-input {field} mismatch",
            )
        _require(
            current["checkpoint_config_sha256"]
            == reference["checkpoint_config_sha256"],
            "cross-input checkpoint config hash mismatch",
        )
        _require(
            current["fixed_manifest_sha256"] == reference["fixed_manifest_sha256"],
            "cross-input fixed-prior identity mismatch",
        )
        _require(current["source_identity"] == reference["source_identity"], "cross-input source identity mismatch")
        _require(current["task_manifest_hashes"] == reference["task_manifest_hashes"], "cross-input task/order manifest mismatch")
        _require(current["mujoco_gl_values"] == reference["mujoco_gl_values"], "cross-input render/runtime backend mismatch")
    models = [item["model_sha256"] for item in validated]
    _require(len(models) == len(set(models)), "multiple inputs must represent distinct model hashes")


def _merge_validated_task_shards(
    validated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge disjoint task shards without treating them as training seeds.

    Each shard is fully validated before reaching this function, including its
    own resume identity and block-local manifests.  We therefore retain those
    immutable shard artifacts and merge only their normalized analysis views.
    Shards can be joined iff they use the same model and every non-task
    scientific identity agrees.  Overlapping task IDs fail closed.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in validated:
        artifact = item["artifact"]
        identity = {
            "checkpoint": artifact["checkpoint"],
            "model_sha256": item["model_sha256"],
            "checkpoint_config_sha256": item["checkpoint_config_sha256"],
            "fixed_manifest_sha256": item["fixed_manifest_sha256"],
            "source_identity": item["source_identity"],
            "suite": artifact["suite"],
            "state_ids": artifact["state_ids"],
            "repeats": artifact["repeats"],
            "seed_base": artifact["seed_base"],
            "order_seed": artifact["order_seed"],
            "control_frequency_hz": artifact["control_frequency_hz"],
            "shuffle_seed": artifact["shuffle_seed"],
            "fixed_duration": artifact["fixed_duration"],
            "policy_type": artifact["policy_type"],
            "n_action_steps": artifact["n_action_steps"],
            "mujoco_gl_values": item["mujoco_gl_values"],
        }
        groups[canonical_json_sha256(identity)].append(item)

    merged_items: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) == 1:
            item = dict(group[0])
            item["input_paths"] = [item["path"]]
            item["input_sha256s"] = [item["artifact_sha256"]]
            merged_items.append(item)
            continue

        task_ids: list[int] = []
        task_manifest_hashes: dict[str, str] = {}
        blocks: list[dict[str, Any]] = []
        paths: list[str] = []
        digests: list[str] = []
        for item in group:
            current_tasks = [int(value) for value in item["artifact"]["task_ids"]]
            overlap = sorted(set(task_ids).intersection(current_tasks))
            _require(
                not overlap,
                f"same-model task shards overlap on task IDs {overlap}",
            )
            task_ids.extend(current_tasks)
            for key, value in item["task_manifest_hashes"].items():
                _require(
                    key not in task_manifest_hashes,
                    f"duplicate task manifest for task {key}",
                )
                task_manifest_hashes[key] = value
            blocks.extend(item["blocks"])
            paths.append(item["path"])
            digests.append(item["artifact_sha256"])

        reference = group[0]
        artifact = dict(reference["artifact"])
        artifact["task_ids"] = sorted(task_ids)
        blocks.sort(key=lambda block: (block["task_id"], block["state_id"], block["repeat_index"]))
        shuffled = [block["episodes"]["shuffled"] for block in blocks]
        changed = sum(ep["n_intervened_duration_calls"] for ep in shuffled)
        calls = sum(ep["n_duration_calls"] for ep in shuffled)
        exact_pairs = {
            name: all(
                _episode_exact(
                    block["episodes"][first], block["episodes"][second]
                )
                for block in blocks
            )
            for name, (first, second) in PAIR_NAMES.items()
        }
        merged_items.append(
            {
                **reference,
                "path": " + ".join(paths),
                "artifact_sha256": canonical_json_sha256(
                    {"task_shard_sha256s": sorted(digests)}
                ),
                "artifact": artifact,
                "task_manifest_hashes": task_manifest_hashes,
                "blocks": blocks,
                "exact_predicted_replay": all(exact_pairs.values()),
                "exact_predicted_replay_by_pair": exact_pairs,
                "shuffle_strength": changed / calls,
                "shuffle_strength_by_task": {
                    key: value
                    for item in group
                    for key, value in item["shuffle_strength_by_task"].items()
                },
                "input_paths": paths,
                "input_sha256s": digests,
            }
        )
    return sorted(merged_items, key=lambda item: item["model_sha256"])


def exact_mcnemar(success_first: Iterable[bool], success_second: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(success_first, success_second, strict=True))
    first_only = sum(bool(first) and not bool(second) for first, second in pairs)
    second_only = sum(not bool(first) and bool(second) for first, second in pairs)
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_success_second_failure": first_only,
        "first_failure_second_success": second_only,
        "n_discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, dict[str, Any]] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (total - rank + 1) * p_value))
        adjusted[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": running,
            "reject_familywise_0.05": running <= 0.05,
        }
    return adjusted


def _contrast(blocks: list[dict[str, Any]], control: str) -> dict[str, Any]:
    predicted = [bool(block["episodes"]["predicted_eval"]["success"]) for block in blocks]
    comparator = [bool(block["episodes"][control]["success"]) for block in blocks]
    mcnemar = exact_mcnemar(predicted, comparator)
    predicted_rate = statistics.fmean(predicted)
    control_rate = statistics.fmean(comparator)
    return {
        "first_arm": "predicted_eval",
        "second_arm": control,
        "n_paired_blocks": len(blocks),
        "predicted_successes": sum(predicted),
        "control_successes": sum(comparator),
        "predicted_success_rate": predicted_rate,
        "control_success_rate": control_rate,
        "delta_predicted_minus_control_pp": 100.0 * (predicted_rate - control_rate),
        "discordants": mcnemar,
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def hierarchical_task_state_bootstrap(
    blocks: list[dict[str, Any]],
    control: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    values: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for block in blocks:
        difference = float(block["episodes"]["predicted_eval"]["success"]) - float(block["episodes"][control]["success"])
        values[block["task_id"]][block["state_id"]].append(difference)
    tasks = sorted(values)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        task_means: list[float] = []
        for sampled_task in rng.choices(tasks, k=len(tasks)):
            states = sorted(values[sampled_task])
            state_means = [
                statistics.fmean(values[sampled_task][sampled_state])
                for sampled_state in rng.choices(states, k=len(states))
            ]
            task_means.append(statistics.fmean(state_means))
        draws.append(100.0 * statistics.fmean(task_means))
    draws.sort()
    return {
        "protocol": "paired_hierarchical_task_then_state_percentile_bootstrap_v1",
        "unit": "percentage_points_predicted_eval_minus_control",
        "repeat_handling": "all repeats in a sampled task/state remain together and are averaged",
        "samples": samples,
        "seed": seed,
        "estimate_pp": 100.0 * statistics.fmean(
            [
                statistics.fmean(
                    [statistics.fmean(values[task][state]) for state in values[task]]
                )
                for task in tasks
            ]
        ),
        "percentile_95_ci_pp": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def _t_critical_975(df: int) -> float:
    table = {
        2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
        12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
        17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
        22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96 if df >= 120 else 2.0)


def _training_seed_summary(
    per_checkpoint: list[dict[str, Any]], control: str
) -> dict[str, Any]:
    effects = [item["primary_contrasts"][control]["delta_predicted_minus_control_pp"] for item in per_checkpoint]
    if len(effects) < 3:
        return {
            "available": False,
            "n_independent_checkpoints": len(effects),
            "reason": "training-seed confidence intervals require at least 3 distinct model hashes",
            "checkpoint_effects_pp": effects,
        }
    mean = statistics.fmean(effects)
    standard_error = statistics.stdev(effects) / math.sqrt(len(effects))
    critical = _t_critical_975(len(effects) - 1)
    return {
        "available": True,
        "protocol": "two_sided_95pct_student_t_interval_over_independent_checkpoint_effects",
        "n_independent_checkpoints": len(effects),
        "checkpoint_effects_pp": effects,
        "mean_effect_pp": mean,
        "95_ci_pp": [mean - critical * standard_error, mean + critical * standard_error],
    }


def _period_order_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm_period: dict[str, dict[str, list[bool]]] = {
        arm: defaultdict(list) for arm in EFFICACY_ARMS
    }
    by_permutation: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    relative: dict[str, dict[str, list[float]]] = {
        control: {"predicted_before_control": [], "predicted_after_control": []}
        for control in PRIMARY_CONTROLS
    }
    for block in blocks:
        permutation = "/".join(block["efficacy_permutation"])
        for arm in EFFICACY_ARMS:
            period = block["efficacy_permutation"].index(arm) + 1
            by_arm_period[arm][str(period)].append(bool(block["episodes"][arm]["success"]))
            by_permutation[permutation][arm].append(bool(block["episodes"][arm]["success"]))
        for control in PRIMARY_CONTROLS:
            key = (
                "predicted_before_control"
                if block["efficacy_permutation"].index("predicted_eval")
                < block["efficacy_permutation"].index(control)
                else "predicted_after_control"
            )
            relative[control][key].append(
                float(block["episodes"]["predicted_eval"]["success"])
                - float(block["episodes"][control]["success"])
            )

    def summarize_binary(values: list[bool]) -> dict[str, Any]:
        return {"n": len(values), "successes": sum(values), "success_rate": statistics.fmean(values)}

    period = {
        arm: {position: summarize_binary(values) for position, values in sorted(periods.items())}
        for arm, periods in by_arm_period.items()
    }
    ranges = {
        arm: 100.0 * (max(cell["success_rate"] for cell in cells.values()) - min(cell["success_rate"] for cell in cells.values()))
        for arm, cells in period.items()
    }
    return {
        "efficacy_period_by_arm": period,
        "success_rate_range_across_periods_pp": ranges,
        "efficacy_permutation": {
            permutation: {arm: summarize_binary(values) for arm, values in arms.items()}
            for permutation, arms in sorted(by_permutation.items())
        },
        "relative_order_contrast": {
            control: {
                key: {
                    "n": len(values),
                    "mean_delta_predicted_minus_control_pp": (
                        100.0 * statistics.fmean(values) if values else None
                    ),
                }
                for key, values in strata.items()
            }
            for control, strata in relative.items()
        },
        "interpretation": "descriptive sensitivity analysis; it is not an additional randomized efficacy test",
    }


def _predicted_repeatability(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (first_arm, second_arm) in PAIR_NAMES.items():
        first = [bool(block["episodes"][first_arm]["success"]) for block in blocks]
        second = [bool(block["episodes"][second_arm]["success"]) for block in blocks]
        mcnemar = exact_mcnemar(first, second)
        exact = [
            _episode_exact(block["episodes"][first_arm], block["episodes"][second_arm])
            for block in blocks
        ]
        output[name] = {
            "first_arm": first_arm,
            "second_arm": second_arm,
            "n_blocks": len(blocks),
            "first_success_rate": statistics.fmean(first),
            "second_success_rate": statistics.fmean(second),
            "delta_second_minus_first_pp": 100.0 * (statistics.fmean(second) - statistics.fmean(first)),
            "outcome_agreements": sum(a == b for a, b in zip(first, second)),
            "outcome_agreement_rate": statistics.fmean(a == b for a, b in zip(first, second)),
            "discordants": mcnemar,
            "n_exact_replays": sum(exact),
            "exact_replay_fraction": statistics.fmean(exact),
        }
    return output


def analyze(
    paths: list[Path], *, bootstrap_samples: int = 20_000, bootstrap_seed: int = 20_260_821
) -> dict[str, Any]:
    validated_shards = [validate_artifact(path) for path in paths]
    validated = _merge_validated_task_shards(validated_shards)
    _cross_validate(validated)
    per_checkpoint: list[dict[str, Any]] = []
    for item in validated:
        blocks = item["blocks"]
        contrasts = {control: _contrast(blocks, control) for control in PRIMARY_CONTROLS}
        holm = holm_adjust(
            {control: contrasts[control]["discordants"]["exact_two_sided_p"] for control in PRIMARY_CONTROLS}
        )
        for control in PRIMARY_CONTROLS:
            contrasts[control]["holm_across_two_primary_tests"] = holm[control]
        taskwise = {
            str(task_id): {
                control: _contrast([block for block in blocks if block["task_id"] == task_id], control)
                for control in PRIMARY_CONTROLS
            }
            for task_id in item["artifact"]["task_ids"]
        }
        per_checkpoint.append(
            {
                "input_path": item["path"],
                "input_sha256": item["artifact_sha256"],
                "input_paths": item["input_paths"],
                "input_sha256s": item["input_sha256s"],
                "checkpoint": item["artifact"]["checkpoint"],
                "model_sha256": item["model_sha256"],
                "n_blocks": len(blocks),
                "primary_contrasts": contrasts,
                "per_task_contrasts": taskwise,
                "period_and_order_sensitivity": _period_order_summary(blocks),
                "predicted_donor_eval_closure_repeatability": _predicted_repeatability(blocks),
                "exact_predicted_replay": item["exact_predicted_replay"],
                "shuffle_strength": item["shuffle_strength"],
                "shuffle_strength_by_task": item["shuffle_strength_by_task"],
            }
        )

    single_seed_bootstrap: dict[str, Any]
    if len(validated) == 1:
        single_seed_bootstrap = {
            "available": True,
            "scope": "exploratory single-checkpoint pilot; not training-seed uncertainty",
            "contrasts": {
                control: hierarchical_task_state_bootstrap(
                    validated[0]["blocks"], control, samples=bootstrap_samples, seed=bootstrap_seed + index
                )
                for index, control in enumerate(PRIMARY_CONTROLS)
            },
        }
    else:
        single_seed_bootstrap = {
            "available": False,
            "reason": "single-checkpoint task/state bootstrap is emitted only for exactly one input",
        }

    training_seed = {
        control: _training_seed_summary(per_checkpoint, control)
        for control in PRIMARY_CONTROLS
    }
    reference = validated[0]["artifact"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "generated_from_validated_inputs_only": True,
        "claim_status": (
            "exploratory_single_checkpoint_only"
            if len(validated) == 1
            else (
                "multi_checkpoint_with_training_seed_intervals"
                if len(validated) >= 3
                else "exploratory_two_checkpoint_replication_no_seed_ci"
            )
        ),
        "claim_warning": (
            "A duration mechanism claim requires at least three independently "
            "trained checkpoints with a consistent predicted advantage over "
            "both controls; block-level p-values and task/state bootstrap do "
            "not substitute for training-seed replication."
        ),
        "n_independent_checkpoints": len(validated),
        "n_input_artifacts": len(validated_shards),
        "all_exact_predicted_replay": all(item["exact_predicted_replay"] for item in validated),
        "exact_replay_required_by_all_inputs": all(item["artifact"].get("exact_predicted_replay_required") for item in validated),
        "validated_shared_identity": {
            "suite": reference["suite"],
            "task_ids": reference["task_ids"],
            "state_ids": reference["state_ids"],
            "repeats": reference["repeats"],
            "seed_base": reference["seed_base"],
            "order_seed": reference["order_seed"],
            "control_frequency_hz": reference["control_frequency_hz"],
            "checkpoint_config_sha256": validated[0]["checkpoint_config_sha256"],
            "fixed_manifest_sha256": validated[0]["fixed_manifest_sha256"],
            "task_order_manifest_sha256_by_task": validated[0]["task_manifest_hashes"],
            "source_identity": validated[0]["source_identity"],
            "mujoco_gl_values": validated[0]["mujoco_gl_values"],
        },
        "per_checkpoint": per_checkpoint,
        "single_checkpoint_hierarchical_bootstrap": single_seed_bootstrap,
        "training_seed_inference": training_seed,
        "statistical_contract": {
            "primary_tests": ["predicted_eval_vs_fixed", "predicted_eval_vs_shuffled"],
            "binary_test": "exact two-sided McNemar on paired task/state/repeat blocks",
            "multiple_testing": "Holm across the two suite-level primary tests within each checkpoint",
            "bootstrap": "task then state resampling; all within-state repeats retained as one block cluster",
            "period_order": "descriptive sensitivity only",
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Duration mechanism v3 — claim-safe analysis",
        "",
        f"Status: **{result['claim_status']}**.",
        "",
        f"> {result['claim_warning']}",
        "",
        f"Validated checkpoints: {result['n_independent_checkpoints']}; exact predicted replay: "
        f"{result['all_exact_predicted_replay']}; renderer(s): "
        f"{', '.join(result['validated_shared_identity']['mujoco_gl_values'])}.",
        "",
    ]
    for checkpoint_index, checkpoint in enumerate(result["per_checkpoint"], start=1):
        lines.extend(
            [
                f"## Checkpoint {checkpoint_index}",
                "",
                f"Model `{checkpoint['model_sha256'][:12]}…`; paired blocks: {checkpoint['n_blocks']}; "
                f"shuffle changed-call fraction: {checkpoint['shuffle_strength']:.3f}.",
                "",
                "| primary contrast | predicted | control | delta (pp) | discordants P-only / C-only | exact p | Holm p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for control in PRIMARY_CONTROLS:
            cell = checkpoint["primary_contrasts"][control]
            disc = cell["discordants"]
            holm = cell["holm_across_two_primary_tests"]
            lines.append(
                f"| predicted_eval vs {control} | {cell['predicted_success_rate']:.3f} | "
                f"{cell['control_success_rate']:.3f} | {cell['delta_predicted_minus_control_pp']:+.1f} | "
                f"{disc['first_success_second_failure']} / {disc['first_failure_second_success']} | "
                f"{disc['exact_two_sided_p']:.4g} | {holm['holm_adjusted_p']:.4g} |"
            )
        lines.extend(["", "Per-task deltas (predicted minus control, pp):", ""])
        lines.append("| task | fixed | shuffled |")
        lines.append("|---:|---:|---:|")
        for task_id, cells in checkpoint["per_task_contrasts"].items():
            lines.append(
                f"| {task_id} | {cells['fixed']['delta_predicted_minus_control_pp']:+.1f} | "
                f"{cells['shuffled']['delta_predicted_minus_control_pp']:+.1f} |"
            )
        lines.extend(["", "Predicted-position replay diagnostics:", ""])
        lines.append("| pair | outcome agreement | exact replay | McNemar p |")
        lines.append("|---|---:|---:|---:|")
        for name, cell in checkpoint["predicted_donor_eval_closure_repeatability"].items():
            lines.append(
                f"| {name} | {cell['outcome_agreement_rate']:.3f} | "
                f"{cell['exact_replay_fraction']:.3f} | {cell['discordants']['exact_two_sided_p']:.4g} |"
            )
        lines.extend(
            [
                "",
                "Period/order sensitivity is descriptive. Success-rate ranges across efficacy periods (pp): "
                + ", ".join(
                    f"{arm}={value:.1f}"
                    for arm, value in checkpoint["period_and_order_sensitivity"]["success_rate_range_across_periods_pp"].items()
                )
                + ".",
                "",
            ]
        )
    bootstrap = result["single_checkpoint_hierarchical_bootstrap"]
    if bootstrap["available"]:
        lines.extend(["## Exploratory hierarchical uncertainty", ""])
        for control, cell in bootstrap["contrasts"].items():
            low, high = cell["percentile_95_ci_pp"]
            lines.append(
                f"- Predicted vs {control}: {cell['estimate_pp']:+.1f} pp, task/state "
                f"bootstrap 95% interval [{low:+.1f}, {high:+.1f}] pp."
            )
        lines.extend(["", "This interval measures task/state sampling uncertainty for one checkpoint, not training-seed uncertainty.", ""])
    lines.extend(["## Training-seed inference", ""])
    for control, cell in result["training_seed_inference"].items():
        if cell["available"]:
            low, high = cell["95_ci_pp"]
            lines.append(f"- Predicted vs {control}: mean {cell['mean_effect_pp']:+.1f} pp, 95% t interval [{low:+.1f}, {high:+.1f}] pp across {cell['n_independent_checkpoints']} checkpoints.")
        else:
            lines.append(f"- Predicted vs {control}: unavailable ({cell['reason']}).")
    lines.extend(["", "The companion JSON contains full validation identities, period/order strata, discordants, and per-task results.", ""])
    return "\n".join(lines)


def _atomic_write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another process created
        # the destination after our preflight check.
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_outputs(result: dict[str, Any], out_prefix: Path) -> tuple[Path, Path]:
    json_path = Path(f"{out_prefix}.json")
    markdown_path = Path(f"{out_prefix}.md")
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"refusing partial/complete overwrite: {json_path}, {markdown_path}"
        )
    json_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    markdown_bytes = render_markdown(result).encode("utf-8")
    _atomic_write_new(json_path, json_bytes)
    try:
        _atomic_write_new(markdown_path, markdown_bytes)
    except Exception:
        # Roll back our just-published JSON so a pair is never left half-written.
        json_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="v3 combined JSON artifact(s)")
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_821)
    args = parser.parse_args()
    result = analyze(
        args.inputs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    json_path, markdown_path = write_outputs(result, args.out_prefix)
    print(f"wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
