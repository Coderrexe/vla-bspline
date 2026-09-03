"""Dependency-light contracts for stochastic paired duration blocks."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from typing import Any


WATERMARK = (
    "STOCHASTIC_PAIRED_BLOCK_RESULT: trajectories are not bitwise replay-"
    "deterministic; episodes are paired blocks, not independent replicates"
)
BLOCK_SEED_PROTOCOL = (
    "sha256(duration_stochastic_v3,base,task,state,repeat): "
    "first u32 env seed, next u64 policy seed"
)
COUNTERBALANCE_PROTOCOL = (
    "preregistered seeded random assignment with exact per-task quotas over all "
    "six permutations of (predicted_eval,fixed,shuffled)"
)
OBSERVATION_COMPONENT_PROTOCOL = (
    "observation_component_merkle_sha256_v1: leaves use the evaluator's "
    "typed structured hash; mapping nodes hash sorted child-name/digest pairs"
)
EFFICACY_PERMUTATIONS = (
    ("predicted_eval", "fixed", "shuffled"),
    ("predicted_eval", "shuffled", "fixed"),
    ("fixed", "predicted_eval", "shuffled"),
    ("fixed", "shuffled", "predicted_eval"),
    ("shuffled", "predicted_eval", "fixed"),
    ("shuffled", "fixed", "predicted_eval"),
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _json_pointer_token(value: str) -> str:
    """Escape a mapping key using RFC 6901 JSON-pointer rules."""

    return value.replace("~", "~0").replace("/", "~1")


def observation_component_hashes(
    value: Any, leaf_hasher: Any
) -> dict[str, str]:
    """Hash every observation component without retaining observation arrays.

    Leaves use the evaluator's exact typed hash. Mapping-node hashes are Merkle
    hashes over their sorted child names and hashes, so large camera arrays are
    read only once per component trace instead of once for every ancestor. The
    root hash remains the evaluator's existing whole-observation hash and is
    intentionally not duplicated here.
    """

    components: dict[str, str] = {}

    def validate_digest(digest: Any) -> str:
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("component leaf hasher must return a SHA-256 hex digest")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError(
                "component leaf hasher must return a SHA-256 hex digest"
            ) from error
        return digest

    def visit(node: Any, path: str) -> str:
        if isinstance(node, dict):
            if not all(isinstance(key, str) for key in node):
                raise TypeError("observation component mappings require string keys")
            children: list[list[str]] = []
            for key in sorted(node):
                child_path = f"{path}/{_json_pointer_token(key)}"
                child_digest = visit(node[key], child_path)
                components[child_path] = child_digest
                children.append([key, child_digest])
            return canonical_json_sha256(
                {
                    "protocol": OBSERVATION_COMPONENT_PROTOCOL,
                    "node_type": "mapping",
                    "children": children,
                }
            )
        return validate_digest(leaf_hasher(node))

    root_digest = visit(value, "")
    if not isinstance(value, dict):
        # A non-mapping observation has one explicit root component.
        components["/"] = root_digest
    elif not value:
        # Preserve an explicit signal for an empty root mapping.
        components["/"] = root_digest
    return components


def component_first_differences(
    first: dict[str, list[str]], second: dict[str, list[str]]
) -> dict[str, int | None]:
    """Return each component's first differing step, including schema drift."""

    paths = sorted(set(first) | set(second))
    differences: dict[str, int | None] = {}
    for path in paths:
        if path not in first or path not in second:
            differences[path] = 0
        else:
            differences[path] = first_difference(first[path], second[path])
    return differences


def validate_embedded_sha256(
    payload: dict[str, Any], field: str = "manifest_sha256"
) -> None:
    reported = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    expected = canonical_json_sha256(unhashed)
    if reported != expected:
        raise ValueError(
            f"embedded {field} mismatch: expected {expected}, found {reported}"
        )


def block_seeds(
    seed_base: int, task_id: int, state_id: int, repeat_index: int
) -> tuple[int, int]:
    values = (seed_base, task_id, state_id, repeat_index)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("block seed coordinates must be integers")
    if any(value < 0 for value in values):
        raise ValueError("block seed coordinates must be nonnegative")
    payload = (
        f"duration_stochastic_v3|{seed_base}|{task_id}|{state_id}|{repeat_index}"
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    env_seed = int.from_bytes(digest[:4], "little")
    policy_seed = int.from_bytes(digest[4:12], "little")
    return env_seed, policy_seed


def materialize_task_order_manifest(
    *,
    suite: str,
    task_id: int,
    state_ids: list[int],
    repeats: int,
    seed_base: int,
    order_seed: int,
) -> dict[str, Any]:
    if not state_ids or len(state_ids) != len(set(state_ids)):
        raise ValueError("state_ids must be nonempty and duplicate-free")
    scalar_values = (task_id, seed_base, order_seed)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in scalar_values
    ):
        raise TypeError("task_id, seed_base, and order_seed must be integers")
    if any(value < 0 for value in scalar_values):
        raise ValueError("task_id, seed_base, and order_seed must be nonnegative")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in state_ids
    ):
        raise ValueError("state_ids must contain nonnegative integers")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    coordinates = [
        (state_id, repeat_index)
        for state_id in state_ids
        for repeat_index in range(repeats)
    ]
    quotient, remainder = divmod(len(coordinates), len(EFFICACY_PERMUTATIONS))
    randomization_payload = (
        f"duration_order_v3|{order_seed}|{suite}|{task_id}|"
        f"{','.join(str(value) for value in state_ids)}|{repeats}"
    ).encode("ascii")
    rng = random.Random(
        int.from_bytes(hashlib.sha256(randomization_payload).digest()[:8], "little")
    )
    extra_indices = list(range(len(EFFICACY_PERMUTATIONS)))
    rng.shuffle(extra_indices)
    permutation_indices = [
        index
        for index in range(len(EFFICACY_PERMUTATIONS))
        for _ in range(quotient)
    ] + extra_indices[:remainder]
    rng.shuffle(permutation_indices)

    blocks: list[dict[str, Any]] = []
    counts = {"/".join(order): 0 for order in EFFICACY_PERMUTATIONS}
    for task_block_ordinal, ((state_id, repeat_index), permutation_index) in enumerate(
        zip(coordinates, permutation_indices, strict=True)
    ):
        env_seed, policy_seed = block_seeds(
            seed_base, task_id, state_id, repeat_index
        )
        order = EFFICACY_PERMUTATIONS[permutation_index]
        counts["/".join(order)] += 1
        blocks.append(
            {
                "task_block_ordinal": task_block_ordinal,
                "state_id": state_id,
                "repeat_index": repeat_index,
                "env_seed_u32": env_seed,
                "policy_seed_u64": policy_seed,
                "efficacy_permutation_index": permutation_index,
                "efficacy_permutation": list(order),
                "arm_order": [
                    "predicted_donor",
                    *order,
                    "predicted_closure",
                ],
            }
        )
    values = list(counts.values())
    if max(values) - min(values) > 1:
        raise AssertionError("per-task permutation quotas are not balanced")
    manifest = {
        "schema_version": 1,
        "protocol": "task_local_preregistered_permutation_manifest_v3",
        "counterbalance_protocol": COUNTERBALANCE_PROTOCOL,
        "suite": suite,
        "task_id": task_id,
        "state_ids": state_ids,
        "repeats": repeats,
        "seed_base": seed_base,
        "order_seed": order_seed,
        "permutation_counts": counts,
        "blocks": blocks,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    validate_embedded_sha256(manifest)
    return manifest


def first_difference(left: list[Any], right: list[Any]) -> int | None:
    for index, (first, second) in enumerate(zip(left, right)):
        if first != second:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def predicted_pair_diagnostic(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Summarize any two predicted-position rollouts as drift telemetry."""

    first_success = bool(first["success"])
    second_success = bool(second["success"])
    action_difference = first_difference(
        first["executed_action_step_sha256"],
        second["executed_action_step_sha256"],
    )
    prefix_difference = first_difference(
        first["executed_action_prefix_sha256"],
        second["executed_action_prefix_sha256"],
    )
    duration_difference = first_difference(
        first["predicted_durations"], second["predicted_durations"]
    )
    raw_observation_difference = first_difference(
        first["raw_observation_step_sha256"],
        second["raw_observation_step_sha256"],
    )
    processed_observation_difference = first_difference(
        first["processed_observation_step_sha256"],
        second["processed_observation_step_sha256"],
    )
    replan_input_difference = first_difference(
        first["policy_replan_inputs"], second["policy_replan_inputs"]
    )
    simulator_state_difference = first_difference(
        first["mujoco_integration_state_step_sha256"],
        second["mujoco_integration_state_step_sha256"],
    )
    raw_component_differences = component_first_differences(
        first["raw_observation_component_step_sha256"],
        second["raw_observation_component_step_sha256"],
    )
    processed_component_differences = component_first_differences(
        first["processed_observation_component_step_sha256"],
        second["processed_observation_component_step_sha256"],
    )
    return {
        "outcome_agreement": first_success == second_success,
        "first_success": first_success,
        "second_success": second_success,
        "outcome_pair": f"{int(first_success)}{int(second_success)}",
        "step_count_agreement": first["steps"] == second["steps"],
        "first_steps": int(first["steps"]),
        "second_steps": int(second["steps"]),
        "raw_initial_observation_agreement": (
            first["initial_raw_observation_sha256"]
            == second["initial_raw_observation_sha256"]
        ),
        "processed_initial_observation_agreement": (
            first["initial_processed_observation_sha256"]
            == second["initial_processed_observation_sha256"]
        ),
        "mujoco_integration_initial_state_agreement": (
            first["initial_mujoco_integration_state_sha256"]
            == second["initial_mujoco_integration_state_sha256"]
        ),
        "first_different_raw_observation_step": raw_observation_difference,
        "first_different_processed_observation_step": (
            processed_observation_difference
        ),
        "first_different_raw_observation_step_by_component": (
            raw_component_differences
        ),
        "first_different_processed_observation_step_by_component": (
            processed_component_differences
        ),
        "first_different_policy_replan_input": replan_input_difference,
        "first_different_mujoco_integration_state_step": simulator_state_difference,
        "exact_action_trace_agreement": (
            first["executed_action_trace_sha256"]
            == second["executed_action_trace_sha256"]
        ),
        "first_different_action_step": action_difference,
        "first_different_action_prefix": prefix_difference,
        "exact_predicted_duration_trace_agreement": duration_difference is None,
        "first_different_predicted_duration_chunk": duration_difference,
        "first_duration_calls": int(first["n_duration_calls"]),
        "second_duration_calls": int(second["n_duration_calls"]),
    }


def _divergence_summary(values: list[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "minimum": min(values) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "maximum": max(values) if values else None,
    }


def aggregate_predicted_pairs(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    if not diagnostics:
        raise ValueError("cannot aggregate an empty diagnostic list")
    pairs = {name: 0 for name in ("00", "01", "10", "11")}
    for diagnostic in diagnostics:
        pairs[diagnostic["outcome_pair"]] += 1
    action_divergences = [
        int(diagnostic["first_different_action_step"])
        for diagnostic in diagnostics
        if diagnostic["first_different_action_step"] is not None
    ]
    duration_divergences = [
        int(diagnostic["first_different_predicted_duration_chunk"])
        for diagnostic in diagnostics
        if diagnostic["first_different_predicted_duration_chunk"] is not None
    ]
    raw_observation_divergences = [
        int(diagnostic["first_different_raw_observation_step"])
        for diagnostic in diagnostics
        if diagnostic["first_different_raw_observation_step"] is not None
    ]
    processed_observation_divergences = [
        int(diagnostic["first_different_processed_observation_step"])
        for diagnostic in diagnostics
        if diagnostic["first_different_processed_observation_step"] is not None
    ]
    replan_input_divergences = [
        int(diagnostic["first_different_policy_replan_input"])
        for diagnostic in diagnostics
        if diagnostic["first_different_policy_replan_input"] is not None
    ]
    simulator_state_divergences = [
        int(diagnostic["first_different_mujoco_integration_state_step"])
        for diagnostic in diagnostics
        if diagnostic["first_different_mujoco_integration_state_step"] is not None
    ]
    agreements = sum(diagnostic["outcome_agreement"] for diagnostic in diagnostics)
    return {
        "n_blocks": len(diagnostics),
        "n_outcome_agreements": agreements,
        "outcome_agreement_rate": agreements / len(diagnostics),
        "outcome_pairs": pairs,
        "n_exact_action_trace_agreements": sum(
            diagnostic["exact_action_trace_agreement"]
            for diagnostic in diagnostics
        ),
        "n_exact_predicted_duration_trace_agreements": sum(
            diagnostic["exact_predicted_duration_trace_agreement"]
            for diagnostic in diagnostics
        ),
        "n_raw_initial_observation_agreements": sum(
            diagnostic["raw_initial_observation_agreement"]
            for diagnostic in diagnostics
        ),
        "n_processed_initial_observation_agreements": sum(
            diagnostic["processed_initial_observation_agreement"]
            for diagnostic in diagnostics
        ),
        "n_mujoco_integration_initial_state_agreements": sum(
            diagnostic["mujoco_integration_initial_state_agreement"]
            for diagnostic in diagnostics
        ),
        "first_action_divergence_step": _divergence_summary(action_divergences),
        "first_raw_observation_divergence_step": _divergence_summary(
            raw_observation_divergences
        ),
        "first_processed_observation_divergence_step": _divergence_summary(
            processed_observation_divergences
        ),
        "first_replan_input_divergence_index": _divergence_summary(
            replan_input_divergences
        ),
        "first_mujoco_integration_state_divergence_step": _divergence_summary(
            simulator_state_divergences
        ),
        "first_duration_divergence_chunk": _divergence_summary(
            duration_divergences
        ),
    }
