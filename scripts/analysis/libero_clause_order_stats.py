#!/usr/bin/env python3
"""Fail-closed analysis for paired LIBERO clause-order steering runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

GATE_PROTOCOL = "libero_clause_order_exact_replay_gate_v1"
RESULT_PROTOCOL = "libero_t04_paired_clause_order_predicate_oracle_v1"
TASKS = (0, 4)
CONDITIONS = ("normal", "reversed")
NONINFERIORITY_MARGIN = -0.10


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def exact_binomial_two_sided(successes: int, trials: int, null: float = 0.5) -> float:
    """Probability-ordering exact two-sided binomial p-value."""

    if not 0 <= successes <= trials or trials < 0 or not 0 < null < 1:
        raise ValueError("invalid binomial parameters")
    if trials == 0:
        return 1.0
    probabilities = [
        math.comb(trials, value) * null**value * (1 - null) ** (trials - value)
        for value in range(trials + 1)
    ]
    observed = probabilities[successes]
    return min(1.0, sum(value for value in probabilities if value <= observed + 1e-15))


def _load_passed_gate(path: Path, expected_states: int) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = json.loads(path.read_text())
    if gate.get("protocol") != GATE_PROTOCOL or gate.get("passed") is not True:
        raise ValueError(f"not a passed clause-order replay gate: {path}")
    if gate.get("failures"):
        raise ValueError(f"gate contains replay failures: {path}")
    primary = Path(gate["primary_result"])
    replay = Path(gate["replay_result"])
    if _sha256(primary) != gate.get("primary_result_sha256"):
        raise ValueError(f"primary hash mismatch: {path}")
    if _sha256(replay) != gate.get("replay_result_sha256"):
        raise ValueError(f"replay hash mismatch: {path}")
    result = json.loads(primary.read_text())
    replica = json.loads(replay.read_text())
    required = {
        "protocol": RESULT_PROTOCOL,
        "renderer_backend": "osmesa",
        "suite": "libero_10",
        "task_ids": [0, 4],
        "state_ids": list(range(expected_states)),
        "seed_base": 100000,
        "control_frequency_hz": 20,
        "n_action_steps": 10,
        "transition_oracle_is_privileged": True,
    }
    for key, expected in required.items():
        if result.get(key) != expected or replica.get(key) != expected:
            raise ValueError(f"{path}: {key} does not equal {expected!r}")
    if result.get("model_sha256") != replica.get("model_sha256"):
        raise ValueError(f"model identity differs across replay: {path}")
    if result.get("checkpoint_config_sha256") != replica.get("checkpoint_config_sha256"):
        raise ValueError(f"config identity differs across replay: {path}")
    expected_rows = expected_states * len(TASKS) * len(CONDITIONS)
    if result.get("n_episodes") != expected_rows or len(result.get("episodes", [])) != expected_rows:
        raise ValueError(f"incomplete episode grid: {path}")
    observed_grid = {
        (int(row["task_id"]), int(row["state_id"]), row["condition"])
        for row in result["episodes"]
    }
    expected_grid = {
        (task_id, state_id, condition)
        for task_id in TASKS
        for state_id in range(expected_states)
        for condition in CONDITIONS
    }
    if observed_grid != expected_grid or len(observed_grid) != expected_rows:
        raise ValueError(f"duplicate or incomplete task/state/condition grid: {path}")
    return result, {
        "gate": str(path),
        "gate_sha256": _sha256(path),
        "primary": str(primary),
        "primary_sha256": _sha256(primary),
        "replay": str(replay),
        "replay_sha256": _sha256(replay),
        "model_sha256": result["model_sha256"],
        "checkpoint_config_sha256": result["checkpoint_config_sha256"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_coordinate: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        coordinate = (int(row["task_id"]), int(row["state_id"]))
        condition = row["condition"]
        if condition not in CONDITIONS or condition in by_coordinate[coordinate]:
            raise ValueError(f"duplicate/invalid condition at {coordinate}: {condition}")
        by_coordinate[coordinate][condition] = row
    if any(set(pair) != set(CONDITIONS) for pair in by_coordinate.values()):
        raise ValueError("normal/reversed pairing is incomplete")

    requested_flips = 0
    determinate_both = 0
    directional_discordant = 0
    antidirectional_discordant = 0
    normal_successes = reversed_successes = 0
    normal_only_success = reversed_only_success = 0
    no_unique_first = 0
    for pair in by_coordinate.values():
        normal, reversed_order = pair["normal"], pair["reversed"]
        normal_first = normal["first_completed_goal_indices"]
        reversed_first = reversed_order["first_completed_goal_indices"]
        if len(normal_first) == 1 and len(reversed_first) == 1:
            determinate_both += 1
            if normal_first == [0] and reversed_first == [1]:
                requested_flips += 1
                directional_discordant += 1
            elif normal_first == [1] and reversed_first == [0]:
                antidirectional_discordant += 1
        else:
            no_unique_first += 1
        normal_success = bool(normal["success"])
        reversed_success = bool(reversed_order["success"])
        normal_successes += normal_success
        reversed_successes += reversed_success
        normal_only_success += normal_success and not reversed_success
        reversed_only_success += reversed_success and not normal_success

    n = len(by_coordinate)
    discordant = directional_discordant + antidirectional_discordant
    success_difference = (reversed_successes - normal_successes) / n
    return {
        "n_paired_states": n,
        "requested_first_flip": {
            "count": requested_flips,
            "rate_all_states": requested_flips / n,
            "rate_among_both_unique": (
                requested_flips / determinate_both if determinate_both else None
            ),
            "both_conditions_unique_first": determinate_both,
            "at_least_one_tie_or_no_completion": no_unique_first,
        },
        "paired_first_goal_mcnemar": {
            "directional_normal0_reversed1": directional_discordant,
            "antidirectional_normal1_reversed0": antidirectional_discordant,
            "discordant_pairs": discordant,
            "null": "direction is equiprobable among discordant unique-first pairs",
            "exact_two_sided_binomial_mcnemar_pvalue": exact_binomial_two_sided(
                directional_discordant, discordant
            ),
        },
        "final_success_descriptive_noninferiority": {
            "normal": normal_successes,
            "reversed": reversed_successes,
            "normal_rate": normal_successes / n,
            "reversed_rate": reversed_successes / n,
            "reversed_minus_normal": success_difference,
            "normal_only": normal_only_success,
            "reversed_only": reversed_only_success,
            "descriptive_margin": NONINFERIORITY_MARGIN,
            "observed_difference_above_margin": success_difference >= NONINFERIORITY_MARGIN,
            "formal_noninferiority_test_performed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected_states_per_task", type=int, default=50)
    parser.add_argument("--expected_model_sha256")
    parser.add_argument("--expected_config_sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.expected_states_per_task <= 0:
        raise ValueError("expected state count must be positive")
    result, provenance = _load_passed_gate(args.gate, args.expected_states_per_task)
    if args.expected_model_sha256 and result["model_sha256"] != args.expected_model_sha256:
        raise ValueError("model hash does not match preregistered checkpoint")
    if args.expected_config_sha256 and result["checkpoint_config_sha256"] != args.expected_config_sha256:
        raise ValueError("isolated n_action_steps=10 config hash does not match expected hash")

    per_task = {
        f"t{task_id}": _summarize(
            [row for row in result["episodes"] if int(row["task_id"]) == task_id]
        )
        for task_id in TASKS
    }
    payload = {
        "schema_version": 1,
        "protocol": "libero_clause_order_preregistered_analysis_v1",
        "claim_ready": True,
        "estimand": (
            "normal-first-goal-0/reversed-first-goal-1 directional order flip "
            "under an official-predicate transition oracle"
        ),
        "privileged_transition_oracle": True,
        "noninferiority_is_descriptive": True,
        "overall": _summarize(result["episodes"]),
        "per_task": per_task,
        "input_provenance": provenance,
    }
    _atomic_write_json_new(args.out, payload)
    print(json.dumps({"claim_ready": True, "overall": payload["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
