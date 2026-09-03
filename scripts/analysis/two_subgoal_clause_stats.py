#!/usr/bin/env python3
"""Strict, no-claim aggregation for the locked two-subgoal clause study.

Inputs are only the *passed* v2 OSMesa replay gates.  The program refuses an
incomplete 3-seed 2x2 result by default, preventing an accidental conversion
of the seed-1000 screen into an aggregate claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    (arm, treatment, mode)
    for arm, mode in (("A", "fixed_clock"), ("C", "event_clock"))
    for treatment in ("original", "clause")
} | {
    (arm, treatment, "compound")
    for arm in ("A", "C")
    for treatment in ("original", "clause")
} | {
    ("C", treatment, "fixed_clock")
    for treatment in ("original", "clause")
}
EXPECTED_DESCRIPTIONS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar test, conditioned on discordance."""

    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    numerator = sum(math.comb(n, value) for value in range(k + 1))
    return min(1.0, 2.0 * numerator / (2**n))


def episode_index(payload: dict[str, Any]) -> dict[tuple[int, int], bool]:
    indexed: dict[tuple[int, int], bool] = {}
    for row in payload["episodes"]:
        key = (int(row["task_id"]), int(row["state_id"]))
        if key in indexed:
            raise ValueError(f"duplicate episode coordinate {key}")
        indexed[key] = bool(row["success"])
    return indexed


def load_gate(path: Path) -> tuple[tuple[str, str, int, str], dict[tuple[int, int], bool], dict[str, Any]]:
    gate = json.loads(path.read_text())
    if gate.get("protocol") != "two_subgoal_t04_osmesa_exact_replay_gate_v2" or not gate.get("passed"):
        raise ValueError(f"not a passed v2 replay gate: {path}")
    primary = Path(gate["primary_result"])
    if sha256(primary) != gate.get("primary_result_sha256"):
        raise ValueError(f"primary result hash mismatch: {path}")
    result = json.loads(primary.read_text())
    required = {
        "protocol": "libero_renderer_comparability_probe_v1",
        "renderer_backend": "osmesa",
        "task_ids": [0, 4],
        "state_ids": list(range(50)),
        "seed_base": 100000,
        "control_frequency_hz": 20,
        "n_action_steps": 10,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"{path}: {key}={result.get(key)!r}, expected {expected!r}")
    if result["n_episodes"] != 100 or result["n_episodes"] != len(result["episodes"]):
        raise ValueError(f"{path}: incomplete t0/t4 grid")
    for row in result["episodes"]:
        if row.get("task_description") != EXPECTED_DESCRIPTIONS.get(row.get("task_id")):
            raise ValueError(f"{path}: task text mismatch")
    manifest = json.loads(Path(gate["training_manifest"]).read_text())
    if sha256(Path(gate["training_manifest"])) != gate.get("training_manifest_sha256"):
        raise ValueError(f"training manifest hash mismatch: {path}")
    if not (manifest["steps"] == 7500 and manifest["batch_size"] == 32 and manifest["episode_count"] == 71 and manifest["frame_count"] == 19378 and manifest["hardware_gate"] == "H100"):
        raise ValueError(f"{path}: training protocol mismatch")
    key = (manifest["arm"], manifest["treatment"], int(manifest["seed"]), result["language_mode"])
    if (key[0], key[1], key[3]) not in EXPECTED:
        raise ValueError(f"unexpected factorial cell: {key}")
    return key, episode_index(result), {"gate": str(path), "result": str(primary), "manifest": gate["training_manifest"]}


def comparison(original: dict[int, dict[tuple[int, int], bool]], clause: dict[int, dict[tuple[int, int], bool]]) -> dict[str, Any]:
    seeds = sorted(set(original) & set(clause))
    if len(seeds) != 3:
        raise ValueError(f"need exactly three paired seeds, got {seeds}")
    per_seed, pooled, by_seed_effects = [], [], []
    for seed in seeds:
        if set(original[seed]) != set(clause[seed]):
            raise ValueError(f"seed {seed}: initial-state grid differs")
        rows = [(original[seed][key], clause[seed][key]) for key in sorted(original[seed])]
        b = sum(left and not right for left, right in rows)
        c = sum(right and not left for left, right in rows)
        per_seed.append({"seed": seed, "n": len(rows), "original_successes": sum(left for left, _ in rows), "clause_successes": sum(right for _, right in rows), "discordant_original_only": b, "discordant_clause_only": c, "clause_minus_original": (c - b) / len(rows), "exact_mcnemar_pvalue": exact_mcnemar_pvalue(b, c)})
        pooled.extend(rows)
        by_seed_effects.append(np.asarray([int(right) - int(left) for left, right in rows]))
    b = sum(left and not right for left, right in pooled)
    c = sum(right and not left for left, right in pooled)
    seed_means = np.asarray([values.mean() for values in by_seed_effects], dtype=np.float64)
    seed_mean = float(seed_means.mean())
    seed_half_width = 4.302652729911275 * float(seed_means.std(ddof=1)) / math.sqrt(3)
    rng = np.random.default_rng(20_260_821)
    draws = np.empty(20_000)
    for draw in range(len(draws)):
        # Preserve the three-seed design and each seed's fixed t0/t4 balance.
        draws[draw] = float(np.mean([rng.choice(values, size=len(values)).mean() for values in by_seed_effects]))
    return {"n_seeds": len(seeds), "n_paired_states": len(pooled), "per_seed": per_seed, "training_seed_mean_effect": seed_mean, "training_seed_t_95_ci_df2": [seed_mean - seed_half_width, seed_mean + seed_half_width], "pooled": {"original_successes": sum(left for left, _ in pooled), "clause_successes": sum(right for _, right in pooled), "discordant_original_only": b, "discordant_clause_only": c, "clause_minus_original": (c - b) / len(pooled), "exact_mcnemar_pvalue": exact_mcnemar_pvalue(b, c), "paired_state_stratified_bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], "bootstrap_draws": len(draws)}}


def matched_clock_head_interaction(
    a_original: dict[int, dict[tuple[int, int], bool]],
    a_clause: dict[int, dict[tuple[int, int], bool]],
    c_original: dict[int, dict[tuple[int, int], bool]],
    c_clause: dict[int, dict[tuple[int, int], bool]],
) -> dict[str, Any]:
    """Difference-in-differences with training seed as the inference unit.

    All four cells use the same fixed language clock. This is the primary test
    of whether clause supervision helps the spline head more than the waypoint
    head without confounding the head comparison by different schedulers.
    """

    seeds = sorted(set(a_original) & set(a_clause) & set(c_original) & set(c_clause))
    if seeds != [1000, 1001, 1002]:
        raise ValueError(f"matched-clock interaction requires seeds 1000–1002, got {seeds}")
    per_seed: list[dict[str, Any]] = []
    state_effects: dict[int, np.ndarray] = {}
    for seed in seeds:
        grids = [a_original[seed], a_clause[seed], c_original[seed], c_clause[seed]]
        keys = set(grids[0])
        if any(set(grid) != keys for grid in grids[1:]):
            raise ValueError(f"seed {seed}: matched-clock four-cell state grid differs")
        rows = np.asarray(
            [
                [int(a_original[seed][key]), int(a_clause[seed][key]),
                 int(c_original[seed][key]), int(c_clause[seed][key])]
                for key in sorted(keys)
            ],
            dtype=np.float64,
        )
        a_effect = rows[:, 1] - rows[:, 0]
        c_effect = rows[:, 3] - rows[:, 2]
        interaction = c_effect - a_effect
        state_effects[seed] = interaction
        per_seed.append({
            "seed": seed,
            "n_states": len(rows),
            "a_clause_minus_original": float(a_effect.mean()),
            "c_clause_minus_original": float(c_effect.mean()),
            "c_minus_a_language_interaction": float(interaction.mean()),
        })
    seed_values = np.asarray(
        [row["c_minus_a_language_interaction"] for row in per_seed], dtype=np.float64
    )
    mean = float(seed_values.mean())
    half_width = 4.302652729911275 * float(seed_values.std(ddof=1)) / math.sqrt(3)
    rng = np.random.default_rng(20_260_822)
    draws = np.empty(20_000, dtype=np.float64)
    for draw in range(len(draws)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        draws[draw] = float(np.mean([
            rng.choice(state_effects[int(seed)], size=len(state_effects[int(seed)]), replace=True).mean()
            for seed in sampled_seeds
        ]))
    return {
        "estimand": "(C_clause-C_original)-(A_clause-A_original), all at fixed clock",
        "n_training_seeds": 3,
        "per_seed": per_seed,
        "mean_interaction": mean,
        "training_seed_t_95_ci_df2": [mean - half_width, mean + half_width],
        "hierarchical_seed_state_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        ],
        "bootstrap_draws": len(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    loaded: dict[tuple[str, str, int, str], dict[tuple[int, int], bool]] = {}
    provenance = []
    for path in args.gate:
        key, episodes, source = load_gate(path)
        if key in loaded:
            raise ValueError(f"duplicate gate for {key}")
        loaded[key] = episodes
        provenance.append(source)
    expected = {(arm, treatment, seed, mode) for arm, treatment, mode in EXPECTED for seed in (1000, 1001, 1002)}
    if set(loaded) != expected:
        missing, extra = sorted(expected - set(loaded)), sorted(set(loaded) - expected)
        raise ValueError(f"incomplete or invalid locked factorial; missing={missing}, extra={extra}")
    comparisons = {}
    for arm, mode in (("A", "fixed_clock"), ("C", "fixed_clock"), ("C", "event_clock"), ("A", "compound"), ("C", "compound")):
        original = {seed: loaded[(arm, "original", seed, mode)] for seed in (1000, 1001, 1002)}
        clause = {seed: loaded[(arm, "clause", seed, mode)] for seed in (1000, 1001, 1002)}
        comparisons[f"{arm}_{mode}"] = comparison(original, clause)
    interaction = matched_clock_head_interaction(
        {seed: loaded[("A", "original", seed, "fixed_clock")] for seed in (1000, 1001, 1002)},
        {seed: loaded[("A", "clause", seed, "fixed_clock")] for seed in (1000, 1001, 1002)},
        {seed: loaded[("C", "original", seed, "fixed_clock")] for seed in (1000, 1001, 1002)},
        {seed: loaded[("C", "clause", seed, "fixed_clock")] for seed in (1000, 1001, 1002)},
    )
    payload = {"schema_version": 2, "protocol": "two_subgoal_clause_locked_3seed_analysis_v2", "claim_ready": True, "primary_matched_clock_head_interaction": interaction, "comparisons": comparisons, "input_provenance": provenance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"claim_ready": True, "comparisons": list(comparisons)}, sort_keys=True))


if __name__ == "__main__":
    main()
