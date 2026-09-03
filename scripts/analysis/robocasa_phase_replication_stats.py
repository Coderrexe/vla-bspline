#!/usr/bin/env python3
"""Validate and summarize the two-seed RoboCasa official-phase experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact


TASKS = ("KettleBoiling", "RinseSinkBasin")
TREATMENTS = ("original", "official")
EXPECTED_MODES = {
    "KettleBoiling": "kettle_official_fixed",
    "RinseSinkBasin": "rinse_official_fixed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def success(episode: dict[str, Any]) -> bool:
    return bool(episode["extrema"]["success"])


def progress_summary(task: str, episodes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if task == "KettleBoiling":
        predicates = {
            "ever_grasped": lambda x: bool(x["object_grasped"]),
            "lifted_at_least_5cm": lambda x: float(x["object_lift"]) >= 0.05,
            "ever_near_burner": lambda x: bool(x["kettle_near_burner"]),
            "ever_stove_contact": lambda x: bool(x["object_stove_contact"]),
            "ever_any_burner_on": lambda x: bool(x["any_burner_on"]),
        }
    elif task == "RinseSinkBasin":
        predicates = {
            "ever_water_on": lambda x: bool(x["water_on"]),
            "washed_at_least_one_region": lambda x: float(x["washed_count"]) >= 1,
            "washed_at_least_two_regions": lambda x: float(x["washed_count"]) >= 2,
        }
    else:
        raise ValueError(task)
    n = len(episodes)
    return {
        name: {
            "episodes": count,
            "n": n,
            "rate": count / n,
        }
        for name, predicate in predicates.items()
        for count in [sum(predicate(episode["extrema"]) for episode in episodes.values())]
    }


def load_shard(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = (root / spec["path"]).resolve()
    payload = json.loads(path.read_text())
    expected_seeds = set(range(int(spec["seed_start"]), int(spec["seed_stop"])))
    episodes = payload["episodes"]
    seeds = [int(episode["seed"]) for episode in episodes]
    if set(seeds) != expected_seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError(f"seed grid mismatch in {path}")
    if len(episodes) != payload["n_episodes"] or len(episodes) != len(expected_seeds):
        raise RuntimeError(f"support mismatch in {path}")
    if payload["task"] != spec["task"]:
        raise RuntimeError(f"task mismatch in {path}")
    if payload["instruction_mode"] != EXPECTED_MODES[spec["task"]]:
        raise RuntimeError(f"not a fixed-clock evaluation: {path}")
    if payload["environment_split"] != "target" or payload["renderer"] != "osmesa":
        raise RuntimeError(f"evaluation-domain mismatch in {path}")
    if sum(success(episode) for episode in episodes) != payload["successes"]:
        raise RuntimeError(f"success count mismatch in {path}")
    return payload, path


def exact_mcnemar(discordant_01: int, discordant_10: int) -> float:
    n = discordant_01 + discordant_10
    if n == 0:
        return 1.0
    tail = sum(
        math.comb(n, k) for k in range(min(discordant_01, discordant_10) + 1)
    ) / (2**n)
    return min(1.0, 2 * tail)


def summarize_pair(
    original: dict[int, dict[str, Any]],
    official: dict[int, dict[str, Any]],
    task: str | None = None,
) -> dict[str, Any]:
    if set(original) != set(official):
        raise RuntimeError("treatments do not share the same evaluation seed grid")
    seeds = sorted(original)
    original_successes = sum(success(original[seed]) for seed in seeds)
    official_successes = sum(success(official[seed]) for seed in seeds)
    exact_initial = [
        seed
        for seed in seeds
        if original[seed]["initial_observation_sha256"]
        == official[seed]["initial_observation_sha256"]
    ]
    contingency = {
        "both_fail": 0,
        "original_fail_official_success": 0,
        "original_success_official_fail": 0,
        "both_success": 0,
    }
    keys = list(contingency)
    for seed in exact_initial:
        index = 2 * int(success(original[seed])) + int(success(official[seed]))
        contingency[keys[index]] += 1
    result = {
        "n_per_treatment": len(seeds),
        "original": {
            "successes": original_successes,
            "rate": original_successes / len(seeds),
            "wilson_95": wilson(original_successes, len(seeds)),
        },
        "official": {
            "successes": official_successes,
            "rate": official_successes / len(seeds),
            "wilson_95": wilson(official_successes, len(seeds)),
        },
        "absolute_gain": (official_successes - original_successes) / len(seeds),
        "fisher_exact_two_sided_pvalue": fisher_exact(
            [
                [original_successes, len(seeds) - original_successes],
                [official_successes, len(seeds) - official_successes],
            ],
            alternative="two-sided",
        ).pvalue,
        "identical_initial_observation_pairs": len(exact_initial),
        "identical_initial_observation_fraction": len(exact_initial) / len(seeds),
        "identical_initial_subset_contingency": contingency,
        "identical_initial_subset_exact_mcnemar_pvalue": exact_mcnemar(
            contingency["original_fail_official_success"],
            contingency["original_success_official_fail"],
        ),
    }
    if task is not None:
        original_progress = progress_summary(task, original)
        official_progress = progress_summary(task, official)
        result["diagnostic_progress"] = {
            name: {
                "original": original_progress[name],
                "official": official_progress[name],
                "absolute_rate_gain": (
                    official_progress[name]["rate"] - original_progress[name]["rate"]
                ),
                "fisher_exact_two_sided_pvalue": fisher_exact(
                    [
                        [
                            original_progress[name]["episodes"],
                            original_progress[name]["n"]
                            - original_progress[name]["episodes"],
                        ],
                        [
                            official_progress[name]["episodes"],
                            official_progress[name]["n"]
                            - official_progress[name]["episodes"],
                        ],
                    ],
                    alternative="two-sided",
                ).pvalue,
            }
            for name in original_progress
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != "robocasa_phase_replication_inputs_v1":
        raise RuntimeError("unknown input manifest schema")

    cells: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    checkpoint_hashes: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    input_hashes: dict[str, str] = {}
    for spec in config["shards"]:
        seed = int(spec["training_seed"])
        task, treatment = spec["task"], spec["treatment"]
        if task not in TASKS or treatment not in TREATMENTS:
            raise RuntimeError(f"unknown cell: {spec}")
        payload, path = load_shard(args.root, spec)
        key = seed, task, treatment
        for episode in payload["episodes"]:
            episode_seed = int(episode["seed"])
            if episode_seed in cells[key]:
                raise RuntimeError(f"duplicate episode seed in {key}: {episode_seed}")
            cells[key][episode_seed] = episode
        checkpoint_hashes[key].add(payload["checkpoint_model_sha256"])
        input_hashes[str(path)] = sha256(path)

    training_seeds = sorted({key[0] for key in cells})
    expected = {
        (seed, task, treatment)
        for seed in training_seeds
        for task in TASKS
        for treatment in TREATMENTS
    }
    if set(cells) != expected:
        raise RuntimeError(f"incomplete factorial: missing={sorted(expected - set(cells))}")
    if any(len(hashes) != 1 for hashes in checkpoint_hashes.values()):
        raise RuntimeError("a cell mixes checkpoint weights across shards")

    by_seed: dict[str, Any] = {}
    seed_gains = []
    pooled_original = pooled_official = pooled_n = 0
    for seed in training_seeds:
        per_task = {}
        aggregate_original = aggregate_official = aggregate_n = 0
        for task in TASKS:
            result = summarize_pair(
                cells[seed, task, "original"],
                cells[seed, task, "official"],
                task=task,
            )
            per_task[task] = result
            aggregate_original += result["original"]["successes"]
            aggregate_official += result["official"]["successes"]
            aggregate_n += result["n_per_treatment"]
        gain = (aggregate_official - aggregate_original) / aggregate_n
        seed_gains.append(gain)
        by_seed[str(seed)] = {
            "per_task": per_task,
            "aggregate": {
                "n_per_treatment": aggregate_n,
                "original_successes": aggregate_original,
                "official_successes": aggregate_official,
                "absolute_gain": gain,
                "fisher_exact_two_sided_pvalue": fisher_exact(
                    [
                        [aggregate_original, aggregate_n - aggregate_original],
                        [aggregate_official, aggregate_n - aggregate_official],
                    ],
                    alternative="two-sided",
                ).pvalue,
            },
        }
        pooled_original += aggregate_original
        pooled_official += aggregate_official
        pooled_n += aggregate_n

    output = {
        "schema": "robocasa_phase_replication_stats_v1",
        "claim_scope": (
            "two training seeds; two preselected tasks; fixed-clock target-split evaluation; "
            "unequal 50/20 scene support is reported per seed"
        ),
        "training_seeds": training_seeds,
        "by_training_seed": by_seed,
        "unweighted_mean_training_seed_absolute_gain": sum(seed_gains) / len(seed_gains),
        "all_training_seed_gains_positive": all(gain > 0 for gain in seed_gains),
        "pooled_descriptive": {
            "n_per_treatment": pooled_n,
            "original_successes": pooled_original,
            "official_successes": pooled_official,
            "absolute_gain": (pooled_official - pooled_original) / pooled_n,
            "warning": "descriptive only; seed 1000 contributes more scenes than seed 1001",
        },
        "config_sha256": sha256(args.config),
        "inputs_sha256": input_hashes,
    }
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
