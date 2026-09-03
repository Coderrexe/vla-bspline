#!/usr/bin/env python3
"""Validate and summarize the disjoint RoboCasa official-phase evaluation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import fisher_exact


@dataclass(frozen=True)
class Cell:
    task: str
    treatment: str
    pilot: Path
    confirm: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def succeeded(episode: dict) -> bool:
    return bool(episode["extrema"]["success"])


def validate_shard(payload: dict, expected_seeds: set[int]) -> None:
    episodes = payload["episodes"]
    seeds = [int(episode["seed"]) for episode in episodes]
    if len(episodes) != payload["n_episodes"] or set(seeds) != expected_seeds:
        raise RuntimeError("episode count or seed grid mismatch")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("duplicate episode seeds")
    if payload["environment_split"] != "target" or payload["renderer"] != "osmesa":
        raise RuntimeError("unexpected evaluation domain")
    if payload["instruction_mode"] not in {
        "kettle_official_fixed",
        "rinse_official_fixed",
    }:
        raise RuntimeError("not a deployable fixed-clock cell")
    observed = sum(succeeded(episode) for episode in episodes)
    if observed != int(payload["successes"]):
        raise RuntimeError("success count mismatch")
    if len({episode["initial_observation_sha256"] for episode in episodes}) != len(episodes):
        raise RuntimeError("duplicate initial observations within a shard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", type=Path, default=Path("outputs/robocasa_official_final"))
    parser.add_argument("--confirm_root", type=Path, default=Path("outputs/robocasa_confirm_n40"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robocasa_confirm_n40/official_phase_50scene_stats.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = [
        Cell(
            "KettleBoiling",
            "original",
            args.pilot_root / "kettleA40_original30_kettle_official_fixed_target_s1000_n10_2327930.json",
            args.confirm_root / "confirmK_original_s1000_kettle_official_fixed_target_s1010_n40_2332999.json",
        ),
        Cell(
            "KettleBoiling",
            "official",
            args.pilot_root / "kettleA40_official30_kettle_official_fixed_target_s1000_n10_2327934.json",
            args.confirm_root / "confirmK_official_s1000_kettle_official_fixed_target_s1010_n40_2333000.json",
        ),
        Cell(
            "RinseSinkBasin",
            "original",
            args.pilot_root / "officialR_original_rinse_official_fixed_target_s1000_n10_2327787.json",
            args.confirm_root / "confirmR_original_s1000_rinse_official_fixed_target_s1010_n40_2332997.json",
        ),
        Cell(
            "RinseSinkBasin",
            "official",
            args.pilot_root / "officialR_official_rinse_official_fixed_target_s1000_n10_2327788.json",
            args.confirm_root / "confirmR_official_s1000_rinse_official_fixed_target_s1010_n40_2332998.json",
        ),
    ]

    loaded: dict[tuple[str, str], tuple[dict, dict]] = {}
    inputs: dict[str, str] = {}
    for cell in cells:
        if not cell.pilot.is_file() or not cell.confirm.is_file():
            raise FileNotFoundError(cell)
        pilot, confirm = load(cell.pilot), load(cell.confirm)
        validate_shard(pilot, set(range(1000, 1010)))
        validate_shard(confirm, set(range(1010, 1050)))
        if pilot["checkpoint_model_sha256"] != confirm["checkpoint_model_sha256"]:
            raise RuntimeError("pilot and confirmation use different checkpoint weights")
        for key in (
            "checkpoint_config_sha256",
            "source_sha256",
            "environment_split",
            "instruction_mode",
            "phase_exec_rate_ratios",
            "renderer",
        ):
            if pilot[key] != confirm[key]:
                raise RuntimeError(f"pilot/confirmation mismatch in {key}")
        loaded[cell.task, cell.treatment] = pilot, confirm
        inputs[str(cell.pilot)] = sha256(cell.pilot)
        inputs[str(cell.confirm)] = sha256(cell.confirm)

    per_task: dict[str, dict] = {}
    totals = {"original": 0, "official": 0}
    for task in ("KettleBoiling", "RinseSinkBasin"):
        records: dict[str, dict] = {}
        for treatment in ("original", "official"):
            pilot, confirm = loaded[task, treatment]
            pilot_successes = int(pilot["successes"])
            confirm_successes = int(confirm["successes"])
            total = pilot_successes + confirm_successes
            totals[treatment] += total
            records[treatment] = {
                "pilot": {"successes": pilot_successes, "n": 10},
                "disjoint_confirmation": {"successes": confirm_successes, "n": 40},
                "combined": {"successes": total, "n": 50, "rate": total / 50},
                "checkpoint_model_sha256": pilot["checkpoint_model_sha256"],
            }

        original = records["original"]["combined"]["successes"]
        official = records["official"]["combined"]["successes"]
        records["contrast"] = {
            "absolute_gain": (official - original) / 50,
            "fold_change": official / original if original else None,
            "fisher_exact_two_sided_pvalue": fisher_exact(
                [[original, 50 - original], [official, 50 - official]],
                alternative="two-sided",
            ).pvalue,
        }
        per_task[task] = records

    aggregate_p = fisher_exact(
        [[totals["original"], 100 - totals["original"]],
         [totals["official"], 100 - totals["official"]]],
        alternative="two-sided",
    ).pvalue
    result = {
        "schema": "robocasa_official_phase_50scene_stats_v1",
        "claim_scope": "one training seed; two preselected tasks; deployable fixed clocks",
        "per_task": per_task,
        "aggregate": {
            "original": {"successes": totals["original"], "n": 100},
            "official": {"successes": totals["official"], "n": 100},
            "absolute_gain": (totals["official"] - totals["original"]) / 100,
            "fold_change": totals["official"] / totals["original"],
            "fisher_exact_two_sided_pvalue": aggregate_p,
        },
        "inputs_sha256": inputs,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
