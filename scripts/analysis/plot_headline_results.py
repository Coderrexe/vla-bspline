#!/usr/bin/env python3
"""Render the headline language and RoboCasa results from immutable JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=Path,
        default=Path("outputs/language_clause_exact/locked_3seed_factorial_2327539.json"),
    )
    parser.add_argument(
        "--robocasa_stats",
        type=Path,
        default=Path("outputs/robocasa_confirm_n40/official_phase_50scene_stats.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("paper/figures/headline_results.pdf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load(args.language)
    assert stats["claim_ready"] is True
    seeds = [1000, 1001, 1002]
    a = stats["comparisons"]["A_fixed_clock"]["per_seed"]
    c = stats["comparisons"]["C_fixed_clock"]["per_seed"]
    assert [row["seed"] for row in a] == seeds
    assert [row["seed"] for row in c] == seeds
    a_gain = np.array([100 * row["clause_minus_original"] for row in a])
    c_gain = np.array([100 * row["clause_minus_original"] for row in c])

    interaction = stats["primary_matched_clock_head_interaction"]
    did = np.array([100 * row["c_minus_a_language_interaction"] for row in interaction["per_seed"]])
    did_mean = 100 * interaction["mean_interaction"]
    did_ci = 100 * np.array(interaction["hierarchical_seed_state_bootstrap_95_ci"])

    rc = load(args.robocasa_stats)
    assert rc["schema"] == "robocasa_official_phase_50scene_stats_v1"
    kettle = rc["per_task"]["KettleBoiling"]
    rinse = rc["per_task"]["RinseSinkBasin"]
    rc_original = np.array(
        [
            kettle["original"]["combined"]["rate"],
            rinse["original"]["combined"]["rate"],
            rc["aggregate"]["original"]["successes"] / rc["aggregate"]["original"]["n"],
        ]
    )
    rc_official = np.array(
        [
            kettle["official"]["combined"]["rate"],
            rinse["official"]["combined"]["rate"],
            rc["aggregate"]["official"]["successes"] / rc["aggregate"]["official"]["n"],
        ]
    )
    rc_counts = [(2, 50, 6, 50), (2, 50, 10, 50), (4, 100, 16, 100)]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
    blue, orange = "#3B6FB6", "#D26A32"

    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, a_gain, width, color=blue, label="Waypoint A")
    axes[0].bar(x + width / 2, c_gain, width, color=orange, label="Spline C")
    for xpos, values in ((x - width / 2, a_gain), (x + width / 2, c_gain)):
        for xx, yy in zip(xpos, values, strict=True):
            axes[0].text(xx, yy + 1.1, f"+{yy:.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, [str(seed) for seed in seeds])
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Clause gain (percentage points)")
    axes[0].set_ylim(0, 56)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("a  Executable language clauses", loc="left", fontweight="bold")

    axes[1].axhline(0, color="0.65", linewidth=1)
    axes[1].scatter(x, did, color=orange, s=35, zorder=3, label="Per seed")
    axes[1].errorbar(
        3.1,
        did_mean,
        yerr=[[did_mean - did_ci[0]], [did_ci[1] - did_mean]],
        fmt="D",
        color="black",
        capsize=4,
        label="Mean + hierarchical 95% CI",
    )
    axes[1].set_xticks([0, 1, 2, 3.1], ["1000", "1001", "1002", "Mean"])
    axes[1].set_ylabel("Spline − waypoint clause gain (pp)")
    axes[1].set_ylim(-3, 21)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].set_title("b  Head-specific interaction", loc="left", fontweight="bold")

    labels = ["Kettle\nfixed", "Rinse\nfixed", "Aggregate\nfixed"]
    x = np.arange(3)
    axes[2].bar(x - width / 2, 100 * rc_original, width, color="0.72", label="Original labels")
    axes[2].bar(x + width / 2, 100 * rc_official, width, color=orange, label="Official phases")
    for index, (original, n_original, official, n_official) in enumerate(rc_counts):
        axes[2].text(index - width / 2, 100 * rc_original[index] + 0.8, f"{original}/{n_original}", ha="center", va="bottom", fontsize=7.5)
        axes[2].text(index + width / 2, 100 * rc_official[index] + 0.8, f"{official}/{n_official}", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Target-split success (%)")
    axes[2].set_ylim(0, 27)
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[2].set_title("c  RoboCasa semantic phases", loc="left", fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
