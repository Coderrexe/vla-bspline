#!/usr/bin/env python3
"""Plot RoboCasa phase-supervision stage visitation from validated statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = {
    "KettleBoiling": [
        ("ever_grasped", "grasp"),
        ("lifted_at_least_5cm", "lift ≥5 cm"),
        ("ever_near_burner", "near burner"),
        ("ever_stove_contact", "stove contact"),
        ("ever_any_burner_on", "burner on"),
    ],
    "RinseSinkBasin": [
        ("ever_water_on", "water on"),
        ("washed_at_least_one_region", "wash ≥1 region"),
        ("washed_at_least_two_regions", "wash ≥2 regions"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--training_seed", type=int, default=1000)
    parser.add_argument("--out_pdf", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.stats.read_text())
    if payload.get("schema") != "robocasa_phase_replication_stats_v1":
        raise RuntimeError("unexpected statistics schema")
    seed = payload["by_training_seed"][str(args.training_seed)]

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    colors = {"original": "#8B95A5", "official": "#E06C3B"}
    for axis, task in zip(axes, STAGES):
        result = seed["per_task"][task]
        labels = [label for _, label in STAGES[task]] + ["full success"]
        original = [
            result["diagnostic_progress"][key]["original"]["rate"]
            for key, _ in STAGES[task]
        ] + [result["original"]["rate"]]
        official = [
            result["diagnostic_progress"][key]["official"]["rate"]
            for key, _ in STAGES[task]
        ] + [result["official"]["rate"]]
        x = np.arange(len(labels))
        width = 0.36
        axis.bar(x - width / 2, original, width, color=colors["original"], label="whole caption")
        axis.bar(x + width / 2, official, width, color=colors["official"], label="official phases")
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylim(0, 1)
        axis.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)
        n = result["n_per_treatment"]
        axis.set_title(f"{task} (n={n}/condition)")
        for index, (left, right) in enumerate(zip(original, official)):
            axis.text(index - width / 2, left + 0.025, f"{left:.0%}", ha="center", fontsize=7)
            axis.text(index + width / 2, right + 0.025, f"{right:.0%}", ha="center", fontsize=7)
    axes[0].set_ylabel("episodes reaching stage")
    axes[0].legend(frameon=False, loc="upper right")
    figure.suptitle(f"RoboCasa phase supervision — training seed {args.training_seed}")
    for destination in (args.out_pdf, args.out_png):
        destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight")
    figure.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
