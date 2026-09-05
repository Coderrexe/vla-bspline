#!/usr/bin/env python3
"""Render a matched RoboCasa Kettle failure/success storyboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("paper/figures/kettle_storyboard_frames"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/kettle_phase_storyboard.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    columns = [
        "Start\n(step 0)",
        "Pick clause\n(step 150)",
        "Place clause\n(step 200)",
        "Burner clause\n(step 400)",
        "Final",
    ]
    rows = [
        ("original", "Original-caption training\nFAIL"),
        ("official", "Official-phase training\nSUCCESS"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(10.4, 4.25), constrained_layout=True)
    for row_index, (directory, label) in enumerate(rows):
        for column_index, title in enumerate(columns, start=1):
            path = args.frames / directory / f"frame_{column_index:02d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
            axis = axes[row_index, column_index - 1]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            if column_index == 1:
                axis.set_ylabel(label, fontsize=9, fontweight="bold")
            if column_index == 5:
                result = "FAIL: burner not actuated" if row_index == 0 else "SUCCESS: full task complete"
                color = "#555555" if row_index == 0 else "#B95018"
                axis.text(
                    0.5,
                    0.04,
                    result,
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=7.5,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none", "alpha": 0.9},
                )
            for spine in axis.spines.values():
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#B0B0B0" if row_index == 0 else "#D26A32")

    fig.suptitle(
        "Matched RoboCasa Kettle rollout: phase supervision repairs long-horizon execution",
        fontsize=11,
        fontweight="bold",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
