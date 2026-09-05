#!/usr/bin/env python3
"""Render the manuscript's audited timing and control-rate results.

The CALVIN values come from the three-seed official-chain table in
docs/TEAM_UPDATE.md and docs/STATS_TABLES.md.  The LIBERO values come from the
protocol-crossed retiming and canonical paired-rate cells in paper/results.md.
This script intentionally plots absolute outcomes and explicitly identifies each
comparator; deltas are annotations, not substitutes for the underlying values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#8A949C"
LIGHT = "#DCE3E8"
INK = "#17212B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ICRA/figures/control_results.pdf"))
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB3BA")
    ax.tick_params(labelsize=7, length=2.5, color="#7C8790")
    ax.grid(axis="y", color=LIGHT, linewidth=0.55, alpha=0.8, zorder=0)


def panel_calvin(ax) -> None:
    waypoint = np.array([1.510, 1.764, 1.416])
    base = np.array([1.422, 1.239, 1.273])
    retimed = np.array([1.734, 1.579, 1.759])
    x = np.arange(3)

    for seed in range(3):
        ax.plot(x[1:], [base[seed], retimed[seed]], color="#82CAB6", lw=1.2, zorder=1)
    ax.scatter(np.full(3, x[0]), waypoint, marker="s", s=25, color=GRAY, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(np.full(3, x[1]), base, marker="o", s=28, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(np.full(3, x[2]), retimed, marker="D", s=30, color=GREEN, edgecolor="white", linewidth=0.5, zorder=3)
    for xpos, values, color in zip(x, [waypoint, base, retimed], [GRAY, BLUE, GREEN], strict=True):
        mean = values.mean()
        ax.plot([xpos - 0.20, xpos + 0.20], [mean, mean], color=INK, lw=1.5, zorder=4)
        ax.text(xpos, 1.825, f"mean {mean:.2f}", ha="center", va="bottom", fontsize=6.6, color=color, fontweight="bold")

    ax.text(1.50, 1.115, r"$+0.379$", ha="center", va="center", fontsize=7.6, color=GREEN, fontweight="bold")
    ax.text(1.50, 1.065, r"at $1.42\times$ speed", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, ["Waypoint A", "Spline C\nbase", "Spline C\nretimed"])
    ax.set_ylim(1.00, 1.92)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_ylabel("CALVIN chain length", fontsize=7.4)
    ax.set_title("a  CALVIN retiming", loc="left", fontsize=8.5, fontweight="bold", color=INK)
    style_axis(ax)


def panel_pareto(ax) -> None:
    speed = np.array([1.00, 1.30, 1.38])
    success = np.array([94.0, 92.0, 88.7])
    ax.plot(speed, success, color=BLUE, lw=1.4, zorder=2)
    ax.scatter(speed, success, s=[30, 38, 38], color=[GRAY, BLUE, GREEN], edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([1.43], [43.0], marker="X", s=46, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)

    ax.annotate("C base\n94%", (1.00, 94.0), xytext=(1.035, 87.0), fontsize=6.6, ha="left", color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
    ax.annotate("interval-selective\n92%", (1.30, 92.0), xytext=(1.20, 78.0), fontsize=6.6, ha="center", color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.annotate("uniform\n88.7%", (1.38, 88.7), xytext=(1.405, 94.5), fontsize=6.6, ha="center", color=GREEN,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    ax.annotate("speed-as-input\n43%", (1.43, 43.0), xytext=(1.31, 51.0), fontsize=6.6, ha="center", color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))

    ax.set_xlim(0.97, 1.48)
    ax.set_ylim(35, 101)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4])
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlabel("realized speed ratio", fontsize=7.4)
    ax.set_ylabel("LIBERO Object success (%)", fontsize=7.4)
    ax.set_title("b  LIBERO speed--success", loc="left", fontsize=8.5, fontweight="bold", color=INK)
    style_axis(ax)


def panel_rate(ax) -> None:
    labels = ["600/1,200\nsteps", "1,000/2,000\nsteps"]
    native = np.array([38.0, 34.0])
    doubled = np.array([68.0, 72.0])
    x = np.arange(2)
    width = 0.28
    ax.bar(x - width / 2, native, width, color=GRAY, edgecolor="white", linewidth=0.5, label="Spline C at 1×", zorder=2)
    ax.bar(x + width / 2, doubled, width, color=BLUE, edgecolor="white", linewidth=0.5, hatch="//", label="Spline C at 2×", zorder=2)
    for index in range(2):
        ax.text(x[index] - width / 2, native[index] + 2.0, f"{native[index]:.0f}%", ha="center", fontsize=6.8, color=INK)
        ax.text(x[index] + width / 2, doubled[index] + 2.0, f"{doubled[index]:.0f}%", ha="center", fontsize=6.8, color=BLUE, fontweight="bold")
    ax.scatter([x[0] - 0.43], [0], marker="X", s=35, color=ORANGE, clip_on=False, zorder=4)
    ax.text(x[0] - 0.43, 5.5, "waypoint\n0/50 at 2×", ha="center", va="bottom", fontsize=6.2, color=ORANGE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 88)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("LIBERO Spatial t5 success (%)", fontsize=7.4)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2, handlelength=1.4, columnspacing=0.8)
    ax.set_title("c  LIBERO rate adaptation", loc="left", fontsize=8.5, fontweight="bold", color=INK)
    style_axis(ax)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.20), constrained_layout=True)
    panel_calvin(axes[0])
    panel_pareto(axes[1])
    panel_rate(axes[2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.025)
    print(args.out)


if __name__ == "__main__":
    main()
