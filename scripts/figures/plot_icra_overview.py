#!/usr/bin/env python3
"""Render the ICRA page-one overview as editable vector graphics.

All quantitative values are taken from the immutable primary artifacts cited in
paper/figures/README.md and paper/results.md.  The first two panels are explanatory
schematics; the latter two contain claim-bearing values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import BSpline


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
INK = "#17212B"
MID = "#66727E"
LIGHT = "#F4F7F9"
GRID = "#D7DEE4"


def rounded(ax, xy, width, height, *, fc, ec="none", lw=0.8, radius=0.025):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MID, lw=1.1, scale=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def setup_panel(ax, letter: str, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, (0.0, 0.0), 1.0, 1.0, fc=LIGHT, ec=GRID, lw=0.7, radius=0.025)
    ax.text(0.035, 0.95, letter, color=MID, fontsize=8.2, fontweight="bold", va="top")
    ax.text(0.13, 0.95, title, color=INK, fontsize=8.2, fontweight="bold", va="top")


def panel_language(ax):
    setup_panel(ax, "a", "Executable clauses")
    rounded(ax, (0.08, 0.68), 0.84, 0.16, fc="white", ec=GRID)
    ax.text(
        0.50,
        0.76,
        "put the soup and\ncream cheese in the basket",
        ha="center",
        va="center",
        fontsize=6.25,
        color=INK,
        style="italic",
    )
    arrow(ax, (0.50, 0.67), (0.50, 0.58))
    ax.text(0.30, 0.625, "split at event", ha="center", va="center", fontsize=5.4, color=MID)
    rounded(ax, (0.055, 0.33), 0.41, 0.20, fc="#EAF4FA", ec=SKY)
    rounded(ax, (0.535, 0.33), 0.41, 0.20, fc="#FCEFE9", ec="#E69F73")
    ax.text(0.26, 0.46, "clause 1", ha="center", fontsize=6.2, color=BLUE, fontweight="bold")
    ax.text(0.26, 0.385, "put soup\nin basket", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(0.74, 0.46, "clause 2", ha="center", fontsize=6.2, color=ORANGE, fontweight="bold")
    ax.text(0.74, 0.385, "put cream\ncheese in\nbasket", ha="center", va="center", fontsize=5.9, color=INK, linespacing=0.92)
    ax.text(0.50, 0.22, "ordered program", ha="center", fontsize=6.0, color=MID)
    ax.plot([0.18, 0.82], [0.16, 0.16], color=GRID, lw=2.8, solid_capstyle="round")
    ax.scatter([0.18, 0.82], [0.16, 0.16], s=22, c=[BLUE, ORANGE], zorder=3, edgecolor="white", linewidth=0.5)


def spline_curve():
    ctrl = np.array(
        [
            [0.05, 0.18],
            [0.12, 0.30],
            [0.20, 0.66],
            [0.38, 0.72],
            [0.56, 0.33],
            [0.69, 0.24],
            [0.78, 0.63],
            [0.94, 0.70],
        ]
    )
    degree = 3
    knots = np.concatenate(
        [
            np.zeros(degree),
            np.linspace(0.0, 1.0, len(ctrl) - degree + 1),
            np.ones(degree),
        ]
    )
    u = np.linspace(0.0, 1.0, 300)
    curve = np.column_stack([BSpline(knots, ctrl[:, dim], degree)(u) for dim in range(2)])
    return ctrl, curve


def panel_representation(ax):
    setup_panel(ax, "b", "Spline action program")
    ax.text(0.07, 0.78, "waypoint chunk", fontsize=6.2, color=MID, fontweight="bold")
    x = np.linspace(0.12, 0.91, 25)
    y = 0.70 + 0.025 * np.sin(np.linspace(0, 3 * np.pi, len(x)))
    ax.scatter(x, y, s=5, color="#9AA4AD", linewidth=0)
    ax.text(0.93, 0.705, "50 fixed-rate\ndeltas", ha="right", va="bottom", fontsize=5.6, color=MID)

    ax.text(0.07, 0.56, "ours", fontsize=6.2, color=BLUE, fontweight="bold")
    ctrl, curve = spline_curve()
    x0, x1, y0, y1 = 0.08, 0.89, 0.16, 0.52
    cp = np.column_stack([x0 + (x1 - x0) * ctrl[:, 0], y0 + (y1 - y0) * ctrl[:, 1]])
    cv = np.column_stack([x0 + (x1 - x0) * curve[:, 0], y0 + (y1 - y0) * curve[:, 1]])
    ax.plot(cp[:, 0], cp[:, 1], "--", color="#91A5B3", lw=0.8, zorder=1)
    ax.plot(cv[:, 0], cv[:, 1], color=BLUE, lw=2.0, zorder=2)
    ax.scatter(cp[:, 0], cp[:, 1], s=17, color="white", edgecolor=BLUE, linewidth=1.0, zorder=3)
    ax.scatter([cp[0, 0], cp[-1, 0]], [cp[0, 1], cp[-1, 1]], s=25, color=[BLUE, ORANGE], edgecolor="white", linewidth=0.7, zorder=4)
    clock = Circle((0.84, 0.18), 0.075, transform=ax.transAxes, facecolor="white", edgecolor=ORANGE, lw=1.1)
    ax.add_patch(clock)
    ax.plot([0.84, 0.84], [0.18, 0.225], color=ORANGE, lw=1.0, transform=ax.transAxes)
    ax.plot([0.84, 0.875], [0.18, 0.18], color=ORANGE, lw=1.0, transform=ax.transAxes)
    ax.text(0.76, 0.075, r"learned duration $\hat T$", ha="center", fontsize=5.35, color=ORANGE)
    ax.text(0.11, 0.075, "8 control points", fontsize=5.8, color=BLUE, fontweight="bold")
    ax.text(0.11, 0.025, "+ gripper curve", fontsize=5.5, color=MID)


def panel_language_result(ax):
    setup_panel(ax, "c", "Long-horizon gain")
    chart = ax.inset_axes([0.13, 0.31, 0.80, 0.49])
    whole = np.array([7.0, 11.67])
    clauses = np.array([41.33, 55.33])
    x = np.arange(2)
    width = 0.31
    chart.bar(x - width / 2, whole, width, color="#B9C1C7", label="whole")
    chart.bar(x + width / 2, clauses, width, color=[SKY, ORANGE], label="clauses")
    chart.set_ylim(0, 64)
    chart.set_xticks(x, ["Waypoint", "Spline"])
    chart.set_yticks([0, 20, 40, 60])
    chart.tick_params(labelsize=5.8, length=2, pad=1.5)
    chart.spines[["top", "right"]].set_visible(False)
    chart.spines[["left", "bottom"]].set_color(GRID)
    chart.set_ylabel("success (%)", fontsize=5.7, labelpad=1)
    chart.text(x[0] - width / 2, whole[0] + 1.7, "7.0", ha="center", fontsize=5.1, color=MID)
    chart.text(x[1] - width / 2, whole[1] + 1.7, "11.7", ha="center", fontsize=5.1, color=MID)
    chart.text(x[0] + width / 2, clauses[0] + 2.2, "+34.3", ha="center", fontsize=5.9, color=BLUE, fontweight="bold")
    chart.text(x[1] + width / 2, clauses[1] + 2.2, "+43.7", ha="center", fontsize=5.9, color=ORANGE, fontweight="bold")
    ax.text(0.50, 0.225, "additional spline benefit", ha="center", fontsize=5.35, color=MID)
    rounded(ax, (0.15, 0.105), 0.70, 0.08, fc="white", ec=GRID, lw=0.6, radius=0.018)
    ax.text(0.50, 0.145, "+9.33 pp   95% CI [0.67, 18.0]", ha="center", va="center", fontsize=5.65, color=INK, fontweight="bold")
    ax.text(0.50, 0.045, "3 seeds • exact paired replay", ha="center", fontsize=5.4, color=MID)


def panel_time(ax):
    setup_panel(ax, "d", "Controllable time")
    chart = ax.inset_axes([0.15, 0.25, 0.76, 0.53])
    chart.scatter([1.0], [1.311], s=31, color="#9AA4AD", zorder=3)
    chart.scatter([1.42], [1.690], s=47, marker="D", color=GREEN, edgecolor="white", linewidth=0.6, zorder=4)
    chart.annotate(
        "",
        xy=(1.405, 1.66),
        xytext=(1.035, 1.34),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=GREEN),
    )
    chart.text(1.0, 1.27, "base spline", ha="left", va="top", fontsize=5.6, color=MID)
    chart.text(1.42, 1.715, "retimed", ha="right", va="bottom", fontsize=5.8, color=GREEN, fontweight="bold")
    chart.text(1.31, 1.39, "+0.379 chain length", ha="center", va="center", fontsize=5.5, color=GREEN, fontweight="bold")
    chart.set_xlim(0.96, 1.47)
    chart.set_ylim(1.20, 1.79)
    chart.set_xticks([1.0, 1.2, 1.4])
    chart.set_yticks([1.3, 1.5, 1.7])
    chart.tick_params(labelsize=5.7, length=2, pad=1.5)
    chart.spines[["top", "right"]].set_visible(False)
    chart.spines[["left", "bottom"]].set_color(GRID)
    chart.set_xlabel("realized speed", fontsize=5.7, labelpad=1)
    chart.set_ylabel("CALVIN chain length", fontsize=5.7, labelpad=1)
    ax.text(0.50, 0.105, "1.42× faster + more successful", ha="center", fontsize=5.6, color=GREEN, fontweight="bold")
    ax.text(0.50, 0.045, "same checkpoint • decode only", ha="center", fontsize=5.5, color=MID)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ICRA/figures/overview.pdf"))
    return parser.parse_args()


def main():
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
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.12, 2.28),
        gridspec_kw={"width_ratios": [1.08, 1.18, 1.0, 1.0], "wspace": 0.10},
    )
    panel_language(axes[0])
    panel_representation(axes[1])
    panel_language_result(axes[2])
    panel_time(axes[3])
    fig.subplots_adjust(left=0.008, right=0.992, top=0.985, bottom=0.03)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.015)
    print(args.out)


if __name__ == "__main__":
    main()
