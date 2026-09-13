"""Evidence-led figures at their final two-column publication size.

All plotted results come from JSON artifacts. The method diagram alone is
schematic. The teaser uses unmodified simulator frames, not generated imagery.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ICRA/figures"
BLUE, ORANGE, GRAY = "#0072B2", "#B95018", "#59636D"
WIDTH = 7.12


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def teaser():
    fig = plt.figure(figsize=(WIDTH, 1.94))
    frame_ids = [1, 2, 3, 5]
    labels = ["Initial scene", "Kettle grasped", "Above the burner", "Task complete"]
    for i, (frame_id, label) in enumerate(zip(frame_ids, labels)):
        ax = fig.add_axes([.005 + .251 * i, .15, .237, .73])
        ax.imshow(Image.open(ROOT / f"paper/figures/kettle_storyboard_frames/official/frame_{frame_id:02d}.png"))
        ax.axis("off")
        ax.set_title(label, fontsize=9, pad=5)
    fig.text(.5, .048, "Pick up the kettle   →   Place it on the stove   →   Turn on the burner",
             ha="center", va="center", fontsize=10)
    save(fig, "teaser_v2")


def method():
    fig = plt.figure(figsize=(WIDTH, 1.92))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, 1.92)
    ax.axis("off")
    text_color = "#20252B"
    pale_blue, pale_orange, panel, white = "#EAF3F8", "#FFF0E7", "#F5F7F8", "#FFFFFF"
    header_size, body_size, secondary_size = 8.2, 7.8, 7.3

    # Explicit layout grid. All content is placed inside these panel bounds.
    panels = [(.05, 1.78), (2.03, 1.16), (3.39, 1.72), (5.31, 1.76)]
    panel_y, panel_h = .34, 1.46
    fills = [panel, pale_blue, panel, panel]
    titles = ["1   SEMANTIC INPUT", "2   POLICY", "3   EVENT SPLINE", "4   EXECUTION"]
    for (x, width), fill, title in zip(panels, fills, titles):
        ax.add_patch(Rectangle((x, panel_y), width, panel_h, fc=fill, ec="none"))
        ax.text(x + width / 2, 1.62, title, ha="center", va="center",
                fontsize=header_size, weight="bold", color=text_color)

    # Semantic input: the row treatment, not a floating label, marks the active clause.
    x, width = panels[0]
    center = x + width / 2
    ax.text(center, 1.31, "image  +  robot state", ha="center", va="center",
            fontsize=body_size, color=text_color)
    row_x, row_w, row_h = x + .13, width - .26, .27
    ax.add_patch(Rectangle((row_x, .91), row_w, row_h, fc=pale_orange, ec="none"))
    ax.add_patch(Rectangle((row_x, .91), .045, row_h, fc=ORANGE, ec="none"))
    ax.text(center + .015, 1.045, "1   soup  →  basket", ha="center", va="center",
            fontsize=body_size, weight="bold", color=text_color)
    ax.add_patch(Rectangle((row_x, .58), row_w, row_h, fc=white, ec="#D9DEE2", lw=.7))
    ax.text(center + .015, .715, "2   tomato  →  basket", ha="center", va="center",
            fontsize=body_size, color=GRAY)

    # Policy: generous internal padding and two centered typographic levels.
    x, width = panels[1]
    box_x, box_w = x + .13, width - .26
    ax.add_patch(Rectangle((box_x, .70), box_w, .68, fc=white, ec=BLUE, lw=1.15))
    ax.text(x + width / 2, 1.10, "SmolVLA", ha="center", va="center",
            fontsize=8.8, weight="bold", color=text_color)
    ax.text(x + width / 2, .88, "flow action head", ha="center", va="center",
            fontsize=secondary_size, color=GRAY)

    # Event spline: curve and labels each occupy their own horizontal band.
    x, width = panels[2]
    u = np.linspace(0, 1, 120)
    curve_x = x + .18 + (width - .36) * u
    curve_y = 1.22 + .15 * np.sin(np.pi * u) - .065 * np.sin(2 * np.pi * u)
    uc = np.linspace(0, 1, 8)
    ctrl_x = x + .18 + (width - .36) * uc
    ctrl_y = 1.22 + .15 * np.sin(np.pi * uc) - .065 * np.sin(2 * np.pi * uc)
    ctrl_y += np.array([0, -.05, .055, -.04, .045, -.035, .04, 0])
    ax.plot(ctrl_x, ctrl_y, color="#9AA2A9", lw=.75, ls="--", zorder=1)
    ax.plot(curve_x, curve_y, color=BLUE, lw=1.8, zorder=2)
    ax.scatter(ctrl_x, ctrl_y, s=13, fc="white", ec=ORANGE, lw=1.0, zorder=3)
    ax.text(x + width / 2, .91, "8 pose controls  •  gripper curve", ha="center",
            va="center", fontsize=body_size, color=text_color)
    duration_w = 1.12
    ax.add_patch(Rectangle((x + (width - duration_w) / 2, .54), duration_w, .23,
                           fc=pale_orange, ec="none"))
    ax.text(x + width / 2, .655, r"duration  $\widehat T$", ha="center", va="center",
            fontsize=body_size, color=text_color)

    # Execution: a two-column internal grid keeps labels away from the marks.
    x, width = panels[3]
    label_x, mark_l, mark_r = x + .17, x + .90, x + width - .16
    ax.text(label_x, 1.34, r"pace  $\alpha$", ha="left", va="center",
            fontsize=body_size, color=text_color)
    ax.plot([mark_l, mark_r - .04], [1.34, 1.34], color=ORANGE, lw=2)
    ax.annotate("", (mark_r, 1.34), (mark_r - .11, 1.34),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.text(label_x, 1.04, r"rate  $\rho$", ha="left", va="center",
            fontsize=body_size, color=text_color)
    sample_x = np.linspace(mark_l, mark_r, 8)
    ax.plot([mark_l, mark_r], [1.04, 1.04], color=BLUE, lw=1.2)
    ax.scatter(sample_x, np.full_like(sample_x, 1.04), s=9, color=BLUE)
    ax.text(x + width / 2, .73, r"$h=\mathrm{round}(\alpha\rho\widehat T)$",
            ha="center", va="center", fontsize=8.5, color=text_color)
    ax.text(x + width / 2, .48, "curve samples  →  actions", ha="center", va="center",
            fontsize=secondary_size, color=GRAY)

    # Arrows are confined to fixed inter-panel gutters.
    for (left_x, left_w), (right_x, _) in zip(panels[:-1], panels[1:]):
        ax.annotate("", (right_x - .035, 1.06), (left_x + left_w + .035, 1.06),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
    ax.text(WIDTH / 2, .13,
            "A clause selects the subgoal; several event-aligned segments execute it.",
            ha="center", va="center", fontsize=8.3, color=GRAY)
    save(fig, "method_v4")


def language(stats):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=.085, right=.98, bottom=.22, top=.80, wspace=.39)
    ax = axes[0]
    for key, label, color, offset in [("A", "Waypoint", BLUE, -.035), ("C", "Event spline", ORANGE, .035)]:
        rows = stats["comparisons"][key + "_fixed_clock"]["per_seed"]
        data = np.array([[r["original_successes"], r["clause_successes"]] for r in rows])
        for i, values in enumerate(data):
            ax.plot(np.array([0, 1]) + offset, values, color=color, alpha=.3, lw=.9)
        ax.plot(np.array([0, 1]) + offset, data.mean(0), marker="o", ms=4.5,
                color=color, lw=1.6, label=label)
        ax.annotate(f"{data.mean(0)[1]:.1f}%", (1 + offset, data.mean(0)[1]),
                    xytext=(5, 0), textcoords="offset points", va="center", color=color, fontsize=9)
    ax.set_xlim(-.20, 1.42)
    ax.set_ylim(0, 65)
    ax.set_xticks([0, 1], ["Whole-task labels", "Clause labels"])
    ax.set_ylabel("Scheduled success (%)")
    ax.set_yticks([0, 20, 40, 60])
    ax.set_title("(a) Same clause schedule at test time", loc="left", fontsize=9, pad=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.3)
    ax = axes[1]
    x = np.arange(3)
    w = .32
    for key, label, color, shift in [("A", "Waypoint", BLUE, -w / 2), ("C", "Event spline", ORANGE, w / 2)]:
        rows = stats["comparisons"][key + "_fixed_clock"]["per_seed"]
        values = [100 * r["clause_minus_original"] for r in rows]
        ax.bar(x + shift, values, w, color=color, label=label)
        for xi, yi in zip(x + shift, values):
            ax.text(xi, yi + 1, f"{yi:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 57)
    ax.set_xticks(x, ["1000", "1001", "1002"])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Clause-label gain (pp)")
    ax.set_yticks([0, 20, 40])
    ax.set_title("(b) Larger label effect for the spline head", loc="left", fontsize=9, pad=9)
    save(fig, "language_v2")


def clock(calvin, matched):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=.085, right=.98, bottom=.23, top=.83, wspace=.39)
    ax = axes[0]
    for key, label, color, style in [("A", "Waypoint, native", BLUE, "-"),
            ("C", "Spline, native", GRAY, "--"), ("U", "Spline, retimed", ORANGE, "-")]:
        ax.plot(np.arange(1, 6), calvin[key]["sr_pooled"], style,
                marker="o", ms=3.5, lw=1.3, color=color, label=label)
    ax.set_ylim(0, 80)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Consecutive tasks completed")
    ax.set_ylabel("Chains reaching depth (%)")
    ax.set_title("(a) Official CALVIN chains", loc="left", fontsize=9, pad=9)
    ax.legend(frameon=False, fontsize=8, handlelength=1.5)
    ax = axes[1]
    x = np.array([0., 1.])
    for native, retimed, label, color, shift in [
            ("A_native", "A_linear", "Waypoint", BLUE, -.025),
            ("C_native", "C_retime", "Event spline", ORANGE, .025)]:
        values = np.array([matched["arms"][native]["per_seed_mean"],
                           matched["arms"][retimed]["per_seed_mean"]]).T
        for row in values:
            ax.plot(x + shift, row, color=color, alpha=.25, lw=.9)
        mean = values.mean(axis=0)
        ax.plot(x + shift, mean, color=color, marker="o", ms=4.5, lw=1.7, label=label)
        ax.text(1.06 + shift, mean[1], f"+{mean[1] - mean[0]:.2f}", color=color,
                fontsize=8.5, va="center")
    ax.set_xticks(x, ["Native", "Retimed"])
    ax.set_xlim(-.20, 1.34)
    ax.set_ylim(1.0, 2.35)
    ax.set_yticks([1.0, 1.4, 1.8, 2.2])
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Mean completed tasks")
    ax.set_title("(b) Matched screen at five actions/query", loc="left", fontsize=9, pad=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.3)
    save(fig, "clock_v3")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white"})
    stats = json.loads((ROOT / "outputs/language_clause_exact/locked_3seed_factorial_2327539.json").read_text())
    assert stats["claim_ready"]
    calvin = json.loads((ROOT / "scripts/figures/data/calvin_seeded_final.json").read_text())
    matched = json.loads((ROOT / "outputs/calvin_matched_timing_20260905/summary_2397013_audited_2423928.json").read_text())
    teaser()
    method()
    language(stats)
    clock(calvin, matched)


if __name__ == "__main__":
    main()
