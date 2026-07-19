"""Meeting/paper figures from the measured results (numbers hardcoded = final).
Okabe-Ito colorblind-safe palette; direct labels; minimal chrome. -> outputs/fig_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
BLUE, VERM, PURPLE = "#0072B2", "#D55E00", "#CC79A7"

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def fig_pareto():
    # success vs steps-to-completion (on successes). lower steps = faster (left).
    # C-final (h24, 100k) checkpoint, 100 episodes/point, replan nas=5.
    # alpha grid: 1.0, 0.8, 0.7, 0.6, 0.5 (selective = alpha only when T-hat > 20)
    import numpy as np
    anchor = (145.3, 93.0)
    C_sel = [anchor, (124.1, 94.0), (119.4, 85.0), (122.7, 95.0), (113.2, 76.0)]
    C_uni = [anchor, (116.5, 90.0), (110.4, 85.0), (109.6, 73.0), (97.5, 57.0)]
    B_ref = (116.5, 77.0)   # B fixed-T, uniform alpha .6 (no time head -> uniform only)
    N = 100

    def ci95(p):  # binomial half-width, percent
        return 196.0 * np.sqrt(p / 100 * (1 - p / 100) / N)

    alphas = [1.0, 0.8, 0.7, 0.6, 0.5]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    for pts, c, lab, m in [
        (C_sel, BLUE, "duration-aware selective speedup (ours)", "o"),
        (C_uni, VERM, "uniform speedup (blind)", "s"),
    ]:
        steps, succ = zip(*pts)
        ax.errorbar(alphas, succ, yerr=[ci95(y) for y in succ], fmt="-" + m,
                    color=c, lw=2.2, ms=8, label=lab, markeredgecolor="white",
                    markeredgewidth=1.2, capsize=3, elinewidth=1.1, zorder=3)
        ax2.plot(alphas, steps, "-" + m, color=c, lw=2.2, ms=8,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    ax.errorbar(0.6, B_ref[1], yerr=ci95(B_ref[1]), fmt="^", color=GREEN, ms=10,
                markeredgecolor="white", markeredgewidth=1.2, capsize=3, zorder=3,
                label="B: fixed-time head (uniform is its only option)")
    # selectivity gap at matched alpha=.6
    ax.annotate("", xy=(0.6, 76.0), xytext=(0.6, 92.2),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.5))
    ax.text(0.585, 84, "+22 pts", fontsize=10.5, color="#333", va="center", ha="right")
    ax.annotate("95% — above the\nno-speedup baseline", (0.6, 95.0),
                textcoords="offset points", xytext=(-6, 10), fontsize=9.5,
                color=BLUE, ha="center")
    ax2.annotate("16% faster at\nno success cost", (0.6, 122.7), textcoords="offset points",
                 xytext=(0, -26), fontsize=9.5, color=BLUE, ha="center")

    for a in (ax, ax2):
        a.set_xlabel("speedup factor α  (→ more aggressive)")
        a.set_xlim(1.04, 0.46)          # aggressive to the right
        a.set_xticks(alphas)
    ax.set_ylabel("success rate  (%)")
    ax.set_ylim(45, 102)
    ax2.set_ylabel("execution time  (env steps, on successes)")
    ax.set_title("Where to spend speed: rushing everything fails,\n"
                 "rushing only long chunks is free", fontsize=12, loc="left")
    ax2.set_title("Execution time vs α\nLIBERO-Object, 100 episodes/point",
                  fontsize=12, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig("outputs/fig_pareto.png"); plt.close(fig)
    print("saved outputs/fig_pareto.png")


def fig_hz():
    rates = [10, 20, 40]
    series = [
        ("waypoint VLA (naive)", VERM, "o", [0, 93, 0]),
        ("waypoint + interpolation", ORANGE, "s", [67, 93, 59]),
        ("B-spline (ours)", BLUE, "^", [56, 95, 50]),
        ("C: spline + time-alloc (ours)", GREEN, "D", [50, 91, 52]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for lab, c, m, ys in series:
        ax.plot(rates, ys, "-", color=c, lw=2, marker=m, ms=9, label=lab,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{ys[0]}", (rates[0], ys[0]), textcoords="offset points",
                    xytext=(-16, -2), fontsize=9.5, color=c, va="center")

    ax.axvline(20, color="#999", ls=":", lw=1.2, zorder=1)
    ax.text(20.4, 4, "training rate", fontsize=9, color="#777", ha="left", va="bottom")
    # feasibility-stretch gains at f=10 (annotated arrows)
    for y0, y1, c in [(56, 72, BLUE), (50, 66, GREEN), (67, 85, ORANGE)]:
        ax.annotate("", xy=(10.6, y1), xytext=(10.6, y0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.6, alpha=0.9))
    ax.text(11.2, 78, "+ feasibility\nstretch (ours)", fontsize=9.5, color="#333")

    ax.set_xlabel("deployment control rate  (Hz)")
    ax.set_ylabel("success rate  (%)")
    ax.set_title("Robustness to deploy-time control-rate mismatch\n"
                 "LIBERO-Object, 100 episodes/point", fontsize=12.5, loc="left")
    ax.set_xticks(rates); ax.set_ylim(-3, 100)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.tight_layout(); fig.savefig("outputs/fig_hz.png"); plt.close(fig)
    print("saved outputs/fig_hz.png")


def fig_headtohead():
    # success vs REALIZED speedup (steps_base/steps), object suite, same rollout
    # protocol. Speed-as-input (TempoVLA-lite, retrained w/ VSTA + conditioning)
    # vs speed-as-output (ours, decode-only alpha).
    # ALL 100k, same rollout protocol, object, n=100/point (July 12 final):
    # tempo: v1.0 98 @ 135.0; v1.5 80 @ 98.9; v2.0 43 @ 94.6 (realized saturates)
    # dsel (data-level selective retiming, fixed): 85 @ 114.3
    # ours (C-n8): base 90 @ 143.4; sel06 90 @ 114.6; uni06 88 @ 102.9; sel05 84 @ 112.5
    t_steps0, o_steps0 = 135.0, 143.4
    tempo = [(1.0, 98), (t_steps0 / 98.9, 80), (t_steps0 / 94.6, 43)]
    ours_sel = [(1.0, 90), (o_steps0 / 114.6, 90), (o_steps0 / 112.5, 84)]
    ours_uni = [(1.0, 90), (o_steps0 / 102.9, 88), (o_steps0 / 105.0, 73)]
    dsel = (t_steps0 / 114.3, 85)
    # interval-level (speed-profile gated, protect-slow), protocol-crossed n=300:
    # pooled base 94.0 @ 143.8 -> prof06 92.0 @ 110.4
    interval = (143.8 / 110.4, 92.0)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(*interval, s=230, marker="*", color=BLUE, edgecolor="white", lw=1.2,
               zorder=5, label="ours: interval-level retiming (n=300, protocol-crossed)")
    ax.annotate("−2 pts @ 1.30×", interval, textcoords="offset points",
                xytext=(8, 7), fontsize=10, color=BLUE, fontweight="bold")
    ax.scatter(*dsel, s=110, marker="^", color=GREEN, edgecolor="white", lw=1.2,
               zorder=3, label="data-level selective retiming (retrained, fixed speed)")
    for pts, c, lab, m in [
        (ours_sel, BLUE, "ours: decode-α selective (no retraining)", "o"),
        (ours_uni, SKY, "ours: decode-α uniform (no retraining)", "D"),
        (tempo, VERM, "speed-as-input (VSTA + conditioning, retrained)", "s"),
    ]:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-" + m, color=c, lw=2.2, ms=9, label=lab,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.annotate("commanded 2×\nrealizes only 1.43×", tempo[-1], textcoords="offset points",
                xytext=(-4, 14), fontsize=9.5, color=VERM, ha="right")
    ax.annotate("88–90% at realized 1.3–1.4×", (1.33, 91), textcoords="offset points",
                xytext=(0, 8), fontsize=10, color=BLUE, ha="center", fontweight="bold")
    ax.annotate("multi-speed training\nhelps at 1× (composable)", (1.0, 98),
                textcoords="offset points", xytext=(10, -2), fontsize=8.5, color=VERM)
    ax.set_xlabel("realized speedup  (mean steps at 1× / mean steps)")
    ax.set_ylabel("success rate  (%)")
    ax.set_title("Where should speed control live? — all arms 100k, same base model\n"
                 "LIBERO-Object, 100 episodes/point, matched rollout protocol",
                 fontsize=12, loc="left")
    ax.set_ylim(30, 103)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig("outputs/fig_headtohead.png"); plt.close(fig)
    print("saved outputs/fig_headtohead.png")


def fig_calvin():
    # CALVIN ABCD->D, official protocol, SEEDED: 3 training seeds x n=1000 x 4 arms.
    # Left: pooled SR curves (n=3000/arm). Right: the granularity INVERSION —
    # retiming delta vs base by gating granularity, LIBERO vs CALVIN.
    import json
    import os
    d = json.load(open(os.path.join(os.path.dirname(__file__), "data",
                                    "calvin_seeded_final.json")))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    ks = [1, 2, 3, 4, 5]
    for key, c, lab, m in [
        ("U", BLUE, "spline + uniform retiming @1.42×  avg 1.69±.08", "*"),
        ("P", SKY, "spline + interval retiming @1.33×  avg 1.58±.11", "D"),
        ("A", BLACK, "waypoint SmolVLA (best config)  avg 1.56±.15", "s"),
        ("C", "0.55", "spline base (un-retimed)  avg 1.31±.08", "o"),
    ]:
        sr = d[key]["sr_pooled"]
        ax.plot(ks, sr, "-" + m, color=c, lw=2.2, ms=12 if m == "*" else 7,
                label=lab, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate("retiming − base: +0.38, t=12.1,\npositive on every seed",
                (3, 27.8), textcoords="offset points", xytext=(10, 12),
                fontsize=9.5, color=BLUE, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlabel("tasks completed in chain  (≥ k)")
    ax.set_ylabel("success rate  (%)")
    ax.set_title("CALVIN ABCD→D, official protocol — 3 training seeds × 1000 chains\n"
                 "pooled per arm (n=3000); all arms 100k, zero per-benchmark tuning",
                 fontsize=11.5, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # granularity inversion: delta vs base by gating granularity (coarse->fine),
    # LIBERO = success pts (n=300 crossed), CALVIN = avg_len % of base (seeded)
    grans = ["uniform", "chunk", "interval"]
    lib = [-5.3, -3.7, -2.0]
    cb = 1.311
    cal = [100 * d["U-C"]["mean"] / cb, 100 * (1.58 - 1.42) / 1.42,  # chunk: sel06 crossed slices
           100 * d["P-C"]["mean"] / cb]
    x = [0, 1, 2]
    ax2.axhline(0, color="0.6", lw=1)
    ax2.plot(x, lib, "-o", color=VERM, lw=2, ms=9, label="LIBERO (scripted): success Δ, pts",
             markeredgecolor="white", markeredgewidth=1.2)
    ax2.plot(x, cal, "-", color=BLUE, lw=2, zorder=2,
             label="CALVIN (human teleop): avg_len Δ, %")
    ax2.plot([0, 2], [cal[0], cal[2]], "o", color=BLUE, ms=9, zorder=3,
             markeredgecolor="white", markeredgewidth=1.2)
    ax2.plot([1], [cal[1]], "o", color="white", ms=9, zorder=3,
             markeredgecolor=BLUE, markeredgewidth=1.8)
    ax2.annotate("n=200\n(others n=3000)", (1, cal[1]), xytext=(0, -30),
                 textcoords="offset points", fontsize=8, color=BLUE, ha="center")
    ax2.annotate("finer gating better\n(margin is scarce)", (2, -2.0), xytext=(-6, -30),
                 textcoords="offset points", fontsize=9, color=VERM, ha="center")
    ax2.annotate("coarsest gating best\n(dead time is everywhere)", (0, cal[0]),
                 xytext=(8, -22), textcoords="offset points", fontsize=9, color=BLUE)
    ax2.set_xticks(x); ax2.set_xticklabels(["uniform\n(coarsest)", "chunk-level", "interval-level\n(finest)"])
    ax2.set_ylabel("retiming effect vs base decode")
    ax2.set_ylim(-9, 33)
    ax2.set_title("The granularity inversion — where to spend time authority\n"
                  "flips with data regime",
                  fontsize=11.5, loc="left")
    ax2.legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("outputs/fig_calvin.png"); plt.close(fig)
    print("saved outputs/fig_calvin.png")


if __name__ == "__main__":
    fig_pareto()
    fig_hz()
    fig_headtohead()
    fig_calvin()
