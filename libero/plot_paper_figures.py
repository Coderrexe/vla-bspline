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


if __name__ == "__main__":
    fig_pareto()
    fig_hz()
