"""Flagship duration-head figures from measured data:
  fig_calibration.png — T-hat vs event-truth scatter, colored by chunk type
                        (events: r=0.84 — the head learned time-to-grasp;
                         caps: the smearing that motivated the mode-snap)
  fig_timeline.png    — closed-loop T-hat staircase over a libero_10 episode
                        with gripper actions overlaid (T-hat falls as the
                        grasp approaches, resets after)
Okabe-Ito palette, direct labels, minimal chrome.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
plt.rcParams.update({
    "figure.dpi": 140, "font.size": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

DIAG = sys.argv[1] if len(sys.argv) > 1 else "diag"


def fig_calibration():
    z = np.load(f"{DIAG}/calib.npz", allow_pickle=True)
    T, H, types = z["T_true"], z["T_hat"].mean(axis=1), z["types"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    rng = np.random.RandomState(0)
    jit = lambda x: x + rng.uniform(-0.35, 0.35, len(x))
    ev = types == "gripper"
    cap = types == "cap"
    ax.scatter(jit(T[ev]), H[ev], s=14, alpha=0.35, color=BLUE,
               label=f"grasp-event chunks (r=0.84, MAE 3.7)", edgecolors="none")
    ax.scatter(jit(T[cap]), H[cap], s=14, alpha=0.35, color=VERM,
               label="capped transport chunks (smear: 40→~30)", edgecolors="none")
    lims = [4, 42]
    ax.plot(lims, lims, "--", color="#888", lw=1.3, zorder=1)
    ax.text(38.5, 39.5, "perfect", color="#777", fontsize=9, rotation=38, ha="right")
    ax.set_xlabel("true time-to-event  (env steps)")
    ax.set_ylabel("predicted duration  T̂  (env steps)")
    ax.set_title("The duration head learned time-to-grasp\n"
                 "1280 dataset frames × 4 samples, C (time-alloc) 100k",
                 fontsize=12.5, loc="left")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig("outputs/fig_calibration.png"); plt.close(fig)
    print("saved outputs/fig_calibration.png")


def fig_timeline():
    # pick an episode with a clear structure: use C_t0 episode 0 (success, 327 steps)
    z = np.load(f"{DIAG}/C_t0.npz", allow_pickle=True)
    ep = 0
    that = np.asarray(z["t_hat"][ep], dtype=float)
    act = np.asarray(z["actions"][ep], dtype=float)
    grip = act[:, 6]
    steps = np.arange(len(that))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(steps, that, color=BLUE, lw=2, label="predicted duration T̂", zorder=3)
    ax.axhline(24, color="#999", ls=":", lw=1.1)
    ax.text(len(that) * 0.99, 24.7, "cap (transport)", color="#777", fontsize=9, ha="right")

    # gripper close/open events
    closes = np.where((grip[1:] > 0) & (grip[:-1] <= 0))[0] + 1
    opens = np.where((grip[1:] <= 0) & (grip[:-1] > 0))[0] + 1
    for i, k in enumerate(closes):
        ax.axvline(k, color=VERM, lw=1.4, alpha=0.8)
        if i == 0:
            ax.text(k + 3, 8, "grasp", color=VERM, fontsize=10, rotation=90, va="bottom")
    for i, k in enumerate(opens):
        ax.axvline(k, color=GREEN, lw=1.4, alpha=0.8)
        if i == 0:
            ax.text(k + 3, 8, "release", color=GREEN, fontsize=10, rotation=90, va="bottom")

    ax.set_xlabel("episode step  (libero_10 task, closed loop)")
    ax.set_ylabel("T̂  (env steps)")
    ax.set_title("Closed loop: T̂ pins to the cap during transport and falls as\n"
                 "each manipulation event approaches", fontsize=12.5, loc="left")
    ax.set_ylim(0, 46)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig("outputs/fig_timeline.png"); plt.close(fig)
    print("saved outputs/fig_timeline.png")


if __name__ == "__main__":
    fig_calibration()
    fig_timeline()
