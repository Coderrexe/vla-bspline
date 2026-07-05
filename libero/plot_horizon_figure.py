"""Figure: B-spline reconstruction RMSE vs (chunk length, n_ctrl) on LIBERO.
Recomputes the sweep (cached data) and saves outputs/horizon_study.png."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot_io import episode_indices, episode_arrays
from bspline_core import fit_chunk, chunk_rmse, FIT_DIMS

WINDOWS = [10, 15, 20, 30]
NCTRL = [4, 5, 6, 8]


def collect(n_episodes=80):
    eps = episode_indices()[:n_episodes]
    acc = {(w, n): {"pos": [], "rot": []} for w in WINDOWS for n in NCTRL}
    for ep in eps:
        _, state, _ = episode_arrays(int(ep))
        L = len(state)
        for W in WINDOWS:
            for k in range(L // W):
                win = state[k * W:(k + 1) * W]
                for n in NCTRL:
                    if n > W:
                        continue
                    try:
                        f = fit_chunk(win, n, FIT_DIMS)
                    except ValueError:
                        continue
                    e = chunk_rmse(f["recon"], win, FIT_DIMS)
                    acc[(W, n)]["pos"].append(e["pos_rmse"])
                    acc[(W, n)]["rot"].append(e["rot_rmse"])
    return acc


def main():
    acc = collect()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(WINDOWS)))
    for w, c in zip(WINDOWS, colors):
        pos = [np.mean(acc[(w, n)]["pos"]) * 1000 for n in NCTRL]
        rot = [np.mean(acc[(w, n)]["rot"]) * 1000 for n in NCTRL]
        ax1.plot(NCTRL, pos, "o-", color=c, label=f"H={w} ({w/10:.1f}s)")
        ax2.plot(NCTRL, rot, "o-", color=c, label=f"H={w} ({w/10:.1f}s)")
    for ax, ylab, ref, reflab in [
        (ax1, "position RMSE (mm)", 1.0, "1 mm"),
        (ax2, "orientation RMSE (mrad)", None, None)]:
        ax.set_xlabel("control points per dim (n_ctrl)")
        ax.set_ylabel(ylab); ax.set_yscale("log"); ax.set_xticks(NCTRL)
        ax.grid(True, which="both", alpha=0.3); ax.legend(title="chunk horizon", fontsize=8)
        if ref:
            ax.axhline(ref, ls="--", color="crimson", lw=1, alpha=0.7)
            ax.text(NCTRL[-1], ref * 1.1, reflab, color="crimson", ha="right", fontsize=8)
    fig.suptitle("Cubic B-spline reconstruction of LIBERO absolute-eef chunks "
                 "(fit to observation.state, 80 episodes)", fontsize=12)
    fig.tight_layout()
    fig.savefig("outputs/horizon_study.png", dpi=130)
    print("saved -> outputs/horizon_study.png")


if __name__ == "__main__":
    main()
