"""
First look at LIBERO trajectories + a first cubic B-spline fit to the ABSOLUTE
end-effector trajectory (observation.state), the target we decided to spline.

Mirrors Quinten's CALVIN `test_bspine_reconstruction.py`, but:
  - fits to ABSOLUTE eef pose from observation.state (not delta actions)
  - uses the dataset's REAL per-frame timestamps (not a hardcoded [0,1] window)
  - fits whole episodes (not fixed 30-frame windows) for this first exploration

Outputs PNGs to ./outputs/.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

from lerobot_io import episode_arrays, load_info

DEGREE = 3
STATE_DIM_NAMES = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip_L", "grip_R"]
ACTION_DIM_NAMES = ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz", "gripper"]
FIT_DIMS = list(range(6))  # pos xyz + axis-angle rx,ry,rz ; gripper (6,7) kept raw


def make_clamped_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector on [0, 1], length n_ctrl + degree + 1."""
    return np.concatenate([
        np.zeros(degree),
        np.linspace(0.0, 1.0, n_ctrl - degree + 1),
        np.ones(degree),
    ])


def fit_state_spline(t: np.ndarray, state: np.ndarray, n_ctrl: int):
    """Least-squares fit a cubic B-spline to each fitted state dim over normalized time.
    Returns (ctrl_pts[dim], knots, recon[L, 8], mse[dim])."""
    u = (t - t[0]) / (t[-1] - t[0])            # normalize time to [0, 1]
    knots = make_clamped_uniform_knots(n_ctrl, DEGREE)

    recon = state.copy()
    ctrl_pts = {}
    mse = {}
    for d in FIT_DIMS:
        spl = make_lsq_spline(u, state[:, d], knots, k=DEGREE)
        ctrl_pts[d] = spl.c.copy()
        vals = BSpline(knots, spl.c, DEGREE)(u)
        recon[:, d] = vals
        mse[d] = float(np.mean((state[:, d] - vals) ** 2))
    return ctrl_pts, knots, recon, mse


def plot_episode(ep_idx: int, n_ctrl: int):
    t, state, action = episode_arrays(ep_idx)
    ctrl_pts, knots, recon, mse = fit_state_spline(t, state, n_ctrl)

    # --- state: original vs spline reconstruction ---
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    axes = axes.flatten()
    for d, ax in enumerate(axes):
        ax.plot(t, state[:, d], label="state (orig)", lw=1.6, alpha=0.85)
        if d in FIT_DIMS:
            ax.plot(t, recon[:, d], "--", lw=1.6, label="B-spline", alpha=0.9)
            tag = f"MSE={mse[d]:.2e}"
        else:
            tag = "[raw, not splined]"
        ax.set_title(f"{STATE_DIM_NAMES[d]}  ({tag})")
        ax.set_xlabel("time (s)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(
        f"LIBERO ep {ep_idx} | absolute eef state vs cubic B-spline "
        f"(n_ctrl={n_ctrl}, L={len(t)}, {t[-1]:.1f}s @10fps)", fontsize=13)
    fig.tight_layout()
    out1 = f"outputs/ep{ep_idx}_state_fit_nctrl{n_ctrl}.png"
    fig.savefig(out1, dpi=110); plt.close(fig)

    # --- action (delta) signals, for contrast ---
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    axes = axes.flatten()
    for d in range(7):
        axes[d].plot(t, action[:, d], color="tab:red", lw=1.2)
        axes[d].set_title(f"action[{d}] {ACTION_DIM_NAMES[d]}")
        axes[d].set_xlabel("time (s)"); axes[d].grid(True, alpha=0.3)
    axes[7].set_visible(False)
    fig.suptitle(f"LIBERO ep {ep_idx} | raw DELTA action signals (noisier; why we spline state)",
                 fontsize=13)
    fig.tight_layout()
    out2 = f"outputs/ep{ep_idx}_action_raw.png"
    fig.savefig(out2, dpi=110); plt.close(fig)

    return mse, out1, out2


def sweep_nctrl(ep_idx: int, nctrl_list):
    """Reconstruction MSE vs number of control points (the key ablation)."""
    t, state, _ = episode_arrays(ep_idx)
    print(f"\n=== ep {ep_idx}: MSE vs n_ctrl (fitted dims pos+rot) ===")
    print(f"{'n_ctrl':>7} | {'pos MSE (avg)':>14} | {'rot MSE (avg)':>14} | {'compression':>11}")
    for n in nctrl_list:
        if n < DEGREE + 1:
            print(f"{n:>7} | {'--- invalid: cubic needs n_ctrl >= degree+1 = 4 ---':>45}")
            continue
        _, _, _, mse = fit_state_spline(t, state, n)
        pos = np.mean([mse[d] for d in (0, 1, 2)])
        rot = np.mean([mse[d] for d in (3, 4, 5)])
        # compression: original L waypoints*6 dims vs n_ctrl*6 + n_ctrl(knots implicit)
        comp = f"{len(t)}->{n}"
        print(f"{n:>7} | {pos:>14.3e} | {rot:>14.3e} | {comp:>11}")


if __name__ == "__main__":
    info = load_info()
    print("fps", info["fps"], "| episodes", info["total_episodes"])

    N_CTRL = 10
    for ep in (0, 1, 2):
        mse, o1, o2 = plot_episode(ep, N_CTRL)
        avg = np.mean(list(mse.values()))
        print(f"ep {ep}: avg fitted-dim MSE = {avg:.3e}  ->  {o1}")

    sweep_nctrl(0, [3, 5, 8, 10, 12, 16, 20])
