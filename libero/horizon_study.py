"""
Reconstruction-accuracy study: how well does a cubic B-spline capture a SHORT
action chunk, as a function of (chunk window length W) x (n_ctrl control points)?

This sets the default hyperparameters for the chunked fitting pipeline, and is a
paper figure. We slide non-overlapping windows over many episodes, fit each window
to the absolute eef state (pos xyz + axis-angle), and report reconstruction RMSE.

RMSE reported in physical units: position in millimetres, rotation in milliradians.
At 10 fps: W=10 -> 1.0s chunk, W=20 -> 2.0s, W=30 -> 3.0s.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

from lerobot_io import episode_arrays, load_episodes_meta

DEGREE = 3
POS_DIMS = (0, 1, 2)
ROT_DIMS = (3, 4, 5)
FIT_DIMS = POS_DIMS + ROT_DIMS


def make_clamped_uniform_knots(n_ctrl, degree):
    return np.concatenate([np.zeros(degree),
                           np.linspace(0.0, 1.0, n_ctrl - degree + 1),
                           np.ones(degree)])


def fit_window_rmse(state_win):
    """Fit each fitted dim of one window, return (pos_rmse_m, rot_rmse_rad) or None if invalid."""
    L = state_win.shape[0]
    results = {}
    for n_ctrl in N_CTRL_LIST:
        if n_ctrl < DEGREE + 1 or L < n_ctrl:
            results[n_ctrl] = None
            continue
        u = np.linspace(0.0, 1.0, L)
        knots = make_clamped_uniform_knots(n_ctrl, DEGREE)
        try:
            sq_pos, sq_rot = [], []
            for d in FIT_DIMS:
                spl = make_lsq_spline(u, state_win[:, d], knots, k=DEGREE)
                vals = BSpline(knots, spl.c, DEGREE)(u)
                se = (state_win[:, d] - vals) ** 2
                (sq_pos if d in POS_DIMS else sq_rot).append(se)
            results[n_ctrl] = (np.sqrt(np.mean(sq_pos)), np.sqrt(np.mean(sq_rot)))
        except Exception:
            results[n_ctrl] = None
    return results


WINDOW_LIST = [10, 15, 20, 30]
N_CTRL_LIST = [4, 5, 6, 8]


def run(n_episodes=80):
    ep_meta = load_episodes_meta()
    ep_indices = ep_meta["episode_index"].to_numpy()[:n_episodes]

    # accumulate squared errors per (W, n_ctrl)
    acc = {(w, n): {"pos": [], "rot": [], "n_chunks": 0}
           for w in WINDOW_LIST for n in N_CTRL_LIST}

    for ep in ep_indices:
        _, state, _ = episode_arrays(int(ep))
        L = len(state)
        for W in WINDOW_LIST:
            n_win = L // W
            for k in range(n_win):
                win = state[k * W:(k + 1) * W]
                res = fit_window_rmse(win)
                for n in N_CTRL_LIST:
                    if res[n] is not None:
                        acc[(W, n)]["pos"].append(res[n][0])
                        acc[(W, n)]["rot"].append(res[n][1])
                        acc[(W, n)]["n_chunks"] += 1

    print(f"\nReconstruction RMSE across {len(ep_indices)} episodes "
          f"(non-overlapping chunks). pos in mm, rot in mrad.\n")
    print(f"{'W (frames/sec)':>16} | {'n_ctrl':>6} | {'pos RMSE (mm)':>13} | "
          f"{'rot RMSE (mrad)':>15} | {'#chunks':>7}")
    print("-" * 74)
    for W in WINDOW_LIST:
        for n in N_CTRL_LIST:
            a = acc[(W, n)]
            if not a["pos"]:
                continue
            pos_mm = np.mean(a["pos"]) * 1000.0
            rot_mrad = np.mean(a["rot"]) * 1000.0
            wtag = f"{W} ({W/10:.1f}s)"
            print(f"{wtag:>16} | {n:>6} | {pos_mm:>13.3f} | {rot_mrad:>15.3f} | {a['n_chunks']:>7}")
        print("-" * 74)


if __name__ == "__main__":
    run(n_episodes=80)
