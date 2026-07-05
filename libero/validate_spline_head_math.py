"""
Validate the mathematical core of the SmolVLA spline head on REAL LIBERO data,
before implementing the policy. Tests:

  T1  Fixed-basis LSQ (precomputed pinv matmul) == scipy make_lsq_spline quality,
      with control point 0 pinned to 0 (chunk starts exactly at current pose).
  T2  Rate retargeting: fit cumsum path on H=20-step windows, decode at
      H' in {10, 20, 40, 80} steps, measure path tracking error vs the
      time-rescaled ground-truth path, and delta CLIPPING at low rates.
  T3  Gripper-as-spline-channel: fit +/-1 signal, decode sign(); measure toggle
      timing error and spurious-toggle rate across many real windows.
  T4  Emit per-token normalization stats (mean/std of control points) for the
      policy config -> outputs/spline_head_stats.json

Conventions (must match the policy implementation exactly):
  - window: H delta actions a_0..a_{H-1}  (7 dims: 6 pose + gripper)
  - path:   p_0 = 0;  p_k = sum_{i<k} a_i  for k = 0..H   (H+1 points, pose dims)
  - u-grid: u_k = k / H
  - basis:  clamped uniform cubic B-spline, n_ctrl control points
  - pose fit: pinned  c_0 = 0, solve c_{1:} = pinv(B[:, 1:]) @ p
  - grip fit: unpinned, on raw +/-1 signal at u_k = k/(H-1), k = 0..H-1
"""
from __future__ import annotations

import json
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from lerobot_io import episode_indices, episode_arrays

DEGREE = 3
N_CTRL = 6
H = 20


def clamped_knots(n_ctrl: int) -> np.ndarray:
    return np.concatenate([np.zeros(DEGREE), np.linspace(0, 1, n_ctrl - DEGREE + 1), np.ones(DEGREE)])


def basis_matrix(u: np.ndarray, n_ctrl: int) -> np.ndarray:
    """(len(u), n_ctrl) matrix of B-spline basis values."""
    kn = clamped_knots(n_ctrl)
    B = np.zeros((len(u), n_ctrl))
    for j in range(n_ctrl):
        c = np.zeros(n_ctrl); c[j] = 1.0
        B[:, j] = BSpline(kn, c, DEGREE)(u)
    return B


# precompute the two operators used by the policy
U_PATH = np.arange(H + 1) / H                     # 21 path points
B_PATH = basis_matrix(U_PATH, N_CTRL)             # (21, 6)
# pin BOTH ends: c_0 = 0 (start at current pose) and c_last = p_H (endpoint exact).
# solve only the middle control points: c_mid = pinv(B_mid) @ (p - B_last * p_H)
PINV_MID = np.linalg.pinv(B_PATH[:, 1:N_CTRL - 1])   # (n_ctrl-2, H+1)
B_LAST = B_PATH[:, N_CTRL - 1:N_CTRL]                # (H+1, 1)
U_GRIP = np.arange(H) / (H - 1)
B_GRIP = basis_matrix(U_GRIP, N_CTRL)             # (20, 6)
PINV_GRIP = np.linalg.pinv(B_GRIP)                # (6, 20)


def fit_pose_pinned(path: np.ndarray) -> np.ndarray:
    """path: (H+1, D), path[0]=0 -> ctrl: (N_CTRL, D) with ctrl[0]=0, ctrl[-1]=path[-1]."""
    p_end = path[-1:, :]                           # (1, D)
    c_mid = PINV_MID @ (path - B_LAST @ p_end)     # (n_ctrl-2, D)
    return np.concatenate([np.zeros((1, path.shape[1])), c_mid, p_end], axis=0)


def decode_path(ctrl: np.ndarray, h_exec: int) -> np.ndarray:
    """ctrl: (N_CTRL, D) -> path samples (h_exec+1, D) at u=j/h_exec."""
    u = np.arange(h_exec + 1) / h_exec
    return basis_matrix(u, N_CTRL) @ ctrl


def windows(max_eps=120, stride=5):
    for ep in episode_indices()[:max_eps]:
        _, _, action = episode_arrays(int(ep))
        for a0 in range(0, len(action) - H, stride):
            yield action[a0:a0 + H]


def main():
    rng = np.random.RandomState(0)
    wins = list(windows())
    print(f"windows: {len(wins)} (H={H}, n_ctrl={N_CTRL})")

    # ---------- T1: matmul-LSQ vs scipy ----------
    idx = rng.choice(len(wins), 300, replace=False)
    err_mine, err_scipy, start_err = [], [], []
    kn = clamped_knots(N_CTRL)
    for i in idx:
        a = wins[i][:, :6]
        path = np.concatenate([np.zeros((1, 6)), np.cumsum(a, axis=0)], axis=0)  # (21, 6)
        ctrl = fit_pose_pinned(path)
        rec = B_PATH @ ctrl
        err_mine.append(np.sqrt(np.mean((rec - path) ** 2)))
        start_err.append(np.abs(rec[0]).max())
        e = 0.0
        for d in range(6):
            spl = make_lsq_spline(U_PATH, path[:, d], kn, k=DEGREE)
            e += np.mean((spl(U_PATH) - path[:, d]) ** 2)
        err_scipy.append(np.sqrt(e / 6))
    print("\n[T1] pinned-matmul RMSE {:.5f}  vs scipy(unpinned) RMSE {:.5f}  "
          "(ratio {:.3f}); max |start error| = {:.2e} (must be 0)".format(
              np.mean(err_mine), np.mean(err_scipy),
              np.mean(err_mine) / max(np.mean(err_scipy), 1e-12), max(start_err)))

    # ---------- T2: rate retargeting + clipping ----------
    print("\n[T2] decode at H' steps; path err vs rescaled GT; delta clipping")
    print(f"{'H_exec':>7} {'rate':>6} {'path RMSE':>10} {'endpoint err':>13} {'|d|>1 frac':>11} {'max|d|':>7}")
    for h_exec in (10, 20, 40, 80):
        perr, eerr, clip, dmax = [], [], [], []
        for i in idx[:150]:
            a = wins[i][:, :6]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(a, axis=0)], axis=0)
            ctrl = fit_pose_pinned(path)
            q = decode_path(ctrl, h_exec)                       # (h_exec+1, 6)
            d = np.diff(q, axis=0)                              # deltas actually emitted
            # ground truth path resampled on the same u-grid (linear interp)
            uu = np.arange(h_exec + 1) / h_exec
            gt = np.stack([np.interp(uu, U_PATH, path[:, k]) for k in range(6)], axis=1)
            perr.append(np.sqrt(np.mean((q - gt) ** 2)))
            eerr.append(np.abs(q[-1] - path[-1]).max())
            clip.append(np.mean(np.abs(d) > 1.0))
            dmax.append(np.abs(d).max())
        print(f"{h_exec:>7} {h_exec/H:>6.2f} {np.mean(perr):>10.5f} {np.max(eerr):>13.2e} "
              f"{np.mean(clip):>11.4f} {np.max(dmax):>7.3f}")

    # ---------- T3: gripper channel ----------
    print("\n[T3] gripper: spline channel + sign() decode")
    n_tog_err, timing_errs, spurious = 0, [], 0
    n_with_toggle = 0
    for i in idx:
        g = wins[i][:, 6]                                       # (20,) in {-1, +1}
        gc = PINV_GRIP @ g                                      # (6,)
        # decode at high resolution to find crossings
        uu = np.linspace(0, 1, 201)
        gs = np.sign(basis_matrix(uu, N_CTRL) @ gc)
        gt_tog = np.where(np.diff(g) != 0)[0]                   # indices of toggles
        dec_tog = np.where(np.diff(gs) != 0)[0]
        if len(gt_tog):
            n_with_toggle += 1
            if len(dec_tog) == len(gt_tog):
                for t_gt, t_dec in zip(gt_tog, dec_tog):
                    timing_errs.append(abs(uu[t_dec] * (H - 1) - (t_gt + 0.5)))
            else:
                n_tog_err += 1
        elif len(dec_tog):
            spurious += 1
    print(f"  windows with toggles: {n_with_toggle}/{len(idx)}; "
          f"toggle-count mismatches: {n_tog_err}; spurious toggles (no-toggle windows): {spurious}")
    if timing_errs:
        print(f"  toggle timing error: mean {np.mean(timing_errs):.2f} steps, "
              f"p95 {np.percentile(timing_errs, 95):.2f} steps (1 step = 1 env step)")

    # ---------- T4: normalization stats ----------
    ctrl_all, grip_all = [], []
    for w in wins:
        a = w[:, :6]
        path = np.concatenate([np.zeros((1, 6)), np.cumsum(a, axis=0)], axis=0)
        ctrl_all.append(fit_pose_pinned(path))                  # (6, 6)
        grip_all.append(PINV_GRIP @ w[:, 6])                    # (6,)
    ctrl_all = np.stack(ctrl_all)                               # (N, 6, 6)
    grip_all = np.stack(grip_all)                               # (N, 6)
    stats = {
        "n_ctrl": N_CTRL, "horizon": H, "degree": DEGREE,
        "pose_ctrl_mean": ctrl_all.mean(0).tolist(),            # (6 tokens, 6 dims)
        "pose_ctrl_std": np.maximum(ctrl_all.std(0), 1e-4).tolist(),
        "grip_ctrl_mean": grip_all.mean(0).tolist(),            # (6 tokens,)
        "grip_ctrl_std": np.maximum(grip_all.std(0), 1e-4).tolist(),
        "n_windows": len(wins),
    }
    with open("outputs/spline_head_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"\n[T4] stats -> outputs/spline_head_stats.json  "
          f"(pose ctrl std range {ctrl_all.std(0).min():.4f}..{ctrl_all.std(0).max():.4f})")


if __name__ == "__main__":
    main()
