"""
v2 groundwork: EVENT-SEGMENTED variable-duration chunks.

The time-allocation head only earns its place if chunk duration — defined as
"env steps until the current sub-motion completes" — (a) varies substantially
across anchors, and (b) is predictable from context. This script measures (a)
on the full dataset and generates the v2 normalization stats.

Event rule (simple, defensible, computable on-the-fly from the fetched window):
  at anchor t, scan k = MIN_SEG..H_MAX-1 of the future window:
    - GRIPPER event: action[t+k, 6] != action[t+k-1, 6]        (toggle)
    - PAUSE event:   ||pose_delta||_2 < PAUSE_FRAC * window median speed
                     for 2 consecutive steps
  T(t) = first event index, else H_MAX (cap).

Outputs: duration distribution + event-type breakdown + per-token control-point
stats over variable-length fits -> smolvla_spline_pkg/spline_stats_libero_v2.json
"""
from __future__ import annotations

import json
import numpy as np

from lerobot_io import episode_indices, episode_arrays
from validate_spline_head_math import basis_matrix  # reuse the validated basis

MIN_SEG, H_MAX = 6, 40
PAUSE_FRAC = 0.15
N_CTRL, DEGREE = 6, 3
STRIDE = 5


def first_event(window: np.ndarray) -> tuple[int, str]:
    """window: (H_MAX, 7) delta actions. Return (T, event_type)."""
    grip = window[:, 6]
    speed = np.linalg.norm(window[:, :6], axis=1)
    med = np.median(speed) + 1e-9
    for k in range(MIN_SEG, H_MAX):
        if grip[k] != grip[k - 1]:
            return k, "gripper"
        if k >= 1 and speed[k] < PAUSE_FRAC * med and speed[k - 1] < PAUSE_FRAC * med:
            return k, "pause"
    return H_MAX, "cap"


def fit_ops(H: int):
    """Pinned-both-ends fit operators for a window of H deltas (H+1 path pts)."""
    u = np.arange(H + 1) / H
    B = basis_matrix(u, N_CTRL)
    return np.linalg.pinv(B[:, 1:N_CTRL - 1]), B[:, N_CTRL - 1:N_CTRL]


def main():
    ops = {H: fit_ops(H) for H in range(MIN_SEG, H_MAX + 1)}
    ug = {H: basis_matrix(np.arange(H) / (H - 1), N_CTRL) for H in range(MIN_SEG, H_MAX + 1)}
    pinv_grip = {H: np.linalg.pinv(ug[H]) for H in ug}

    durations, types = [], []
    ctrl_all, grip_all, logT_all = [], [], []
    eps = episode_indices()
    for ep in eps:
        _, _, action = episode_arrays(int(ep))
        L = len(action)
        for t in range(0, L - MIN_SEG, STRIDE):
            win = action[t:t + H_MAX]
            if len(win) < H_MAX:  # pad by repeating last (matches dataloader clamping)
                win = np.concatenate([win, np.repeat(win[-1:], H_MAX - len(win), axis=0)])
            T, ev = first_event(win)
            durations.append(T); types.append(ev)
            # variable-length pinned fit for stats
            seg = win[:T]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg[:, :6], axis=0)], axis=0)
            pinv_mid, b_last = ops[T]
            p_end = path[-1:, :]
            c_mid = pinv_mid @ (path - b_last @ p_end)
            c_pose = np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0)
            ctrl_all.append(c_pose)
            grip_all.append(pinv_grip[T] @ seg[:, 6])
            logT_all.append(np.log(T))

    durations = np.array(durations); types = np.array(types)
    ctrl_all = np.stack(ctrl_all); grip_all = np.stack(grip_all); logT_all = np.array(logT_all)

    print(f"anchors: {len(durations)} (all {len(eps)} episodes, stride {STRIDE})")
    print(f"\n=== DURATION DISTRIBUTION (env steps; MIN={MIN_SEG}, CAP={H_MAX}) ===")
    qs = np.percentile(durations, [5, 25, 50, 75, 95])
    print(f"mean {durations.mean():.1f} | p5 {qs[0]:.0f}  p25 {qs[1]:.0f}  p50 {qs[2]:.0f}  "
          f"p75 {qs[3]:.0f}  p95 {qs[4]:.0f} | std {durations.std():.1f}")
    print(f"capped at H_MAX: {(durations == H_MAX).mean() * 100:.1f}%")
    print("\nevent types:")
    for ev in ("gripper", "pause", "cap"):
        m = types == ev
        if m.any():
            print(f"  {ev:8s}: {m.mean() * 100:5.1f}%   T mean {durations[m].mean():5.1f}  "
                  f"std {durations[m].std():4.1f}")
    # coefficient of variation — the 'learnable signal' metric
    print(f"\ncoefficient of variation of T: {durations.std() / durations.mean():.2f} "
          "(0 = constant/no signal)")

    stats = {
        "stats_contract_version": 1,
        "version": 2, "n_ctrl": N_CTRL, "degree": DEGREE,
        "min_seg": MIN_SEG, "h_max": H_MAX, "pause_frac": PAUSE_FRAC,
        "action_layout": "eef7", "pose_dims": list(range(6)),
        "grip_idx": 6, "pass_dims": [],
        "boundary_semantics": "production_event_index_k_exclusive",
        "pose_ctrl_mean": ctrl_all.mean(0).tolist(),
        "pose_ctrl_std": np.maximum(ctrl_all.std(0), 1e-4).tolist(),
        "grip_ctrl_mean": grip_all.mean(0).tolist(),
        "grip_ctrl_std": np.maximum(grip_all.std(0), 1e-4).tolist(),
        "logT_mean": float(logT_all.mean()), "logT_std": float(max(logT_all.std(), 1e-4)),
        "n_anchors": len(durations),
    }
    out = "smolvla_spline_pkg/spline_stats_libero_v2.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"\nv2 stats -> {out}")


if __name__ == "__main__":
    main()
