"""Parameterized v2 (event-segmented) spline stats generator — replaces sed-copies
of v2_event_segmentation_study.py. Mirrors the policy's target math exactly,
including the optional contact-weighted (end-weighted) LSQ fit.

  python gen_v2_stats_param.py --n_ctrl 6 --min_seg 6 --h_max 24 --w_end 9 \
      --out <pkg>/spline_stats_libero_v2_h24_w9.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from calvin_io import episode_arrays, episode_indices
from validate_spline_head_math import basis_matrix

DEGREE = 3
PAUSE_FRAC = 0.15
STRIDE = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ctrl", type=int, required=True)
    ap.add_argument("--min_seg", type=int, required=True)
    ap.add_argument("--h_max", type=int, required=True)
    ap.add_argument("--w_end", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_eps", type=int, default=500)
    args = ap.parse_args()
    N, LO, HM = args.n_ctrl, args.min_seg, args.h_max

    ops, pg = {}, {}
    for T in range(LO, HM + 1):
        u = np.arange(T + 1) / T
        B = basis_matrix(u, N)
        if args.w_end is not None:
            sw = np.sqrt(1 + (args.w_end - 1) * np.clip((u - 0.75) / 0.25, 0, 1))
            M = np.linalg.pinv(sw[:, None] * B[:, 1:N - 1]) * sw[None, :]
        else:
            M = np.linalg.pinv(B[:, 1:N - 1])
        ops[T] = (B, M, B[:, N - 1:N])
        ug = np.arange(T) / max(T - 1, 1)
        pg[T] = np.linalg.pinv(basis_matrix(ug, N))

    def first_event(window):
        grip = window[:, 6]
        speed = np.linalg.norm(window[:, :6], axis=1)
        med = np.median(speed) + 1e-9
        for k in range(LO, min(HM, len(window))):
            if grip[k] != grip[k - 1]:
                return k
            if speed[k] < PAUSE_FRAC * med and speed[k - 1] < PAUSE_FRAC * med:
                return k
        return min(HM, len(window))

    ctrl_all, grip_all, logT_all = [], [], []
    for ep in episode_indices()[: args.max_eps]:
        _, _, action = episode_arrays(int(ep))
        L = len(action)
        for t in range(0, L - LO, STRIDE):
            win = action[t:t + HM]
            if len(win) < HM:
                win = np.concatenate([win, np.repeat(win[-1:], HM - len(win), axis=0)])
            T = first_event(win)
            seg = win[:T]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg[:, :6], axis=0)], axis=0)
            B, M, b_last = ops[T]
            p_end = path[-1:, :]
            c_mid = M @ (path - b_last @ p_end)
            ctrl_all.append(np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0))
            grip_all.append(pg[T] @ seg[:, 6])
            logT_all.append(np.log(T))

    ctrl_all = np.stack(ctrl_all); grip_all = np.stack(grip_all); logT_all = np.array(logT_all)
    stats = {
        "version": 2, "n_ctrl": N, "degree": DEGREE,
        "min_seg": LO, "h_max": HM, "pause_frac": PAUSE_FRAC,
        "fit_end_weight": args.w_end,
        "pose_ctrl_mean": ctrl_all.mean(0).tolist(),
        "pose_ctrl_std": np.maximum(ctrl_all.std(0), 1e-4).tolist(),
        "grip_ctrl_mean": grip_all.mean(0).tolist(),
        "grip_ctrl_std": np.maximum(grip_all.std(0), 1e-4).tolist(),
        "logT_mean": float(logT_all.mean()), "logT_std": float(max(logT_all.std(), 1e-4)),
        "n_anchors": len(logT_all),
    }
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"v2 stats -> {args.out}  (n_anchors {len(logT_all)}, "
          f"logT {stats['logT_mean']:.3f}±{stats['logT_std']:.3f})")


if __name__ == "__main__":
    main()
