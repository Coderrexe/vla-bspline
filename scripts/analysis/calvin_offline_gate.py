"""CALVIN offline gate — the same pre-training validation LIBERO went through:
  1. event segmentation statistics (duration distribution, CoV, event fractions)
     under the v2 rules at candidate (min_seg, h_max) settings [CALVIN is 10 fps
     vs LIBERO 20 fps — event scales may differ];
  2. spline fit quality (n_ctrl 6/8): per-step relative reconstruction error,
     tail vs body, + speed-profile error + terminal-deceleration capture.

Pass criteria (LIBERO reference values in brackets): CoV of event durations
comparable [0.38 @ cap40 / 0.20 @ cap24]; gripper-event fraction meaningful
[23-47%]; n8 rel recon err ~[0.04]; speed-profile err ~[0.02].

Run: python calvin_offline_gate.py --max_eps 300
"""
from __future__ import annotations

import argparse

import numpy as np

from calvin_io import episode_arrays, episode_indices
from validate_spline_head_math import basis_matrix

DEGREE = 3
PAUSE_FRAC = 0.15
STRIDE = 4
TAIL = 3


def first_event(window, min_seg, h_max):
    grip = window[:, 6]
    speed = np.linalg.norm(window[:, :6], axis=1)
    med = np.median(speed) + 1e-9
    for k in range(min_seg, min(h_max, len(window))):
        if grip[k] != grip[k - 1]:
            return k, "gripper"
        if speed[k] < PAUSE_FRAC * med and speed[k - 1] < PAUSE_FRAC * med:
            return k, "pause"
    return min(h_max, len(window)), "cap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_eps", type=int, default=300)
    args = ap.parse_args()

    for (lo, hm) in ((6, 24), (4, 16), (8, 32)):
        durs, types = [], []
        for ep in episode_indices(args.max_eps):
            _, _, action = episode_arrays(ep)
            for a0 in range(0, len(action) - lo, STRIDE):
                T, ty = first_event(action[a0:a0 + hm], lo, hm)
                durs.append(T); types.append(ty)
        durs = np.array(durs); types = np.array(types)
        gf = (types == "gripper").mean(); pf = (types == "pause").mean()
        print(f"[seg lo={lo} cap={hm}] n={len(durs)}  T mean {durs.mean():.1f} "
              f"CoV {durs.std()/durs.mean():.2f}  gripper {gf*100:.0f}%  "
              f"pause {pf*100:.0f}%  cap {(types=='cap').mean()*100:.0f}%")

    # fit quality at the LIBERO-promoted settings (lo=8, cap=24 analog -> use lo=6 cap=24 too)
    for n_ctrl in (6, 8):
        lo, hm = max(n_ctrl, 6), 24
        B_cache = {}
        tail_e, body_e, sp_e, tog_demo, tog_rec = [], [], [], [], []
        for ep in episode_indices(args.max_eps):
            _, _, action = episode_arrays(ep)
            for a0 in range(0, len(action) - lo, STRIDE):
                win = action[a0:a0 + hm]
                T, ty = first_event(win, lo, hm)
                if T < lo + TAIL or a0 + T > len(action):
                    continue
                seg = win[:T, :6]
                path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg, axis=0)], axis=0)
                if T not in B_cache:
                    u = np.arange(T + 1) / T
                    B = basis_matrix(u, n_ctrl)
                    B_cache[T] = (B, np.linalg.pinv(B[:, 1:n_ctrl - 1]), B[:, n_ctrl - 1:n_ctrl])
                B, pinv_mid, b_last = B_cache[T]
                p_end = path[-1:, :]
                c_mid = pinv_mid @ (path - b_last @ p_end)
                c = np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0)
                rec = np.diff(B @ c, axis=0)
                sp_demo = np.linalg.norm(seg, axis=1)
                scale = max(sp_demo.mean(), 1e-6)
                err = np.linalg.norm(rec - seg, axis=1) / scale
                tail_e.append(err[-TAIL:].mean()); body_e.append(err[:-TAIL].mean())
                sp_rec = np.linalg.norm(rec, axis=1)
                sp_e.append(np.abs(sp_rec - sp_demo).mean() / scale)
                if ty == "gripper":
                    tog_demo.append(sp_demo[-TAIL:].mean() / scale)
                    tog_rec.append(sp_rec[-TAIL:].mean() / max(sp_rec.mean(), 1e-9))
        print(f"[fit n{n_ctrl}] tail {np.mean(tail_e):.4f}  body {np.mean(body_e):.4f}  "
              f"speed-profile {np.mean(sp_e):.4f}  toggle demo {np.mean(tog_demo):.3f} "
              f"vs recon {np.mean(tog_rec):.3f}")


if __name__ == "__main__":
    main()
