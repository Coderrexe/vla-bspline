"""Offline mechanism study: can end-weighted LSQ and/or end-densified knots fix the
relative-precision deficit near contact — BEFORE spending any training compute?

Setting = C's world: event-terminated segments (contact at u=1 by construction).
For each anchor, T = first gripper-toggle/pause/cap (v2 rules, min_seg=6, cap=24),
fit the segment path with variants and measure per-step residual RELATIVE to local
motion scale, split into [last 3 steps before event] vs [rest of segment].

Variants (n_ctrl=6, cubic, both ends pinned — the production interface):
  U      uniform knots, unweighted           (current C)
  W3/W9  uniform knots, end-weighted LSQ     (weight ramps 1 -> {3,9} over last 25%)
  K15/K2 end-densified knots                 (interior knot warp t' = 1-(1-t)^g, g={1.5,2})
  K2W3   combined
Also n8-U as the "just add capacity" reference.

Run: python fit_variant_study.py   -> prints table + outputs/fit_variant_study.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import BSpline

from lerobot_io import episode_arrays, episode_indices

DEGREE = 3
MIN_SEG, H_MAX, PAUSE_FRAC = 6, 24, 0.15
STRIDE = 5
MAX_EPS = 300
TAIL = 3  # "contact window" = last TAIL steps before the event


def knots_for(n_ctrl, gamma=None):
    interior = np.linspace(0, 1, n_ctrl - DEGREE + 1)[1:-1]
    if gamma is not None:
        interior = 1.0 - (1.0 - interior) ** gamma          # densify toward u=1
    return np.concatenate([np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)])


def basis(u, n_ctrl, kn):
    B = np.zeros((len(u), n_ctrl))
    for j in range(n_ctrl):
        c = np.zeros(n_ctrl); c[j] = 1.0
        B[:, j] = np.nan_to_num(BSpline(kn, c, DEGREE, extrapolate=False)(u))
    return B


def first_event(window):
    grip = window[:, 6]
    speed = np.linalg.norm(window[:, :6], axis=1)
    med = np.median(speed) + 1e-9
    for k in range(MIN_SEG, min(H_MAX, len(window))):
        if grip[k] != grip[k - 1]:
            return k
        if speed[k] < PAUSE_FRAC * med and speed[k - 1] < PAUSE_FRAC * med:
            return k
    return min(H_MAX, len(window))


def make_fitter(n_ctrl, gamma=None, w_end=None):
    """Return fit(path_T) for every T in [MIN_SEG, H_MAX]: pinned WLS operators."""
    kn = knots_for(n_ctrl, gamma)
    ops = {}
    for T in range(MIN_SEG, H_MAX + 1):
        u = np.arange(T + 1) / T
        B = basis(u, n_ctrl, kn)
        w = np.ones(T + 1)
        if w_end is not None:  # ramp 1 -> w_end over the last 25% of the segment
            ramp = np.clip((u - 0.75) / 0.25, 0, 1)
            w = 1.0 + (w_end - 1.0) * ramp
        Bm = B[:, 1:n_ctrl - 1]
        sw = np.sqrt(w)
        # rank-deficiency-safe WLS: c = pinv(sqrt(W) Bm) sqrt(W) r
        M = np.linalg.pinv(sw[:, None] * Bm) * sw[None, :]   # (n-2, T+1)
        ops[T] = (B, M, B[:, n_ctrl - 1:n_ctrl])
    def fit_recon(path):
        T = len(path) - 1
        B, M, b_last = ops[T]
        p_end = path[-1:, :]
        c_mid = M @ (path - b_last @ p_end)
        c = np.concatenate([np.zeros((1, path.shape[1])), c_mid, p_end], axis=0)
        return B @ c
    return fit_recon


def main():
    variants = {
        "U_n6":  make_fitter(6),
        "W3_n6": make_fitter(6, w_end=3),
        "W9_n6": make_fitter(6, w_end=9),
        "K15_n6": make_fitter(6, gamma=1.5),
        "K2_n6": make_fitter(6, gamma=2.0),
        "K2W3_n6": make_fitter(6, gamma=2.0, w_end=3),
        "U_n8":  make_fitter(8),
        "K2W3_n8": make_fitter(8, gamma=2.0, w_end=3),
        "W3_n8": make_fitter(8, w_end=3),
        "W9_n8": make_fitter(8, w_end=9),
    }
    acc = {k: {"tail": [], "body": []} for k in variants}
    n_seg = 0
    for ep in episode_indices()[:MAX_EPS]:
        _, _, action = episode_arrays(int(ep))
        L = len(action)
        for a0 in range(0, L - MIN_SEG, STRIDE):
            win = action[a0:a0 + H_MAX]
            T = first_event(win)
            if T < MIN_SEG + TAIL or a0 + T > L:
                continue
            seg = win[:T, :6]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg, axis=0)], axis=0)
            scale = max(np.linalg.norm(seg, axis=1).mean(), 1e-6)
            for k, fit in variants.items():
                rec = fit(path)
                err = np.linalg.norm(np.diff(rec, axis=0) - seg, axis=1) / scale
                acc[k]["tail"].extend(err[-TAIL:].tolist())
                acc[k]["body"].extend(err[:-TAIL].tolist())
            n_seg += 1

    out = {"n_segments": n_seg, "tail_steps": TAIL, "results": {}}
    print(f"segments: {n_seg}  (tail = last {TAIL} steps before event)")
    print(f"{'variant':>9} {'tail rel-err':>13} {'body rel-err':>13} {'tail/body':>10}")
    for k in variants:
        t, b = float(np.mean(acc[k]["tail"])), float(np.mean(acc[k]["body"]))
        out["results"][k] = {"tail": t, "body": b}
        print(f"{k:>9} {t:>13.4f} {b:>13.4f} {t/b:>10.2f}")
    with open("outputs/fit_variant_study.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
