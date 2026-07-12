"""v3 offline gate study — learned time-map splines (the completed core premise).

v2 (current): spline fit over TIME-uniform parameter; one scalar duration.
              -> control points must encode geometry AND speed profile;
              -> uniform-time decode provably cannot reproduce the demo's
                 terminal deceleration (measured: policies toggle at 0.88x
                 cruise where demos do 0.555x).
v3 (proposed): shape spline over normalized ARC LENGTH (pure geometry) +
              per-knot-span time allocation tau_1..tau_{n-3} (piecewise-linear
              monotone time map t(u)) — the original proposal's [T_1..T_n].

This study fits BOTH representations to the same event-terminated demo segments
(v2 segmentation rules) and measures, per representation:
  shape      per-step delta RMSE, split tail (last 3 pre-event) vs body
  speed      per-step SPEED-profile RMSE relative to mean speed (the metric v2
             must fail and v3 must win for the redesign to proceed)
  toggle     reconstructed terminal-deceleration ratio vs the demo's
  dwell      segments containing dwells (>=2 zero-arc steps): recon error there

Gate to proceed to training: v3 speed-profile error substantially below v2's
with shape error no worse, and toggle ratio matching demos.

Run (cluster): python v3_timemap_study.py   -> prints table + outputs/v3_study.json
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
TAIL = 3
ARC_EPS = 1e-5


def basis(u, n_ctrl):
    kn = np.concatenate([np.zeros(DEGREE + 1), np.linspace(0, 1, n_ctrl - DEGREE + 1)[1:-1], np.ones(DEGREE + 1)])
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


def fit_pinned(u, path, n_ctrl):
    """Pinned-both-ends LSQ at arbitrary sample parameters u (may repeat)."""
    B = basis(u, n_ctrl)
    p_end = path[-1:, :]
    resid = path - B[:, n_ctrl - 1:n_ctrl] @ p_end
    c_mid = np.linalg.pinv(B[:, 1:n_ctrl - 1]) @ resid
    return np.concatenate([np.zeros((1, path.shape[1])), c_mid, p_end], axis=0)


def decode_at(u_eval, ctrl, n_ctrl):
    return basis(u_eval, n_ctrl) @ ctrl


def v2_recon(seg, n_ctrl):
    """Current representation: time-uniform parameter, uniform-time decode."""
    T = len(seg)
    path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg[:, :6], axis=0)], axis=0)
    u = np.arange(T + 1) / T
    ctrl = fit_pinned(u, path, n_ctrl)
    rec = decode_at(u, ctrl, n_ctrl)
    return np.diff(rec, axis=0)


def v3_recon(seg, n_ctrl):
    """Proposed: arc-length shape + per-span time allocation, time-map decode.
    Returns (recon deltas, tau profile) or None if the segment is arc-degenerate."""
    T = len(seg)
    path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg[:, :6], axis=0)], axis=0)
    darc = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(darc)])
    if s[-1] < ARC_EPS:
        return None                                     # pure-dwell chunk: v2 fallback
    u_k = s / s[-1]                                     # (T+1,) monotone, repeats at dwells
    ctrl = fit_pinned(u_k, path, n_ctrl)

    # time allocation: time spent per interior knot span (piecewise-linear u(t))
    n_span = n_ctrl - DEGREE                            # interior spans of clamped cubic
    edges = np.linspace(0, 1, n_span + 1)
    tau = np.zeros(n_span)
    for k in range(T):                                  # step k covers u in [u_k[k], u_k[k+1]]
        a, b = u_k[k], u_k[k + 1]
        if b - a < 1e-12:                               # dwell step: assign to current span
            j = min(np.searchsorted(edges, a, side="right") - 1, n_span - 1)
            tau[max(j, 0)] += 1.0
            continue
        for j in range(n_span):
            lo, hi = edges[j], edges[j + 1]
            ov = max(0.0, min(b, hi) - max(a, lo))
            tau[j] += ov / (b - a)                      # fraction of this 1-step interval

    # decode: invert the piecewise-linear time map at integer step times
    cum = np.concatenate([[0.0], np.cumsum(tau)])       # cum[j] = time at u=edges[j]
    t_steps = np.arange(T + 1, dtype=float)
    u_eval = np.interp(t_steps, cum, edges)             # monotone inverse
    rec = decode_at(u_eval, ctrl, n_ctrl)
    return np.diff(rec, axis=0), tau


def main():
    metrics = {v: {"shape_tail": [], "shape_body": [], "speed_rel": [],
                   "tog_ratio": [], "dwell_speed_rel": []} for v in ("v2_n8", "v3_n8", "v3_n6")}
    demo_tog, n_seg, n_dwell, n_degen = [], 0, 0, 0

    for ep in episode_indices()[:MAX_EPS]:
        _, _, action = episode_arrays(int(ep))
        L = len(action)
        for a0 in range(0, L - MIN_SEG, STRIDE):
            win = action[a0:a0 + H_MAX]
            T = first_event(win)
            if T < MIN_SEG + TAIL or a0 + T > L:
                continue
            seg = win[:T]
            d_demo = seg[:, :6]
            sp_demo = np.linalg.norm(d_demo, axis=1)
            scale = max(sp_demo.mean(), 1e-6)
            has_dwell = (sp_demo < 1e-6).sum() >= 2
            ended_by_toggle = T < min(H_MAX, len(win)) and win[T, 6] != win[T - 1, 6]
            if ended_by_toggle:
                demo_tog.append(sp_demo[-TAIL:].mean() / scale)
            n_seg += 1
            n_dwell += int(has_dwell)

            recons = {"v2_n8": v2_recon(seg, 8), "v3_n8": None, "v3_n6": None}
            r8 = v3_recon(seg, 8)
            r6 = v3_recon(seg, 6)
            if r8 is None or r6 is None:
                n_degen += 1
                continue
            recons["v3_n8"], recons["v3_n6"] = r8[0], r6[0]

            for name, rec in recons.items():
                err = np.linalg.norm(rec - d_demo, axis=1) / scale
                metrics[name]["shape_tail"].append(err[-TAIL:].mean())
                metrics[name]["shape_body"].append(err[:-TAIL].mean())
                sp_rec = np.linalg.norm(rec, axis=1)
                sperr = np.abs(sp_rec - sp_demo).mean() / scale
                metrics[name]["speed_rel"].append(sperr)
                if ended_by_toggle:
                    metrics[name]["tog_ratio"].append(sp_rec[-TAIL:].mean() / max(sp_rec.mean(), 1e-9))
                if has_dwell:
                    metrics[name]["dwell_speed_rel"].append(sperr)

    out = {"n_segments": n_seg, "n_dwell": n_dwell, "n_arc_degenerate": n_degen,
           "demo_toggle_ratio": float(np.mean(demo_tog))}
    print(f"segments {n_seg} (dwell-containing {n_dwell}, arc-degenerate {n_degen}); "
          f"DEMO terminal-deceleration ratio: {out['demo_toggle_ratio']:.3f}")
    print(f"{'rep':>6} {'shape tail':>11} {'shape body':>11} {'SPEED profile':>14} "
          f"{'toggle ratio':>13} {'dwell speed':>12}")
    for name, m in metrics.items():
        row = {k: (float(np.mean(v)) if v else None) for k, v in m.items()}
        out[name] = row
        print(f"{name:>6} {row['shape_tail']:>11.4f} {row['shape_body']:>11.4f} "
              f"{row['speed_rel']:>14.4f} {row['tog_ratio']:>13.3f} "
              f"{(row['dwell_speed_rel'] if row['dwell_speed_rel'] is not None else float('nan')):>12.4f}")
    with open("outputs/v3_study.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
