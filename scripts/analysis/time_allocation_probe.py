"""
Probe the TIME-ALLOCATION design fork (the project's core novelty).

Question: within a fixed-length chunk fit by a cubic B-spline with fixed n_ctrl,
does OPTIMIZING the interior knot positions (= per-segment time allocation, the
FASTER/EGO-Planner idea) beat UNIFORM knot spacing? And does the optimal allocation
VARY across chunks (so a network could learn to predict it)?

For n_ctrl=6, degree=3, clamped: there are 2 interior knots -> 3 time segments
(T1,T2,T3, sum=1). We parametrize the allocation as a 3-way split (softmax) and
minimize reconstruction SSE over the chunk. We report:
  - reconstruction RMSE: uniform vs time-optimized
  - the spread of optimal allocations across chunks (is it a real, varying signal?)
  - whether bigger gains correlate with more non-uniform motion (pause-then-move)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from lerobot_io import episode_indices, episode_arrays
from bspline_core import fit_chunk, chunk_rmse, make_clamped_uniform_knots, DEGREE, FIT_DIMS
from scipy.interpolate import make_lsq_spline, BSpline

N_CTRL = 6
H = 20
N_INTERIOR = N_CTRL - DEGREE - 1   # = 2 for n_ctrl=6


def knots_from_alloc(seg_widths: np.ndarray) -> np.ndarray:
    """3 positive segment widths (sum=1) -> clamped cubic knot vector on [0,1]."""
    interior = np.cumsum(seg_widths)[:-1]   # 2 interior knots
    return np.concatenate([np.zeros(DEGREE), [0.0], interior, [1.0], np.ones(DEGREE)])[:N_CTRL + DEGREE + 1]


def sse_for_alloc(logits, u, chunk):
    w = np.exp(logits - logits.max()); w = np.append(w, 1.0); w = w / w.sum()  # 3 widths
    interior = np.cumsum(w)[:-1]
    # guard Schoenberg-Whitney: keep interior knots strictly inside and separated
    interior = np.clip(interior, 1e-3, 1 - 1e-3)
    knots = np.concatenate([np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)])
    try:
        sse = 0.0
        for d in FIT_DIMS:
            spl = make_lsq_spline(u, chunk[:, d], knots, k=DEGREE)
            sse += np.sum((chunk[:, d] - BSpline(knots, spl.c, DEGREE)(u)) ** 2)
        return sse
    except Exception:
        return 1e9


def optimize_alloc(chunk):
    u = np.linspace(0, 1, len(chunk))
    res = minimize(sse_for_alloc, x0=np.zeros(2), args=(u, chunk), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 200})
    logits = res.x
    w = np.exp(logits - logits.max()); w = np.append(w, 1.0); w = w / w.sum()
    interior = np.clip(np.cumsum(w)[:-1], 1e-3, 1 - 1e-3)
    knots = np.concatenate([np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)])
    recon = np.zeros((len(chunk), len(FIT_DIMS)))
    for i, d in enumerate(FIT_DIMS):
        spl = make_lsq_spline(u, chunk[:, d], knots, k=DEGREE)
        recon[:, i] = BSpline(knots, spl.c, DEGREE)(u)
    return w, chunk_rmse(recon, chunk, FIT_DIMS)


def run(n_chunks=200):
    eps = episode_indices()[:60]
    chunks = []
    for ep in eps:
        _, state, _ = episode_arrays(int(ep))
        for a in range(0, len(state) - H + 1, 15):
            chunks.append(state[a:a + H])
    rng = np.random.RandomState(0)
    idx = rng.choice(len(chunks), size=min(n_chunks, len(chunks)), replace=False)

    uni_pos, opt_pos, allocs, speed_nonunif, gains = [], [], [], [], []
    for i in idx:
        chunk = chunks[i]
        fu = fit_chunk(chunk, N_CTRL, FIT_DIMS)
        eu = chunk_rmse(fu["recon"], chunk, FIT_DIMS)
        w, eo = optimize_alloc(chunk)
        uni_pos.append(eu["pos_rmse"]); opt_pos.append(eo["pos_rmse"]); allocs.append(w)
        # "non-uniformity" of motion: std of per-step speed / mean speed (0=constant speed)
        vel = np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1)
        speed_nonunif.append(vel.std() / (vel.mean() + 1e-9))
        gains.append(1 - eo["pos_rmse"] / (eu["pos_rmse"] + 1e-12))

    uni_pos, opt_pos, allocs = np.array(uni_pos), np.array(opt_pos), np.array(allocs)
    gains, speed_nonunif = np.array(gains), np.array(speed_nonunif)

    print(f"\n=== TIME-ALLOCATION PROBE (H={H}, n_ctrl={N_CTRL}, {len(idx)} chunks) ===")
    print(f"pos RMSE  uniform: mean={uni_pos.mean()*1000:.3f}mm  p90={np.percentile(uni_pos,90)*1000:.3f}mm")
    print(f"pos RMSE  time-opt: mean={opt_pos.mean()*1000:.3f}mm  p90={np.percentile(opt_pos,90)*1000:.3f}mm")
    print(f"mean error reduction: {gains.mean()*100:.1f}%  (p90 chunk: {np.percentile(gains,90)*100:.1f}%)")
    print(f"\nOptimal time allocation (3 segment fractions T1,T2,T3):")
    print(f"  mean   = {allocs.mean(0).round(3)}   (uniform would be [0.333 0.333 0.333])")
    print(f"  std    = {allocs.std(0).round(3)}   <- spread across chunks = the learnable signal")
    r = np.corrcoef(speed_nonunif, gains)[0, 1]
    print(f"\ncorr(motion non-uniformity, error reduction) = {r:.3f}")
    print("  (positive => time-opt helps most exactly when motion is non-uniform, as expected)")


if __name__ == "__main__":
    run()
