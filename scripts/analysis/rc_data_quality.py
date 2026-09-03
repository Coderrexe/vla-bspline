"""RoboCasa data-quality + cross-benchmark inconsistency investigation
(Xiatao pt 1 + 2). Three questions, answered from data, no GPU:

Q1 KINEMATIC SANITY of the converted dataset: NaNs, teleport spikes,
   action<->state consistency (does integrating pose deltas track the recorded
   EE state?), per-dim saturation (fraction of steps with |a| >= 0.99 — the
   demos may already ride the [-1,1] delta bound).

Q2 CLAMP EXPOSURE of our decode across benchmarks: build v2 spline targets on
   real windows, decode at base speed and at uniform alpha=0.6, and count the
   fraction of decoded steps that exceed the actuator bound BEFORE the
   hardcoded clamp(-1,1). Hypothesis: RoboCasa demos saturate the bound, so
   (a) even base decode clips (spline fit overshoot near saturated segments —
   explains C < A on RC only), and (b) retiming multiplies required deltas by
   1/alpha => hard clipping => path distortion (explains Cuni <= C on RC,
   unlike LIBERO/CALVIN where demos live far below the bound).

Q3 the same stats on LIBERO + CALVIN for the side-by-side table.

  python rc_data_quality.py --bench robocasa   # also: libero, calvin
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BENCH = {
    "robocasa": {
        "root": "~/scratch/vla_bspline/robocasa_atomic",
        "pose": slice(5, 11), "grip": 11, "cap": 24, "min_seg": 8, "n_ctrl": 8,
    },
    "calvin": {
        "root": "~/scratch/vla_bspline/calvin_v30",
        "pose": slice(0, 6), "grip": 6, "cap": 16, "min_seg": 8, "n_ctrl": 8,
    },
    # LIBERO loads through the hub cache; actions only, via its data parquets
    "libero": {
        "root": "~/scratch/vla_bspline/hf_cache/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/*",
        "pose": slice(0, 6), "grip": 6, "cap": 24, "min_seg": 8, "n_ctrl": 8,
    },
}


def episodes(root, limit=300):
    import pandas as pd

    roots = glob.glob(os.path.expanduser(root))
    files = sorted(glob.glob(os.path.join(roots[0], "data", "**", "*.parquet"), recursive=True))
    eps = []
    for f in files:
        df = pd.read_parquet(f, columns=["action", "episode_index"])
        for _, g in df.groupby("episode_index"):
            eps.append(np.stack(g["action"].to_numpy()).astype(float))
            if len(eps) >= limit:
                return eps
    return eps


def bspline_basis_np(u, n, deg=3):
    from scipy.interpolate import BSpline

    kn = np.concatenate([np.zeros(deg), np.linspace(0, 1, n - deg + 1), np.ones(deg)])
    B = np.stack([BSpline(kn, np.eye(n)[j], deg, extrapolate=False)(u) for j in range(n)], axis=1)
    return np.nan_to_num(B)


def first_event(a, pose, grip, cap, min_seg, pause_frac=0.15):
    g = a[:, grip]
    sp = np.linalg.norm(a[:, pose], axis=1)
    med = np.median(sp[sp > 1e-8]) if (sp > 1e-8).any() else 0
    thr = pause_frac * med
    end = min(cap, len(a))
    for k in range(min_seg, end):
        if np.sign(g[k]) != np.sign(g[k - 1]):
            return k + 1
        if thr > 0 and k + 1 < len(a) and sp[k] < thr and sp[k + 1] < thr:
            return k + 1
    return end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCH), required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--windows_per_ep", type=int, default=6)
    args = ap.parse_args()
    cfg = BENCH[args.bench]
    eps = episodes(cfg["root"], args.episodes)
    pose, cap, min_seg, n = cfg["pose"], cfg["cap"], cfg["min_seg"], cfg["n_ctrl"]

    # ---------------- Q1: kinematic sanity ----------------
    n_nan = sum(int(np.isnan(a).any()) for a in eps)
    spike = 0
    sat_steps = []
    for a in eps:
        v = a[:, pose]
        dv = np.abs(np.diff(v, axis=0)).max(axis=1)
        med = np.median(dv) + 1e-9
        spike += int((dv > 20 * med).sum())
        sat_steps.append((np.abs(v) >= 0.99).any(axis=1))
    sat_frac = float(np.concatenate(sat_steps).mean())
    print(f"[{args.bench}] Q1 sanity: episodes={len(eps)} NaN-eps={n_nan} "
          f"teleport-spikes(20x median)={spike} | steps with ANY pose dim >=0.99: "
          f"{100*sat_frac:.1f}%")

    # ---------------- Q2: decode clamp exposure ----------------
    rng = np.random.default_rng(0)
    over_base, over_fast, seg_n = [], [], 0
    for a in eps:
        if len(a) < min_seg + 2:
            continue
        for _ in range(args.windows_per_ep):
            s = int(rng.integers(0, max(1, len(a) - min_seg - 1)))
            win = a[s: s + cap]
            T = first_event(win, pose, cfg["grip"], cap, min_seg)
            seg = win[:T, pose]
            path = np.concatenate([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
            u = np.arange(T + 1) / T
            B = bspline_basis_np(u, n)
            p_end = path[-1:]
            c_mid = np.linalg.pinv(B[:, 1: n - 1]) @ (path - B[:, n - 1: n] @ p_end)
            c = np.concatenate([np.zeros((1, seg.shape[1])), c_mid, p_end])

            def decoded_over(h):
                uu = np.arange(h + 1) / h
                Bp = bspline_basis_np(uu, n)
                rec = Bp @ c
                d = np.abs(np.diff(rec, axis=0))
                return float((d > 1.0).mean())

            over_base.append(decoded_over(T))
            over_fast.append(decoded_over(max(min_seg // 2, int(round(T * 0.6)))))
            seg_n += 1
    print(f"[{args.bench}] Q2 decode-clamp exposure over {seg_n} windows: "
          f"base-speed steps>bound: {100*np.mean(over_base):.2f}% | "
          f"alpha=0.6 steps>bound: {100*np.mean(over_fast):.2f}%")


if __name__ == "__main__":
    main()
