"""RoboCasa365 offline gate + v2 stats generation (one pass, no GPU).

Reads the local robocasa365-target-atomic parquet directly (action[12]:
base 0-3, mode 4, EE pose 5-10, grip 11; fps 20) and, per candidate config
(n_ctrl, cap):
  - v2 event-segmentation stats (toggle/pause/cap fractions, T distribution,
    logT sigma = the duration-signal strength that picked CALVIN's config)
  - pinned-LSQ fit error vs the demo jitter floor (the denoiser check)
Then writes the spline_stats JSON for the chosen config (pose/grip/pass ctrl
stats + logT moments) for training.

  python robocasa_prep.py --root ~/scratch/vla_bspline/robocasa_atomic \
      --grid "6:24,8:24,8:16" --episodes 400
  python robocasa_prep.py --root ... --emit 8:24 \
      --out spline_stats_robocasa_v2_n8h24.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

POSE = slice(5, 11)
GRIP = 11
PASS = [0, 1, 2, 3, 4]
MIN_SEG_FRAC = 1 / 3  # min_seg = cap * frac, matching LIBERO/CALVIN ratios


def load_episodes(root, limit):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    import pandas as pd

    eps = []
    for f in files:
        df = pd.read_parquet(f, columns=["action", "episode_index"])
        for ep, g in df.groupby("episode_index"):
            eps.append(np.stack(g["action"].to_numpy()))
            if limit and len(eps) >= limit:
                return eps
    return eps


def segment(a, cap, min_seg, pause_frac=0.15):
    grip = a[:, GRIP]
    speed = np.linalg.norm(a[:, POSE], axis=1)
    med = np.median(speed[speed > 1e-8]) if (speed > 1e-8).any() else 0.0
    thr = pause_frac * med
    segs, start, L = [], 0, len(a)
    while start < L:
        end = min(start + cap, L)
        cut, kind = end, "cap"
        for k in range(start + min_seg, end):
            if np.sign(grip[k]) != np.sign(grip[k - 1]):
                cut, kind = k + 1, "toggle"
                break
            if thr > 0 and k + 1 < L and speed[k] < thr and speed[k + 1] < thr:
                cut, kind = k + 1, "pause"
                break
        segs.append((start, cut, kind))
        start = cut
    return segs


def bspline_basis_np(u, n, deg=3):
    from scipy.interpolate import BSpline

    kn = np.concatenate([np.zeros(deg), np.linspace(0, 1, n - deg + 1), np.ones(deg)])
    B = np.stack([BSpline(kn, np.eye(n)[j], deg, extrapolate=False)(u) for j in range(n)], axis=1)
    return np.nan_to_num(B)


def window_segments(a, cap, min_seg, stride=8, pause_frac=0.15):
    """Sample fixed-cap windows at strided starts and cut each at its first
    event — mirrors the TRAINING dataloader (windows begin at arbitrary
    mid-motion phases), unlike contiguous episode segmentation. Stats computed
    contiguously under-scale mid-motion control points (episode-start segments
    begin from rest) — that mismatch blew training loss to 18."""
    grip = a[:, GRIP]
    speed = np.linalg.norm(a[:, POSE], axis=1)
    med = np.median(speed[speed > 1e-8]) if (speed > 1e-8).any() else 0.0
    thr = pause_frac * med
    out = []
    for s in range(0, max(1, len(a) - min_seg), stride):
        end = min(s + cap, len(a))
        cut, kind = end, "cap"
        for k in range(s + min_seg, end):
            if np.sign(grip[k]) != np.sign(grip[k - 1]):
                cut, kind = k + 1, "toggle"
                break
            if thr > 0 and k + 1 < len(a) and speed[k] < thr and speed[k + 1] < thr:
                cut, kind = k + 1, "pause"
                break
        if cut - s >= min_seg:
            out.append((s, cut, kind))
    return out


def fit_stats(eps, n_ctrl, cap, min_seg, collect=False):
    kinds, Ts, errs, jitter = [], [], [], []
    cp, cg, cpas, logT = [], [], [], []
    seg_fn = window_segments if collect else segment
    for a in eps:
        for (s, e, kind) in seg_fn(a, cap, min_seg):
            T = e - s
            if T < min_seg:
                continue
            kinds.append(kind)
            Ts.append(T)
            seg_a = a[s:e]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(seg_a[:, POSE], axis=0)], axis=0)
            u = np.arange(T + 1) / T
            B = bspline_basis_np(u, n_ctrl)
            p_end = path[-1:, :]
            resid = path - B[:, n_ctrl - 1: n_ctrl] @ p_end
            c_mid = np.linalg.pinv(B[:, 1: n_ctrl - 1]) @ resid
            c = np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0)
            rec = B @ c
            step_err = np.linalg.norm(np.diff(rec, axis=0) - seg_a[:, POSE], axis=1)
            errs.append(step_err.mean())
            # jitter floor: half the mean step-to-step delta change
            dd = np.diff(seg_a[:, POSE], axis=0)
            if len(dd):
                jitter.append(0.5 * np.linalg.norm(np.diff(dd, axis=0), axis=1).mean()
                              if len(dd) > 1 else 0.0)
            if collect:
                ug = np.arange(T) / max(T - 1, 1)
                Bg = bspline_basis_np(ug, n_ctrl)
                pg = np.linalg.pinv(Bg)
                cp.append(c)
                cg.append(pg @ seg_a[:, GRIP])
                cpas.append(pg @ seg_a[:, PASS])
                logT.append(np.log(T))
    kinds = np.array(kinds)
    out = {
        "n_seg": len(kinds),
        "toggle_frac": float((kinds == "toggle").mean()),
        "pause_frac": float((kinds == "pause").mean()),
        "cap_frac": float((kinds == "cap").mean()),
        "T_mean": float(np.mean(Ts)), "T_cov": float(np.std(Ts) / np.mean(Ts)),
        "logT_sigma": float(np.std(np.log(Ts))),
        "fit_err": float(np.mean(errs)), "jitter_floor": float(np.mean(jitter)),
        "density": n_ctrl / np.mean(Ts),
    }
    if collect:
        cp = np.stack(cp); cg = np.stack(cg); cpas = np.stack(cpas)
        out["stats_json"] = {
            "n_ctrl": n_ctrl, "horizon_max": cap, "min_seg": min_seg,
            "pose_ctrl_mean": cp.mean(0).tolist(), "pose_ctrl_std": (cp.std(0) + 1e-6).tolist(),
            "grip_ctrl_mean": cg.mean(0).tolist(), "grip_ctrl_std": (cg.std(0) + 1e-6).tolist(),
            "pass_ctrl_mean": cpas.mean(0).tolist(), "pass_ctrl_std": (cpas.std(0) + 1e-6).tolist(),
            "logT_mean": float(np.mean(logT)), "logT_std": float(np.std(logT) + 1e-6),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--grid", default="6:24,8:24,8:16")
    ap.add_argument("--emit", default=None, help="n:cap — write stats JSON for this config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eps = load_episodes(os.path.expanduser(args.root), args.episodes)
    print(f"loaded {len(eps)} episodes; mean len {np.mean([len(e) for e in eps]):.1f} steps")

    if args.emit:
        n, cap = (int(x) for x in args.emit.split(":"))
        min_seg = max(4, round(cap * MIN_SEG_FRAC))
        r = fit_stats(eps, n, cap, min_seg, collect=True)
        sj = r.pop("stats_json")
        print(json.dumps(r, indent=1))
        out = args.out or f"spline_stats_robocasa_v2_n{n}h{cap}.json"
        with open(out, "w") as f:
            json.dump(sj, f)
        print("WROTE", out)
        return

    print(f"{'cfg':>8} {'segs':>6} {'tog%':>6} {'pau%':>6} {'cap%':>6} "
          f"{'T_mean':>7} {'logTsd':>7} {'fit':>8} {'jitter':>8} {'dens':>5}")
    for spec in args.grid.split(","):
        n, cap = (int(x) for x in spec.split(":"))
        min_seg = max(4, round(cap * MIN_SEG_FRAC))
        r = fit_stats(eps, n, cap, min_seg)
        print(f"{spec:>8} {r['n_seg']:>6} {100*r['toggle_frac']:>6.1f} {100*r['pause_frac']:>6.1f} "
              f"{100*r['cap_frac']:>6.1f} {r['T_mean']:>7.1f} {r['logT_sigma']:>7.3f} "
              f"{r['fit_err']:>8.4f} {r['jitter_floor']:>8.4f} {r['density']:>5.2f}")


if __name__ == "__main__":
    main()
