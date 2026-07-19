"""Dead-time census of public real-robot teleop datasets (REAL_WORLD_PLAN
zero-hardware step): rank candidate tasks by PREDICTED retiming gain.

The CALVIN mechanism says retiming gains come from compressible dead time
(pauses/hesitation in human demos). For each dataset: download only the action
parquets (no video), compute
  - pause fraction (steps with speed < pause_frac x median, sustained)
  - dead-time fraction of wall-clock
  - speed CoV (heterogeneity -> selective-retiming headroom)
  - jitter floor (spline-denoiser relevance)
Prediction: higher dead-time fraction => bigger uniform-retiming win.

  python deadtime_census.py --repos lerobot/aloha_static_coffee ...
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def load_actions(repo: str, cache: str):
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, repo_type="dataset", cache_dir=cache,
                             allow_patterns=["data/**", "meta/**"])
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    import pandas as pd

    eps = []
    for f in files[:200]:
        df = pd.read_parquet(f)
        if "action" not in df.columns:
            return []
        key = "episode_index" if "episode_index" in df.columns else None
        if key:
            for _, g in df.groupby(key):
                eps.append(np.stack(g["action"].to_numpy()))
        else:
            eps.append(np.stack(df["action"].to_numpy()))
        if len(eps) >= 300:
            break
    return eps


def census(eps, pause_frac=0.15):
    stats = []
    for a in eps:
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or len(a) < 20:
            continue
        # pose-motion columns: use all but assume last col is gripper-ish;
        # robust to arm dims by taking the first min(6, D-1) columns
        d = min(6, a.shape[1] - 1) if a.shape[1] > 1 else a.shape[1]
        # deltas if actions look absolute (positions): difference if the
        # per-step delta magnitude is far smaller than value magnitude
        v = a[:, :d]
        step = np.diff(v, axis=0)
        if np.median(np.abs(v)) > 5 * np.median(np.abs(step)) + 1e-9:
            sp = np.linalg.norm(step, axis=1)          # absolute-position actions
        else:
            sp = np.linalg.norm(v, axis=1)[1:]         # delta actions
        med = np.median(sp[sp > 1e-9]) if (sp > 1e-9).any() else 0
        if med == 0:
            continue
        low = sp < pause_frac * med
        pause = (low[1:] & low[:-1])
        dd = np.diff(step, axis=0)
        stats.append({
            "dead_frac": float(pause.mean()),
            "cov": float(sp.std() / sp.mean()),
            "jitter": float(0.5 * np.linalg.norm(dd, axis=1).mean() / (med + 1e-12)),
            "len": len(a),
        })
    if not stats:
        return None
    agg = {k: float(np.mean([s[k] for s in stats])) for k in ("dead_frac", "cov", "jitter")}
    agg["n_eps"] = len(stats)
    agg["mean_len"] = float(np.mean([s["len"] for s in stats]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--cache", default=os.path.expanduser("~/scratch/vla_bspline/census_cache"))
    args = ap.parse_args()

    print(f"{'dataset':42s} {'eps':>4} {'len':>6} {'dead%':>6} {'CoV':>5} {'jitter':>6}  prediction")
    rows = []
    for repo in args.repos:
        try:
            eps = load_actions(repo, args.cache)
            r = census(eps)
        except Exception as e:
            print(f"{repo:42s}  FAILED: {str(e)[:60]}")
            continue
        if r is None:
            print(f"{repo:42s}  no usable action data")
            continue
        rows.append((repo, r))
        pred = "HIGH retiming gain" if r["dead_frac"] > 0.12 else \
               ("moderate" if r["dead_frac"] > 0.05 else "low (little dead time)")
        print(f"{repo:42s} {r['n_eps']:>4} {r['mean_len']:>6.0f} {100*r['dead_frac']:>6.1f} "
              f"{r['cov']:>5.2f} {r['jitter']:>6.2f}  {pred}")
    print("\nreference points: LIBERO (scripted) dead≈0%, CALVIN (teleop) pauses 0% by "
          "flag but jitter 6x LIBERO; RoboCasa pauses 16-20% -> CALVIN/RoboCasa regime "
          "= where retiming won.")


if __name__ == "__main__":
    main()
