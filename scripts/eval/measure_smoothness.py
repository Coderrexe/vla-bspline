"""Jerk / boundary-discontinuity analysis over rollouts recorded by
`lerobot-eval --eval.recording=true` (which saves LeRobotDataset-format parquet
with per-step `action` and `observation.state`).

Usage (on the cluster, any python with pandas+numpy):
  python measure_smoothness.py <recordings_root> <replan_every> <label>

Metrics per episode, aggregated:
  cmd_jerk       mean |second difference of commanded deltas|  (= jerk of the
                 commanded path, since deltas are its first difference)
  exe_jerk       mean |third difference of executed eef position|
  boundary_ratio mean |first difference of deltas| AT replan boundaries
                 divided by the same quantity elsewhere (1.0 = seamless chunks)
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd


def episode_metrics(act: np.ndarray, state: np.ndarray, replan: int):
    d = act[:, :6]                       # commanded deltas
    dd = np.diff(d, axis=0)              # accel of commanded path
    ddd = np.diff(dd, axis=0)            # jerk of commanded path
    pos = state[:, :3]
    exe_jerk = np.abs(np.diff(pos, n=3, axis=0)).mean() if len(pos) > 3 else np.nan

    # replan boundaries: policy re-queried every `replan` steps => discontinuity
    # candidates at k = replan, 2*replan, ... in the delta stream
    step_change = np.linalg.norm(dd, axis=1)         # |Δ deltas| per step
    idx = np.arange(1, len(d))
    at_boundary = (idx % replan) == 0
    b = step_change[at_boundary].mean() if at_boundary.any() else np.nan
    w = step_change[~at_boundary].mean() if (~at_boundary).any() else np.nan
    return {
        "cmd_jerk": float(np.abs(ddd).mean()),
        "exe_jerk": float(exe_jerk),
        "boundary_change": float(b),
        "within_change": float(w),
        "boundary_ratio": float(b / w) if w and w > 0 else np.nan,
        "steps": len(d),
    }


def main(root: str, replan: int, label: str):
    files = sorted(glob.glob(f"{root}/**/data/**/*.parquet", recursive=True))
    if not files:
        files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    assert files, f"no parquet under {root}"
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[{label}] frames: {len(df)} | episodes: {df['episode_index'].nunique()}")

    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        act = np.stack(g["action"].to_numpy()).astype(np.float64)
        state = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        if len(act) < replan * 2 + 3:
            continue
        rows.append(episode_metrics(act, state, replan))
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    agg["n_episodes"] = len(rows)
    agg["label"] = label
    agg["replan_every"] = replan
    print(json.dumps(agg, indent=1))
    return agg


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
