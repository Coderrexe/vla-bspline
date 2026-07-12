"""
Port of Quinten's CALVIN `dataproccess.py` to the LIBERO / LeRobot dataset.

Produces per-chunk B-spline regression targets from LIBERO demos, to later supervise
a VLA action head. For each anchor frame t in each episode we take the future chunk
state[t : t+H], fit a cubic B-spline to the absolute eef pose (pos xyz + axis-angle),
and record the control points + chunk duration + gripper sub-sequence.

Key differences from the CALVIN version:
  - fits ABSOLUTE eef state (obs.state[:6]), not delta actions
  - uses REAL timestamps -> real chunk duration (seconds)
  - keeps (episode_index, anchor_frame) keys so targets JOIN back to the LeRobot
    dataset's images/state without duplicating them
  - gripper kept raw (bang-bang), stored as the raw sub-sequence

TIME ALLOCATION (the project's novelty) — NOT yet variable here:
  This uniform-knot version gives duration_s = H/fps = CONSTANT, so it is only a
  baseline (== the fixed-time B-spline VLA prior work). Making duration a meaningful
  learnable target needs either variable-length semantic chunks or non-uniform knot
  optimization; that is the next design iteration. Schema already carries duration_s
  and knots so the extension is drop-in.

Output: outputs/bspline_targets_H{H}_nc{n}.parquet  (+ printed QC summary)
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from lerobot_io import episode_indices, episode_arrays, load_info
from bspline_core import fit_chunk, chunk_rmse, FIT_DIMS, GRIP_STATE_DIMS


def build(H: int, n_ctrl: int, stride: int, max_episodes: int | None):
    info = load_info()
    fps = float(info["fps"])
    eps = episode_indices()
    if max_episodes:
        eps = eps[:max_episodes]

    rows = []
    pos_rmses, rot_rmses = [], []
    n_short = 0
    for ep in eps:
        t, state, action = episode_arrays(int(ep))
        L = len(t)
        for anchor in range(0, L - H + 1, stride):
            chunk = state[anchor:anchor + H]
            tc = t[anchor:anchor + H]
            duration_s = float(tc[-1] - tc[0])
            try:
                fit = fit_chunk(chunk, n_ctrl, FIT_DIMS)
            except ValueError:
                n_short += 1
                continue
            err = chunk_rmse(fit["recon"], chunk, FIT_DIMS)
            pos_rmses.append(err["pos_rmse"]); rot_rmses.append(err["rot_rmse"])
            grip = state[anchor:anchor + H][:, list(GRIP_STATE_DIMS)]  # (H, 2) raw
            rows.append({
                "episode_index": int(ep),
                "anchor_frame": int(anchor),
                "horizon_frames": H,
                "duration_s": duration_s,
                "n_ctrl": n_ctrl,
                "control_points": fit["control_points"].astype(np.float32).ravel(),  # (6*n_ctrl,)
                "gripper_state": grip.astype(np.float32).ravel(),                     # (H*2,)
                "gripper_action": action[anchor:anchor + H, 6].astype(np.float32),    # (H,) bang-bang
                "pos_rmse_m": err["pos_rmse"],
                "rot_rmse_rad": err["rot_rmse"],
            })

    df = pd.DataFrame(rows)
    out = f"outputs/bspline_targets_H{H}_nc{n_ctrl}.parquet"
    df.to_parquet(out)

    print(f"\n=== B-spline target set built ===")
    print(f"episodes: {len(eps)} | chunks: {len(df)} | H={H} ({H/fps:.1f}s) | "
          f"n_ctrl={n_ctrl} | stride={stride} | skipped-short={n_short}")
    print(f"control_points per chunk: 6 dims x {n_ctrl} = {6*n_ctrl} floats")
    print(f"reconstruction RMSE  pos: mean={np.mean(pos_rmses)*1000:.2f}mm "
          f"p95={np.percentile(pos_rmses,95)*1000:.2f}mm | "
          f"rot: mean={np.mean(rot_rmses)*1000:.2f}mrad "
          f"p95={np.percentile(rot_rmses,95)*1000:.2f}mrad")
    print(f"duration_s: {df['duration_s'].min():.3f}..{df['duration_s'].max():.3f} "
          f"(constant => uniform-knot baseline; see module docstring on time allocation)")
    print(f"saved -> {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20, help="chunk horizon in frames")
    ap.add_argument("--n_ctrl", type=int, default=6, help="control points per dim (cubic: >=4)")
    ap.add_argument("--stride", type=int, default=5, help="anchor stride in frames")
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()
    build(args.H, args.n_ctrl, args.stride, args.max_episodes)
