"""Generate normalization stats for the speed-heterogeneous-demos study.

Outputs (into smolvla_spline_pkg/ == the deployed policy dir):
  1. waypoint_action_stats_libero.json      — COPY of the dataset's per-dim action
     mean/std (must equal what the training preprocessor uses, so the speedaug
     policy's un/re-normalization matches it exactly).
  2. spline_stats_libero_saug.json          — B-arm ctrl-pt stats: v1 pinned LSQ
     fits of speed-augmented (s = SPEEDS[episode_index % 3]) 20-step chunks.
  3. spline_stats_libero_v2_h24_saug.json   — C-arm: pose/grip stats copied from
     v2_h24 (shape targets are UNCHANGED by design); logT stats shifted
     analytically: logT' = logT + log s, s independent of T
     -> mean += E[log s], std = sqrt(std^2 + Var[log s]).

Run on the cluster: python gen_saug_stats.py --pkg <path-to-smolvla_spline-dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lerobot_io import episode_arrays, episode_indices

SPEEDS = [1.0, 1.5, 2.0]
DEGREE, N_CTRL, H = 3, 6, 20


def basis_matrix(u, n_ctrl):
    from scipy.interpolate import BSpline
    kn = np.concatenate([np.zeros(DEGREE), np.linspace(0, 1, n_ctrl - DEGREE + 1), np.ones(DEGREE)])
    B = np.zeros((len(u), n_ctrl))
    for j in range(n_ctrl):
        c = np.zeros(n_ctrl); c[j] = 1.0
        B[:, j] = BSpline(kn, c, DEGREE)(u)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="deployed smolvla_spline package dir")
    ap.add_argument("--max_eps", type=int, default=400)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    # ---- 1. dataset action stats (from lerobot metadata = preprocessor's source)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero")
    a_mean = np.asarray(meta.stats["action"]["mean"], dtype=np.float64).ravel()
    a_std = np.asarray(meta.stats["action"]["std"], dtype=np.float64).ravel()
    out1 = {"mean": a_mean.tolist(), "std": a_std.tolist(),
            "source": "LeRobotDatasetMetadata('HuggingFaceVLA/libero').stats['action']"}
    p1 = os.path.join(args.pkg, "waypoint_action_stats_libero.json")
    json.dump(out1, open(p1, "w"), indent=1)
    print("wrote", p1, "mean[:3] =", np.round(a_mean[:3], 5).tolist())

    # ---- 2. B-arm speed-augmented v1 ctrl-pt stats
    B_PATH = basis_matrix(np.arange(H + 1) / H, N_CTRL)
    PINV_MID = np.linalg.pinv(B_PATH[:, 1:N_CTRL - 1])
    B_LAST = B_PATH[:, N_CTRL - 1:N_CTRL]
    PINV_GRIP = np.linalg.pinv(basis_matrix(np.arange(H) / (H - 1), N_CTRL))

    ctrl_all, grip_all, n_win = [], [], 0
    for ep in episode_indices()[: args.max_eps]:
        _, _, action = episode_arrays(int(ep))
        s = SPEEDS[int(ep) % len(SPEEDS)]
        for a0 in range(0, len(action) - H, args.stride):
            w = action[a0:a0 + H]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(w[:, :6], axis=0)], axis=0)
            # synthetic slow path: sample cumulative path at raw index j/s
            x = np.arange(H + 1) / s
            syn = np.stack([np.interp(x, np.arange(H + 1), path[:, d]) for d in range(6)], axis=1)
            gx = np.clip(((np.arange(H) + 0.5) / s).astype(int), 0, H - 1)
            g = w[gx, 6]
            p_end = syn[-1:, :]
            c_mid = PINV_MID @ (syn - B_LAST @ p_end)
            ctrl_all.append(np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0))
            grip_all.append(PINV_GRIP @ g)
            n_win += 1
    ctrl_all = np.stack(ctrl_all); grip_all = np.stack(grip_all)
    out2 = {
        "n_ctrl": N_CTRL, "horizon": H, "degree": DEGREE,
        "pose_ctrl_mean": ctrl_all.mean(0).tolist(),
        "pose_ctrl_std": np.maximum(ctrl_all.std(0), 1e-4).tolist(),
        "grip_ctrl_mean": grip_all.mean(0).tolist(),
        "grip_ctrl_std": np.maximum(grip_all.std(0), 1e-4).tolist(),
        "n_windows": n_win, "speed_aug": SPEEDS,
    }
    p2 = os.path.join(args.pkg, "spline_stats_libero_saug.json")
    json.dump(out2, open(p2, "w"), indent=1)
    print(f"wrote {p2}  ({n_win} windows, pose std range "
          f"{ctrl_all.std(0).min():.4f}..{ctrl_all.std(0).max():.4f})")

    # ---- 3. C-arm: copy v2_h24, shift logT stats analytically
    src = json.load(open(os.path.join(args.pkg, "spline_stats_libero_v2_h24.json")))
    logs = np.log(SPEEDS)
    src["logT_mean"] = float(src["logT_mean"] + logs.mean())
    src["logT_std"] = float(np.sqrt(src["logT_std"] ** 2 + logs.var()))
    src["speed_aug"] = SPEEDS
    src["note"] = "pose/grip stats identical to v2_h24 (shape targets unchanged); logT shifted for s-scaled durations"
    p3 = os.path.join(args.pkg, "spline_stats_libero_v2_h24_saug.json")
    json.dump(src, open(p3, "w"), indent=1)
    print("wrote", p3, f"logT mean/std -> {src['logT_mean']:.4f}/{src['logT_std']:.4f}")


if __name__ == "__main__":
    main()
