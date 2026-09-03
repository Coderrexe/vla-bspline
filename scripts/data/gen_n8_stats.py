"""v1 ctrl-pt normalization stats for n_ctrl=8 (spatial-gap ablation).
Same pinned-LSQ convention as the policy; horizon 20. -> spline_stats_libero_n8.json
Run: python gen_n8_stats.py --pkg <deployed smolvla_spline dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lerobot_io import episode_arrays, episode_indices

DEGREE, N_CTRL, H = 3, 8, 20


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
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--max_eps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    B_PATH = basis_matrix(np.arange(H + 1) / H, N_CTRL)
    PINV_MID = np.linalg.pinv(B_PATH[:, 1:N_CTRL - 1])
    B_LAST = B_PATH[:, N_CTRL - 1:N_CTRL]
    PINV_GRIP = np.linalg.pinv(basis_matrix(np.arange(H) / (H - 1), N_CTRL))

    ctrl_all, grip_all = [], []
    for ep in episode_indices()[: args.max_eps]:
        _, _, action = episode_arrays(int(ep))
        for a0 in range(0, len(action) - H, args.stride):
            w = action[a0:a0 + H]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(w[:, :6], axis=0)], axis=0)
            p_end = path[-1:, :]
            c_mid = PINV_MID @ (path - B_LAST @ p_end)
            ctrl_all.append(np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0))
            grip_all.append(PINV_GRIP @ w[:, 6])
    ctrl_all = np.stack(ctrl_all); grip_all = np.stack(grip_all)
    out = {
        "stats_contract_version": 1, "action_layout": "eef7",
        "pose_dims": list(range(6)), "grip_idx": 6, "pass_dims": [],
        "boundary_semantics": "fixed_horizon_actions_0_to_h_minus_1",
        "n_ctrl": N_CTRL, "horizon": H, "degree": DEGREE,
        "pose_ctrl_mean": ctrl_all.mean(0).tolist(),
        "pose_ctrl_std": np.maximum(ctrl_all.std(0), 1e-4).tolist(),
        "grip_ctrl_mean": grip_all.mean(0).tolist(),
        "grip_ctrl_std": np.maximum(grip_all.std(0), 1e-4).tolist(),
        "n_windows": len(ctrl_all),
    }
    p = os.path.join(args.pkg, "spline_stats_libero_n8.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p} ({len(ctrl_all)} windows)")


if __name__ == "__main__":
    main()
