"""Export one converted-CALVIN episode (actions + states) to npz for the replay
gate, plus the matching native debug-dataset initial state (robot_obs/scene_obs).

The converted dataset has no scene_obs; the debug dataset (native) does. For the
replay gate we pick an episode FROM THE DEBUG SPLIT's frame range so the initial
scene state is known, then compare against the converted trajectory.

Simplest robust pairing: replay the DEBUG dataset's own native actions instead —
we export the native episode at BOTH rates:
  native 30Hz rel_actions (ground truth for convention checking)
  10fps subsample [every 3rd frame] and 10fps delta-recompute variants
to discover which matches the converted data's convention, and to give the
replay gate its reference trajectory.

  ~/.conda/envs/lerobot/bin/python calvin_export_episode.py \
      --debug_root ~/scratch/vla_bspline/calvin_debug_dataset/validation --out ep.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=180)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.debug_root, "episode_*.npz")))[: args.n_frames]
    assert files, f"no episode_*.npz under {args.debug_root}"
    rel, robot, scene = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        rel.append(z["rel_actions"])
        robot.append(z["robot_obs"])
        scene.append(z["scene_obs"])
    rel = np.stack(rel)          # (L, 7) native 30Hz rel actions
    robot = np.stack(robot)      # (L, 15)
    scene = np.stack(scene)

    # 10fps variants over the same wall-clock span
    sub = rel[::3]                                   # naive subsample (drop 2/3)
    # delta-recompute: rel actions from robot_obs at 10fps, rescaled to CALVIN units
    tcp_pos = robot[::3, :3]
    tcp_orn = robot[::3, 3:6]
    grip = np.sign(robot[::3, 14:15] * 2 - 1) if robot.shape[1] > 14 else rel[::3, 6:7]
    dpos = np.diff(tcp_pos, axis=0) * 50.0           # 1/0.02
    dorn = np.diff(tcp_orn, axis=0) * 20.0           # 1/0.05, naive (no unwrap)
    recomputed = np.concatenate([dpos, dorn, rel[3::3, 6:7]], axis=1)

    np.savez(args.out,
             rel30=rel, robot_obs=robot, scene_obs=scene,
             sub10=sub, recomputed10=recomputed,
             robot_obs0=robot[0], scene_obs0=scene[0])
    print(f"exported {len(files)} native frames -> {args.out}; "
          f"rel30 {rel.shape}, sub10 {sub.shape}, recomputed10 {recomputed.shape}")


if __name__ == "__main__":
    main()
