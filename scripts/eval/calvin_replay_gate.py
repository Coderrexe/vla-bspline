"""CALVIN action-replay semantics gate — MUST pass before any policy eval.

The training data (fywang conversion) is 10 fps; native CALVIN runs 30 Hz. This
gate replays a converted episode's actions in the simulator and measures whether
the scene evolves as recorded, deciding the execution convention:
  A) one converted action per env.step() at the env's native rate
  B) each converted action held for `repeat` env steps (action repeat = 3)

Metric: final EEF position error vs the recorded trajectory endpoint + per-step
tracking RMSE, for both conventions. Pass = one convention tracks closely.

  ~/.conda/envs/calvin/bin/python calvin_replay_gate.py \
      --debug_root ~/scratch/vla_bspline/calvin_debug_dataset/validation \
      --episode 0
Uses the debug dataset's hydra config for env construction (scene, cameras).
The converted episode actions come from a small npz exported beforehand by
calvin_export_episode.py (run in the lerobot env, which can read the parquet).
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import os
import sys

# calvin_env targets py3.8; shim the py3.10 stdlib removals its deps rely on
for _n in ("Mapping", "MutableMapping", "Sequence", "Set", "Iterable", "Callable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
import fractions
import math
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import numpy as np


def make_env(debug_root):
    from omegaconf import OmegaConf
    import hydra

    cfg_path = os.path.join(debug_root, ".hydra", "merged_config.yaml")
    conf = OmegaConf.load(cfg_path)
    env_conf = conf.env
    # headless: no GUI, EGL if available else TinyRenderer
    env_conf["use_egl"] = False
    env_conf["show_gui"] = False
    env_conf["use_vr"] = False
    env_conf["use_scene_info"] = True
    # gate reads only robot_obs; cameras unneeded, and tacto's mp renderer
    # hangs on headless nodes waiting for its GL worker
    env_conf["cameras"] = {}
    env = hydra.utils.instantiate(env_conf)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_root", required=True)
    ap.add_argument("--actions_npz", required=True,
                    help="npz with actions (L,7), rel-convention, from the converted dataset")
    ap.add_argument("--repeats", nargs="+", type=int, default=[1, 3])
    args = ap.parse_args()

    z = np.load(args.actions_npz)
    robot = z["robot_obs"]                              # (L,15) native 30Hz
    env = make_env(args.debug_root)

    def replay(actions, rep, ref_stride, label):
        env.reset(robot_obs=z["robot_obs0"], scene_obs=z["scene_obs0"])
        errs = []
        for k, a in enumerate(actions):
            for _ in range(rep):
                obs, *_ = env.step(a)
            ridx = min((k + 1) * ref_stride, len(robot) - 1)
            eef = np.asarray(obs["robot_obs"][:3]) if isinstance(obs, dict) else \
                np.asarray(env.robot.get_observation()[0][:3])
            errs.append(np.linalg.norm(eef - robot[ridx, :3]))
        errs = np.array(errs)
        print(f"REPLAY {label:>14}: tracking RMSE {errs.mean():.4f} m  "
              f"final {errs[-1]:.4f}  max {errs.max():.4f}", flush=True)

    replay(z["rel30"], 1, 1, "rel30 rep1")               # sanity: native replay
    replay(z["sub10"], 3, 3, "sub10 rep3")               # subsample convention
    replay(z["recomputed10"], 3, 3, "recomp10 rep3")     # delta-recompute convention


if __name__ == "__main__":
    main()
