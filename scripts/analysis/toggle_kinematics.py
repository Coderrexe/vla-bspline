"""Gripper-pose coordination probe: does the spline policy toggle its gripper
while the arm is still moving, where demos (and the waypoint policy) slow down?

Measures |pose delta| in a +/-2-step window around each gripper toggle,
normalized by that trajectory's mean |pose delta| ("toggle speed ratio";
<1 = slows down for the toggle, ~1 = toggles at cruise speed).

  python toggle_kinematics.py --npz "<glob1>" "<glob2>" ... [--demos]
Each glob is labeled by its dirname+prefix; --demos adds the dataset reference.
"""
from __future__ import annotations

import argparse
import glob as globmod
import os

import numpy as np


def toggle_ratios(actions):
    """actions: (L, 7) -> list of toggle speed ratios for this trajectory."""
    grip = np.sign(actions[:, 6])
    tog = np.where(np.diff(grip) != 0)[0] + 1
    speed = np.linalg.norm(actions[:, :6], axis=1)
    base = speed.mean() + 1e-9
    out = []
    for t in tog:
        lo, hi = max(t - 2, 0), min(t + 3, len(speed))
        out.append(speed[lo:hi].mean() / base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--demos", action="store_true")
    ap.add_argument("--max_eps", type=int, default=200)
    args = ap.parse_args()

    if args.demos:
        from lerobot_io import episode_arrays, episode_indices
        ratios, n_tog = [], 0
        for ep in episode_indices()[: args.max_eps]:
            _, _, action = episode_arrays(int(ep))
            r = toggle_ratios(action)
            ratios.extend(r); n_tog += len(r)
        print(f"TOGGLE demos: ratio mean {np.mean(ratios):.3f}  p25 {np.percentile(ratios,25):.3f}  "
              f"p75 {np.percentile(ratios,75):.3f}  (n_toggles={n_tog})", flush=True)

    for pattern in args.npz:
        files = sorted(globmod.glob(os.path.expanduser(pattern)))
        ratios, n_tog = [], 0
        for f in files:
            z = np.load(f, allow_pickle=True)
            for acts in z["actions"]:
                a = np.asarray(acts, dtype=np.float64)
                r = toggle_ratios(a)
                ratios.extend(r); n_tog += len(r)
        label = os.path.basename(files[0]).split("_libero")[0] if files else pattern
        if ratios:
            print(f"TOGGLE {label}: ratio mean {np.mean(ratios):.3f}  p25 {np.percentile(ratios,25):.3f}  "
                  f"p75 {np.percentile(ratios,75):.3f}  (n_toggles={n_tog}, files={len(files)})", flush=True)
        else:
            print(f"TOGGLE {label}: no toggles found ({len(files)} files)", flush=True)


if __name__ == "__main__":
    main()
