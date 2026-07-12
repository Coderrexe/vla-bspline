"""Failure-signature analysis over recorded rollouts (npz from record_rollouts.py).

Computes per-episode behavioral signatures and contrasts success vs failure and
C (time-alloc) vs B (fixed-T) on the same tasks:

  grasp_attempted   did the gripper ever command close (+1)?
  first_close_step  when (fraction of episode budget)
  n_grasp_cycles    open->close transitions (fumbling indicator)
  path_len          total commanded translation (action units)
  stall_frac        fraction of steps with near-zero commanded motion
  that_cap_frac     fraction of replans with predicted T at the cap (C only)
  that_mean         mean predicted duration (C only)

Usage: python failure_signatures.py <dir> <glob_label ...>
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np


def episode_sigs(act, that, budget):
    act = np.asarray(act, dtype=np.float64)   # object-array guard (equal-length eps)
    if that is not None:
        that = np.asarray(that, dtype=np.float64).ravel()
    grip = act[:, 6]
    close = grip > 0
    trans = np.where(~close[:-1] & close[1:])[0]
    speed = np.linalg.norm(act[:, :3], axis=1)
    sig = {
        "grasp_attempted": bool(close.any()),
        "first_close_frac": float((np.argmax(close) if close.any() else len(act)) / budget),
        "n_grasp_cycles": int(len(trans)),
        "path_len": float(speed.sum()),
        "stall_frac": float((speed < 0.02).mean()),
        "steps": len(act),
    }
    if that is not None and len(that) and that.max() > 0:
        valid = that[that > 0]
        sig["that_mean"] = float(valid.mean())
        sig["that_cap_frac"] = float((valid >= 40).mean())
        sig["that_short_frac"] = float((valid <= 10).mean())
    return sig


def main(root, labels):
    budget = 520
    for lab in labels:
        rows_s, rows_f = [], []
        for f in sorted(glob.glob(f"{root}/{lab}_t*.npz")):
            z = np.load(f, allow_pickle=True)
            thats = z["t_hat"] if "t_hat" in z.files else [None] * len(z["ep_lens"])
            for act, th, suc in zip(z["actions"], thats, z["successes"]):
                (rows_s if suc else rows_f).append(episode_sigs(act, th, budget))
        def agg(rows):
            if not rows:
                return {}
            keys = set().union(*rows)
            return {k: round(float(np.mean([r[k] for r in rows if k in r])), 3)
                    for k in sorted(keys)}
        print(f"\n=== {lab}: {len(rows_s)} success / {len(rows_f)} failure ===")
        print("SUCCESS:", json.dumps(agg(rows_s)))
        print("FAILURE:", json.dumps(agg(rows_f)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
