"""Does T-hat stagnation predict failure? (the duration head as a runtime monitor)

Successful approaches show T-hat counting down to events (timeline figure);
failures fumble in grasp loops. If T-hat's downward progress stalls in failures,
the policy has an ENDOGENOUS failure signal, free at deployment.

Metrics per episode from logged t_hat traces (any rollout npz with t_hat):
  stall_frac  — fraction of steps where the 8-step-lag T-hat trend is >= 0
                (no countdown progress) while T-hat < cap (event supposedly near)
  min_that    — closest-to-event the policy ever believed it was
Compare distributions success vs failure + a simple threshold detector's
precision/recall at flagging failures ONLINE (stall_frac over a sliding window).

  python that_stagnation_probe.py --npz "<glob>" [...]
"""
from __future__ import annotations

import argparse
import glob as globmod
import os

import numpy as np


def episode_features(that, cap):
    th = np.asarray(that, dtype=float)
    th = th[th > 0]
    if len(th) < 16:
        return None
    lag = 8
    trend = th[lag:] - th[:-lag]                       # 8-step change in T-hat
    near = th[lag:] < cap                               # event supposedly approaching
    stall = (trend >= 0) & near
    denom = max(near.sum(), 1)
    return {
        "stall_frac": float(stall.sum()) / denom,
        "min_that": float(th.min()),
        "len": len(th),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+")
    ap.add_argument("--cap", type=int, default=24)
    args = ap.parse_args()

    feats, labels, raw_traces = [], [], []
    for pattern in args.npz:
        for f in sorted(globmod.glob(os.path.expanduser(pattern))):
            z = np.load(f, allow_pickle=True)
            if "t_hat" not in z.files:
                continue
            for that, suc in zip(z["t_hat"], z["successes"]):
                ft = episode_features(that, args.cap)
                if ft is not None:
                    feats.append(ft)
                    labels.append(bool(suc))
                    raw_traces.append((that, bool(suc)))
    feats_s = [f for f, l in zip(feats, labels) if l]
    feats_f = [f for f, l in zip(feats, labels) if not l]
    print(f"episodes: {len(feats_s)} success / {len(feats_f)} failure")
    for key in ("stall_frac", "min_that"):
        s = np.array([f[key] for f in feats_s])
        fl = np.array([f[key] for f in feats_f])
        print(f"{key:>11}: success {s.mean():.3f}±{s.std():.3f}   "
              f"failure {fl.mean():.3f}±{fl.std():.3f}")

    # online detector: flag if stall_frac (so far) > thr after warmup
    sf = np.array([f["stall_frac"] for f in feats])
    y = ~np.array(labels)                              # positive = failure
    print("\ndetector: flag failure if episode stall_frac > thr")
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        pred = sf > thr
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = ((~pred) & y).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f"  thr {thr}: precision {prec:.2f}  recall {rec:.2f}  flags {pred.sum()}")

    # ONLINE earliness: first step where trailing-W stall_frac > thr (warmup 60)
    print("\nonline detector (trailing window W=80, warmup 60): flag step stats")
    for thr in (0.35, 0.45):
        flag_steps_f, false_flags_s, lens_f = [], 0, []
        for (that, suc) in raw_traces:
            th = np.asarray(that, dtype=float); th = th[th > 0]
            if len(th) < 70:
                continue
            lag, W = 8, 80
            trend = th[lag:] - th[:-lag]
            near = th[lag:] < args.cap
            stall = ((trend >= 0) & near).astype(float)
            valid = near.astype(float)
            cs, cv = np.cumsum(stall), np.cumsum(valid)
            flag_at = None
            for t in range(60, len(stall)):
                lo = max(t - W, 0)
                v = cv[t] - (cv[lo - 1] if lo > 0 else 0)
                s = cs[t] - (cs[lo - 1] if lo > 0 else 0)
                if v >= 20 and s / v > thr:
                    flag_at = t + lag
                    break
            if suc:
                false_flags_s += int(flag_at is not None)
            elif flag_at is not None:
                flag_steps_f.append(flag_at); lens_f.append(len(th))
        n_s = sum(1 for _, s in raw_traces if s)
        n_f = sum(1 for _, s in raw_traces if not s)
        if flag_steps_f:
            med_flag = np.median(flag_steps_f); med_len = np.median(lens_f)
            print(f"  thr {thr}: failures flagged {len(flag_steps_f)}/{n_f} "
                  f"at median step {med_flag:.0f} of {med_len:.0f} "
                  f"({100 * (1 - med_flag / med_len):.0f}% of episode reclaimable); "
                  f"false alarms {false_flags_s}/{n_s} successes")


if __name__ == "__main__":
    main()
