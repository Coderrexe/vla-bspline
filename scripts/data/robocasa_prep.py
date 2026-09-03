"""RoboCasa365 offline gate + v2 stats generation (one pass, no GPU).

Reads the local robocasa365-target-atomic parquet directly (action[12]:
base 0-3, mode 4, EE pose 5-10, grip 11; fps 20) and, per candidate config
(n_ctrl, cap):
  - v2 event-segmentation stats (toggle/pause/cap fractions, T distribution,
    logT sigma = the duration-signal strength that picked CALVIN's config)
  - pinned-LSQ fit error vs the demo jitter floor (the denoiser check)
Then writes the spline_stats JSON for the chosen config (pose/grip/pass ctrl
stats + logT moments) for training.

  python robocasa_prep.py --root ~/scratch/vla_bspline/robocasa_atomic \
      --grid "6:24,8:24,8:16" --episodes 400
  python robocasa_prep.py --root ... --emit 8:24 \
      --out spline_stats_robocasa_v2_n8h24.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Import the exact implementation that production SmolVLASplinePolicy calls.
# Loading the module directly keeps this offline tool independent of a full
# LeRobot package install while retaining one physical source of truth.
_POLICY_MODULE = Path(__file__).resolve().parents[2] / "policy" / "smolvla_spline"
sys.path.insert(0, str(_POLICY_MODULE))
from event_targets import (  # noqa: E402
    build_event_spline_targets,
    first_event_types,
    make_event_operator_banks,
)

POSE = slice(5, 11)
GRIP = 11
PASS = [0, 1, 2, 3, 4]
MIN_SEG_FRAC = 1 / 3  # min_seg = cap * frac, matching LIBERO/CALVIN ratios


def load_episodes(root, limit, sampling_seed=20260821):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    import pandas as pd

    eps = []
    for f in files:
        df = pd.read_parquet(f, columns=["action", "episode_index"])
        for ep, g in df.groupby("episode_index"):
            eps.append(np.stack(g["action"].to_numpy()))
    if limit and limit < len(eps):
        # Episode indices are grouped by task in the RoboCasa conversion, so
        # taking the first N episodes biases the statistics toward early tasks.
        rng = np.random.default_rng(sampling_seed)
        selected = sorted(rng.choice(len(eps), size=limit, replace=False).tolist())
        eps = [eps[index] for index in selected]
    return eps


def training_windows(a, cap, stride=8):
    """Yield dataloader-style windows plus ``action_is_pad`` masks.

    LeRobot repeats the final action to make a full future window and separately
    marks repeated positions as padding.  Production computes its per-window
    speed median on the repeated raw action, then treats the first eligible pad
    position as an episode-end event.  Reproducing both details is necessary for
    exact statistics near episode boundaries.
    """

    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if not len(a):
        return
    for start in range(0, len(a), stride):
        valid = min(cap, len(a) - start)
        window = a[start : start + valid]
        pad = np.zeros(cap, dtype=bool)
        if valid < cap:
            window = np.concatenate([window, np.repeat(window[-1:], cap - valid, axis=0)])
            pad[valid:] = True
        yield start, window, pad


def fit_stats(eps, n_ctrl, cap, min_seg, collect=False, stride=8, pause_frac=0.15):
    kinds, Ts, errs, jitter = [], [], [], []
    cp, cg, cpas, logT = [], [], [], []
    pm_bank, bl_bank, pg_bank, bp_bank = make_event_operator_banks(
        n_ctrl=n_ctrl,
        degree=3,
        min_seg=min_seg,
        horizon_max=cap,
    )
    for a in eps:
        samples = list(training_windows(a, cap, stride=stride))
        if not samples:
            continue
        windows = torch.as_tensor(np.stack([sample[1] for sample in samples]), dtype=torch.float32)
        pads = torch.as_tensor(np.stack([sample[2] for sample in samples]), dtype=torch.bool)
        targets = build_event_spline_targets(
            windows,
            pads,
            pm_bank=pm_bank,
            bl_bank=bl_bank,
            pg_bank=pg_bank,
            pose_lo=POSE.start,
            grip_idx=GRIP,
            pass_dims=PASS,
            min_seg=min_seg,
            horizon_max=cap,
            pause_frac=pause_frac,
        )
        batch_kinds = first_event_types(
            windows,
            pads,
            pose_lo=POSE.start,
            grip_idx=GRIP,
            min_seg=min_seg,
            horizon_max=cap,
            pause_frac=pause_frac,
        )
        for row, (duration, kind) in enumerate(zip(targets.duration.tolist(), batch_kinds)):
            T = int(duration)
            kinds.append(kind)
            Ts.append(T)

            seg_pose = windows[row, :T, POSE].clone()
            seg_pose[pads[row, :T]] = 0
            basis = bp_bank[T - min_seg, : T + 1]
            reconstructed = basis @ targets.pose_ctrl[row]
            step_err = torch.linalg.vector_norm(torch.diff(reconstructed, dim=0) - seg_pose, dim=1)
            errs.append(step_err.mean())
            # jitter floor: half the mean step-to-step delta change
            delta_delta = torch.diff(seg_pose, dim=0)
            if len(delta_delta):
                jitter.append(
                    0.5 * torch.linalg.vector_norm(torch.diff(delta_delta, dim=0), dim=1).mean().item()
                    if len(delta_delta) > 1
                    else 0.0
                )
            if collect:
                cp.append(targets.pose_ctrl[row].numpy())
                cg.append(targets.grip_ctrl[row].numpy())
                assert targets.pass_ctrl is not None
                cpas.append(targets.pass_ctrl[row].numpy())
                logT.append(np.log(T))
    kinds = np.array(kinds)
    errs = np.asarray([float(value) for value in errs])
    out = {
        "n_seg": len(kinds),
        "toggle_frac": float((kinds == "toggle").mean()),
        "pause_frac": float((kinds == "pause").mean()),
        "episode_end_frac": float((kinds == "episode_end").mean()),
        "cap_frac": float((kinds == "cap").mean()),
        "T_mean": float(np.mean(Ts)), "T_cov": float(np.std(Ts) / np.mean(Ts)),
        "logT_sigma": float(np.std(np.log(Ts))),
        "fit_err": float(np.mean(errs)), "jitter_floor": float(np.mean(jitter)),
        "density": n_ctrl / np.mean(Ts),
    }
    if collect:
        cp = np.stack(cp); cg = np.stack(cg); cpas = np.stack(cpas)
        out["stats_json"] = {
            "stats_contract_version": 1,
            "n_ctrl": n_ctrl, "degree": 3, "horizon_max": cap, "min_seg": min_seg,
            "pause_frac": pause_frac,
            "action_layout": "robocasa12",
            "pose_dims": [5, 6, 7, 8, 9, 10], "grip_idx": 11,
            "pass_dims": [0, 1, 2, 3, 4],
            "boundary_semantics": "production_event_index_k_exclusive",
            "segmentation": "production_event_index_k_exclusive",
            "window_stride": stride,
            "target_dtype": "float32",
            "pose_ctrl_mean": cp.mean(0).tolist(), "pose_ctrl_std": (cp.std(0) + 1e-6).tolist(),
            "grip_ctrl_mean": cg.mean(0).tolist(), "grip_ctrl_std": (cg.std(0) + 1e-6).tolist(),
            "pass_ctrl_mean": cpas.mean(0).tolist(), "pass_ctrl_std": (cpas.std(0) + 1e-6).tolist(),
            "logT_mean": float(np.mean(logT)), "logT_std": float(np.std(logT) + 1e-6),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--grid", default="6:24,8:24,8:16")
    ap.add_argument("--emit", default=None, help="n:cap — write stats JSON for this config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=8, help="training-anchor subsampling stride")
    ap.add_argument("--pause_frac", type=float, default=0.15)
    ap.add_argument("--sampling_seed", type=int, default=20260821)
    args = ap.parse_args()

    eps = load_episodes(
        os.path.expanduser(args.root),
        args.episodes,
        sampling_seed=args.sampling_seed,
    )
    print(f"loaded {len(eps)} episodes; mean len {np.mean([len(e) for e in eps]):.1f} steps")

    if args.emit:
        n, cap = (int(x) for x in args.emit.split(":"))
        min_seg = max(4, round(cap * MIN_SEG_FRAC))
        r = fit_stats(
            eps,
            n,
            cap,
            min_seg,
            collect=True,
            stride=args.stride,
            pause_frac=args.pause_frac,
        )
        sj = r.pop("stats_json")
        sj["source_episode_sample_count"] = len(eps)
        sj["source_episode_sampling_seed"] = args.sampling_seed
        print(json.dumps(r, indent=1))
        out = args.out or f"spline_stats_robocasa_v2_n{n}h{cap}.json"
        with open(out, "w") as f:
            json.dump(sj, f)
        print("WROTE", out)
        return

    print(f"{'cfg':>8} {'segs':>6} {'tog%':>6} {'pau%':>6} {'end%':>6} {'cap%':>6} "
          f"{'T_mean':>7} {'logTsd':>7} {'fit':>8} {'jitter':>8} {'dens':>5}")
    for spec in args.grid.split(","):
        n, cap = (int(x) for x in spec.split(":"))
        min_seg = max(4, round(cap * MIN_SEG_FRAC))
        r = fit_stats(
            eps,
            n,
            cap,
            min_seg,
            stride=args.stride,
            pause_frac=args.pause_frac,
        )
        print(f"{spec:>8} {r['n_seg']:>6} {100*r['toggle_frac']:>6.1f} {100*r['pause_frac']:>6.1f} "
              f"{100*r['episode_end_frac']:>6.1f} "
              f"{100*r['cap_frac']:>6.1f} {r['T_mean']:>7.1f} {r['logT_sigma']:>7.3f} "
              f"{r['fit_err']:>8.4f} {r['jitter_floor']:>8.4f} {r['density']:>5.2f}")


if __name__ == "__main__":
    main()
