"""Language-commanded TIME probe: does a speed adverb steer the predicted
duration T-hat? (The axis only a duration head exposes to language.)

For N labeled segment-start observations: query the policy with the segment's
clause under three prompts — plain / "quickly ..." / "slowly and carefully ..."
— and read last_predicted_T. Paired per observation; the control checkpoint
(Cn8, never trained on adverbs) should show no systematic shift.

  python t_adverb_probe.py --ckpt .../ClangAdv_100k/.../pretrained_model \
      --labels ~/scratch/vla_bspline/molmo_libero_object/seg_labels.jsonl --n 32
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--repo", default="HuggingFaceVLA/libero")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo)
    ep_from = ds.meta.episodes["dataset_from_index"]

    recs = [json.loads(l) for l in open(os.path.expanduser(args.labels))]
    # segment-start frames, mid-episode only (skip t=0 rest frames), plain labels
    rng = np.random.default_rng(0)
    cand = [r for r in recs if r["seg_start"] > 8]
    rng.shuffle(cand)
    cand = cand[: args.n]

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta)
    policy.eval()
    preproc, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    cams = [k for k in ds.meta.features if k.startswith("observation.images")]

    def that_for(item, prompt, drain=8):
        """Returns (T-hat, decoded mean pose speed over the first `drain` steps).
        Adverbs encode SPEED terciles — the coupling lives in control-point
        spacing (per-step delta magnitude), not necessarily chunk length."""
        batch = {"observation.state": item["observation.state"].unsqueeze(0).float(),
                 "task": [prompt]}
        for c in cams:
            batch[c] = item[c].unsqueeze(0).float()
        policy.reset()
        speeds = []
        with torch.inference_mode():
            for _ in range(drain):
                b = preproc(dict(batch))
                a = policy.select_action(b)
                a = a.cpu().numpy().reshape(-1)
                speeds.append(float(np.linalg.norm(a[:6])))
        return float(getattr(policy, "last_predicted_T", -1)), float(np.mean(speeds))

    rows = []
    for r in cand:
        gi = int(ep_from[r["episode_index"]]) + r["seg_start"]
        item = ds[gi]
        base = r["label"]
        for pre, tag in (("", "plain"), ("", "plain2"),
                         ("quickly ", "quick"), ("slowly and carefully ", "slow")):
            rows.append((tag, *that_for(item, pre + base)))
    import collections
    byT = collections.defaultdict(list)
    byV = collections.defaultdict(list)
    for tag, t, v in rows:
        byT[tag].append(t)
        byV[tag].append(v)
    def arr(d, k):
        return np.array(d[k])
    print(f"n={len(byT['plain'])} segment-start observations; drain=8 steps/prompt")
    for name, d in (("T-hat", byT), ("decoded speed", byV)):
        P, P2, Q, S = (arr(d, k) for k in ("plain", "plain2", "quick", "slow"))
        noise = np.abs(P - P2)
        dq, dsl, spread = Q - P, S - P, S - Q
        se = lambda x: x.std(ddof=1) / len(x) ** 0.5 + 1e-12
        print(f"[{name}] plain {P.mean():.4f} | repeat-noise {noise.mean():.4f} | "
              f"d(quick) {dq.mean():+.4f}+/-{se(dq):.4f} | d(slow) {dsl.mean():+.4f}+/-{se(dsl):.4f} | "
              f"SPREAD slow-quick {spread.mean():+.4f} t={spread.mean()/se(spread):.2f}")


if __name__ == "__main__":
    main()
