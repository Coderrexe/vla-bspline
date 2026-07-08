"""Duration-head calibration: predicted chunk duration T-hat vs event-rule truth T
on dataset frames. NOTE: training used ALL episodes, so this is ON-TRAIN
calibration — it detects bias / mode-collapse / cap-dragging, not generalization
(closed-loop rollout analysis covers held-out states separately).

Mirrors the training data path exactly: factory-resolved delta timestamps from the
checkpoint's own config, the checkpoint's own preprocessor pipeline, then K
stochastic flow-matching samples per frame via the policy's internals (no queues).

  python calibrate_duration.py --ckpt <pretrained_model> --batches 40 --k 4 \
      --out /path/prefix
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def first_event_with_type(a: torch.Tensor, cfg) -> tuple[torch.Tensor, list[str]]:
    """Same rule as SmolVLASplinePolicy._first_event, but also returns the type."""
    B, Hm, _ = a.shape
    grip = a[..., 6]
    tog = torch.zeros(B, Hm, dtype=torch.bool)
    tog[:, 1:] = grip[:, 1:] != grip[:, :-1]
    speed = a[..., :6].norm(dim=-1)
    med = torch.quantile(speed, 0.5, dim=1, keepdim=True) + 1e-9
    low = speed < cfg.pause_frac * med
    pau = torch.zeros_like(tog)
    pau[:, 1:] = low[:, 1:] & low[:, :-1]
    tog[:, : cfg.min_seg] = False
    pau[:, : cfg.min_seg] = False
    ev = tog | pau
    any_ev = ev.any(dim=1)
    first = torch.argmax(ev.int(), dim=1)
    T = torch.where(any_ev, first, torch.full_like(first, cfg.horizon_max))
    T = T.clamp(cfg.min_seg, cfg.horizon_max)
    types = []
    for b in range(B):
        if not any_ev[b]:
            types.append("cap")
        else:
            k = int(first[b])
            types.append("gripper" if tog[b, k] else "pause")
    return T, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--k", type=int, default=4, help="flow-matching samples per frame")
    ap.add_argument("--out", required=True, help="output prefix (npz + json)")
    args = ap.parse_args()

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    assert getattr(policy_cfg, "predict_duration", False), "checkpoint has no duration head"

    # dataset with the SAME action window the policy trained on
    from lerobot.datasets.factory import resolve_delta_timestamps
    ds_meta = LeRobotDataset("HuggingFaceVLA/libero").meta
    delta_ts = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset("HuggingFaceVLA/libero", delta_timestamps=delta_ts)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
        generator=torch.Generator().manual_seed(7), drop_last=True,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    preproc, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    T_true_all, T_hat_all, types_all = [], [], []
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if bi >= args.batches:
                break
            batch = preproc(batch)
            # truth from the raw window (ACTION normalization is IDENTITY)
            T_true, types = first_event_with_type(batch[ACTION].cpu(), policy.config)
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lt, lm = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            hats = []
            for _ in range(args.k):
                tokens = policy.model.sample_actions(images, img_masks, lt, lm, state)
                t_un = tokens[:, :, :8] * policy._tgt_std + policy._tgt_mean
                hats.append(policy._predicted_T_batch(t_un).cpu())
            T_true_all.append(T_true)
            T_hat_all.append(torch.stack(hats, dim=1))  # (B, K)
            types_all += types
            if bi % 10 == 0:
                print(f"batch {bi}/{args.batches}", flush=True)

    T_true = torch.cat(T_true_all).numpy()                      # (N,)
    T_hat = torch.cat(T_hat_all).numpy()                        # (N, K)
    types = np.array(types_all)
    T_hat_mean = T_hat.mean(axis=1)

    def stats(mask, label):
        if mask.sum() == 0:
            return {"label": label, "n": 0}
        t, h = T_true[mask].astype(float), T_hat_mean[mask]
        mae = float(np.abs(h - t).mean())
        bias = float((h - t).mean())
        r = float(np.corrcoef(t, h)[0, 1]) if t.std() > 0 else float("nan")
        return {"label": label, "n": int(mask.sum()), "MAE": round(mae, 2),
                "bias": round(bias, 2), "pearson_r": round(r, 3),
                "true_mean": round(float(t.mean()), 1), "hat_mean": round(float(h.mean()), 1)}

    report = {
        "overall": stats(np.ones(len(T_true), bool), "overall"),
        "by_type": [stats(types == t, t) for t in ("gripper", "pause", "cap")],
        "hat_at_cap_frac": round(float((T_hat_mean >= policy.config.horizon_max - 0.5).mean()), 3),
        "true_at_cap_frac": round(float((T_true >= policy.config.horizon_max).mean()), 3),
        "sample_std_within_frame": round(float(T_hat.std(axis=1).mean()), 2),
        "note": "ON-TRAIN calibration (no held-out split existed for this run)",
    }
    print(json.dumps(report, indent=1))
    np.savez(f"{args.out}.npz", T_true=T_true, T_hat=T_hat, types=types)
    with open(f"{args.out}.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {args.out}.npz/.json")


if __name__ == "__main__":
    main()
