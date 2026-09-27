"""Load/predict checks and raw-action errors on held-out recorded observations.

This is not a closed-loop success evaluation. Five whole episodes are reserved;
sample frames evenly across each, including late insertion phases. Predictions
are made one observation at a time, matching deployment's duration decoding.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from apollo_predictor import ApolloPredictor
from apollo_legacy_state import CheckpointStateAdapter
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames-per-episode", type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    predictor = ApolloPredictor(args.checkpoint)
    state_adapter = CheckpointStateAdapter(args.checkpoint)
    meta = json.loads((args.dataset / "export_manifest.json").read_text())
    native = json.loads((args.dataset / "apollo_manifest.json").read_text())
    if native["fps"] != 25:
        raise ValueError("This Apollo integration was validated for the recorded 25 Hz interface")
    ds = LeRobotDataset(native["repo_id"], root=args.dataset,
                        delta_timestamps={"action": [i / 25 for i in range(24)]},
                        video_backend="pyav", return_uint8=True)
    episodes = pq.read_table(args.dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    rows, predicted, targets, holds, latencies = [], [], [], [], []
    sample_index = 0
    image_contract = None
    for episode in episodes[meta["train_episodes"]:]:
        predictor.reset()
        indices = np.linspace(0, episode["length"] - 25, args.frames_per_episode).round().astype(int)
        for frame in indices:
            item = ds[episode["dataset_from_index"] + int(frame)]
            state = state_adapter.stabilize_training_state(item["observation.state"].numpy())
            view = item["observation.images.view_wrist"].permute(1, 2, 0).numpy()
            grip = item["observation.images.grip_wrist"].permute(1, 2, 0).numpy()
            if image_contract is None:
                batch = predictor.prepare(state, view, grip)
                prepared, masks = predictor.policy.prepare_images(batch)
                image_contract = {"camera_order": list(predictor.cfg.image_features),
                                  "images": [{"shape": list(x.shape), "min": x.min().item(),
                                              "max": x.max().item()} for x in prepared]}
                assert len(prepared) == 2 and all(bool(m.all()) for m in masks)
                assert all(x.is_floating_point() and -1 <= x.min() and x.max() <= 1 for x in prepared)
            torch.manual_seed(20260912 + sample_index)
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = predictor.predict_chunk(state, view, grip)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if action.shape != (8, 16):
                raise ValueError(f"Unexpected execution prefix: {action.shape}")
            if not (np.all(action[:, 7:14] == 0) and np.all(action[:, 14] == 1) and np.all(action[:, 15] == 0)):
                raise ValueError("Inactive arm output violated")
            target = item["action"].numpy()[:len(action)]
            hold = np.zeros_like(action)
            hold[:, 6] = state[7]
            predicted.append(action); targets.append(target); holds.append(hold)
            latencies.append(elapsed)
            rows.append({"episode_index": episode["episode_index"],
                         "source_episode_id": episode["source_episode_id"], "frame": int(frame),
                         "latency_seconds": elapsed,
                         "translation_rmse_m": float(np.sqrt(np.mean((action[:, :3]-target[:, :3])**2))),
                         "gripper_mae": float(np.mean(np.abs(action[:, 6]-target[:, 6]))),
                         "predicted_duration": getattr(predictor.policy, "last_predicted_T", None)})
            sample_index += 1
    pred, target, hold = map(np.stack, (predicted, targets, holds))

    def metrics(values):
        err = values - target
        return {"translation_component_rmse_m": float(np.sqrt(np.mean(err[..., :3]**2))),
                "rotation_component_rmse_rad": float(np.sqrt(np.mean(err[..., 3:6]**2))),
                "gripper_open_fraction_mae": float(np.mean(np.abs(err[..., 6]))),
                "eight_step_translation_endpoint_rmse_m": float(np.sqrt(np.mean(err[..., :3].sum(1)**2)))}

    report = {"checkpoint": str(args.checkpoint.resolve()), "task": meta["task"],
              "validation_episodes": len(episodes)-meta["train_episodes"], "observations": len(rows),
              "seed": 20260912, "prefix_actions": 8, "nominal_hz": 25,
              "state_pose_convention": state_adapter.pose_convention,
              "parked_features_projected_to_checkpoint_reference": np.flatnonzero(state_adapter.mask).tolist(),
              "image_contract": image_contract, "load_predict": "PASS",
              "policy": metrics(pred), "zero_motion_hold_gripper_reference": metrics(hold),
              "prediction_abs_max_active_channels": np.max(np.abs(pred[..., :7]), axis=(0,1)).tolist(),
              "latency_ms_median_excluding_first": float(1000*np.median(latencies[1:])),
              "latency_ms_p95_excluding_first": float(1000*np.quantile(latencies[1:], .95)),
              "gpu": torch.cuda.get_device_name(), "samples": rows,
              "interpretation": "Offline command agreement, not physical safety or task success"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), predicted=pred, target=target, hold=hold)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
