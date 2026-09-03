"""Parity smoke for the direct-Parquet duration-prior scanner.

The scanner deliberately avoids constructing ``LeRobotDataset`` so that it can
read 273k action windows without decoding images.  This utility checks a small,
explicit list of episode/frame anchors against the installed LeRobot revision's
*actual* action query helpers.  Those helpers are the same ones used by
``LeRobotDataset.__getitem__``, but this script never calls the video-query path.

Example (run only in the pinned LeRobot environment)::

    python duration_prior_parity.py \
      --ckpt /path/to/pretrained_model \
      --dataset_root /path/to/libero_l10gran \
      --event_target_source /path/to/event_targets.py \
      --sample 0:0 --sample 0:20 --sample 0:49 \
      --sample 1:0 --sample 1:31 --sample 1:63 \
      --out /tmp/duration_prior_parity.json

The sample coordinates are intentional inputs, never sampled implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from scripts.analysis.duration_training_prior import (
    _load_action_episodes,
    future_windows,
    load_first_event_indices,
    sha256,
)


def parse_sample(value: str) -> tuple[int, int]:
    """Parse one explicit ``EPISODE:FRAME`` anchor."""

    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("sample must have form EPISODE:FRAME")
    try:
        episode, frame = (int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sample coordinates must be integers"
        ) from error
    if episode < 0 or frame < 0:
        raise argparse.ArgumentTypeError("sample coordinates must be nonnegative")
    return episode, frame


def _first_difference(left: torch.Tensor, right: torch.Tensor) -> str:
    mismatch = torch.nonzero(left != right, as_tuple=False)
    if len(mismatch) == 0:
        return "none"
    index = tuple(int(value) for value in mismatch[0].tolist())
    return (
        f"index={index} direct={left[index].item()!r} lerobot={right[index].item()!r}"
    )


def compare_window_and_target(
    *,
    direct_action: torch.Tensor,
    direct_pad: torch.Tensor,
    lerobot_action: torch.Tensor,
    lerobot_pad: torch.Tensor,
    first_event_indices: Callable[..., torch.Tensor],
    event_kwargs: dict[str, Any],
) -> int:
    """Fail on any window/mask mismatch, then compare the pinned event target."""

    direct_action = torch.as_tensor(direct_action, dtype=torch.float32)
    lerobot_action = torch.as_tensor(lerobot_action, dtype=torch.float32)
    direct_pad = torch.as_tensor(direct_pad, dtype=torch.bool)
    lerobot_pad = torch.as_tensor(lerobot_pad, dtype=torch.bool)
    if direct_action.shape != lerobot_action.shape:
        raise AssertionError(
            f"action shape mismatch: direct={tuple(direct_action.shape)} "
            f"lerobot={tuple(lerobot_action.shape)}"
        )
    if direct_pad.shape != lerobot_pad.shape:
        raise AssertionError(
            f"pad shape mismatch: direct={tuple(direct_pad.shape)} "
            f"lerobot={tuple(lerobot_pad.shape)}"
        )
    if not torch.equal(direct_action, lerobot_action):
        raise AssertionError(
            "action window mismatch: "
            + _first_difference(direct_action, lerobot_action)
        )
    if not torch.equal(direct_pad, lerobot_pad):
        raise AssertionError(
            "padding mismatch: " + _first_difference(direct_pad, lerobot_pad)
        )

    direct_target = first_event_indices(
        direct_action.unsqueeze(0), direct_pad.unsqueeze(0), **event_kwargs
    )
    lerobot_target = first_event_indices(
        lerobot_action.unsqueeze(0), lerobot_pad.unsqueeze(0), **event_kwargs
    )
    if not torch.equal(direct_target, lerobot_target):
        raise AssertionError(
            f"event-target mismatch: direct={direct_target.tolist()} "
            f"lerobot={lerobot_target.tolist()}"
        )
    return int(direct_target.item())


def _git_commit_for_module(module: object) -> str | None:
    source = Path(inspect.getfile(module)).resolve()
    for candidate in (source.parent, *source.parents):
        if (candidate / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def _load_lerobot_action_query(
    *, checkpoint: Path, dataset_root: Path, dataset_repo_id: str, horizon: int
):
    """Return the pinned LeRobot dataset and its action-only core query.

    We intentionally use DatasetReader's private query helpers.  The installed
    revision has no public action-window-only API: ``dataset[idx]`` also calls
    ``_query_videos``.  Failing on an API change is safer than falling back to
    full observation/video decoding in what should be a tiny CPU audit.
    """

    try:
        import lerobot
        # The project's spline policy is an untracked extension of LeRobot and
        # must be imported explicitly before Draccus resolves the config type.
        importlib.import_module(
            "lerobot.policies.smolvla_spline.configuration_smolvla_spline"
        )
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ImportError as error:
        raise RuntimeError(
            "run this smoke inside the pinned LeRobot environment"
        ) from error

    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    if "action" not in delta_timestamps:
        raise RuntimeError("resolved delta timestamps contain no action query")
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    reader = dataset.reader
    required = ("_get_query_indices", "_query_hf_dataset")
    if any(not hasattr(reader, name) for name in required):
        raise RuntimeError(
            "installed DatasetReader lacks the audited action-only helpers; "
            "do not fall back to dataset[idx] because it decodes video"
        )
    action_offsets = list(reader.delta_indices.get("action", []))
    expected_offsets = list(range(horizon))
    if action_offsets != expected_offsets:
        raise AssertionError(
            f"resolved action offsets differ: got={action_offsets}, expected={expected_offsets}"
        )

    def query(episode: int, frame: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not 0 <= episode < len(dataset.meta.episodes):
            raise IndexError(f"LeRobot metadata has no episode {episode}")
        episode_metadata = dataset.meta.episodes[episode]
        metadata_episode = int(episode_metadata["episode_index"])
        if metadata_episode != episode:
            raise AssertionError(
                "LeRobot episode metadata is not indexed by episode_index: "
                f"row {episode} describes episode {metadata_episode}"
            )
        episode_start = int(episode_metadata["dataset_from_index"])
        episode_end = int(episode_metadata["dataset_to_index"])
        absolute_index = episode_start + frame
        if absolute_index >= episode_end:
            raise IndexError(
                f"episode {episode} has {episode_end - episode_start} frames, got frame {frame}"
            )
        # Raw-row access verifies the global-index mapping and never decodes video.
        raw = dataset.get_raw_item(absolute_index)
        raw_episode = int(torch.as_tensor(raw["episode_index"]).item())
        raw_frame = int(torch.as_tensor(raw["frame_index"]).item())
        raw_index = int(torch.as_tensor(raw["index"]).item())
        if (raw_episode, raw_frame, raw_index) != (episode, frame, absolute_index):
            raise AssertionError(
                "LeRobot coordinate mapping mismatch: "
                f"raw={(raw_episode, raw_frame, raw_index)}, "
                f"expected={(episode, frame, absolute_index)}"
            )
        query_indices, padding = reader._get_query_indices(absolute_index, episode)
        action = reader._query_hf_dataset({"action": query_indices["action"]})["action"]
        return action, padding["action_is_pad"], absolute_index

    return dataset, query, delta_timestamps, _git_commit_for_module(lerobot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--event_target_source", type=Path, required=True)
    parser.add_argument(
        "--sample",
        type=parse_sample,
        action="append",
        required=True,
        help="explicit EPISODE:FRAME anchor; repeat for each sample",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.sample)) != len(args.sample):
        raise ValueError("duplicate parity samples are not allowed")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.ckpt.expanduser().resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("type") != "smolvla_spline" or not config.get("predict_duration"):
        raise ValueError("parity smoke requires a duration-enabled spline checkpoint")
    horizon = int(config["horizon_max"])
    layout = str(config.get("action_layout", "eef7"))
    pose_lo, grip_idx = (5, 11) if layout == "robocasa12" else (0, 6)
    event_kwargs = {
        "pose_lo": pose_lo,
        "grip_idx": grip_idx,
        "min_seg": int(config["min_seg"]),
        "horizon_max": horizon,
        "pause_frac": float(config["pause_frac"]),
    }
    first_event_indices, event_source = load_first_event_indices(
        args.event_target_source
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    actions_by_episode, action_digest, _ = _load_action_episodes(dataset_root)
    dataset, lerobot_query, delta_timestamps, lerobot_commit = (
        _load_lerobot_action_query(
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            dataset_repo_id=args.dataset_repo_id,
            horizon=horizon,
        )
    )

    rows = []
    for episode, frame in args.sample:
        if episode not in actions_by_episode:
            raise IndexError(f"direct scanner has no episode {episode}")
        if frame >= len(actions_by_episode[episode]):
            raise IndexError(
                f"episode {episode} has {len(actions_by_episode[episode])} frames, got {frame}"
            )
        direct_windows, direct_padding = future_windows(
            actions_by_episode[episode], horizon
        )
        lerobot_action, lerobot_pad, absolute_index = lerobot_query(episode, frame)
        duration = compare_window_and_target(
            direct_action=direct_windows[frame],
            direct_pad=direct_padding[frame],
            lerobot_action=lerobot_action,
            lerobot_pad=lerobot_pad,
            first_event_indices=first_event_indices,
            event_kwargs=event_kwargs,
        )
        rows.append(
            {
                "episode_index": episode,
                "frame_index": frame,
                "absolute_index": absolute_index,
                "duration_target": duration,
                "pad_count": int(torch.as_tensor(lerobot_pad).sum().item()),
                "action_window_sha256": hashlib.sha256(
                    torch.as_tensor(lerobot_action, dtype=torch.float32)
                    .contiguous()
                    .numpy()
                    .astype("<f4", copy=False)
                    .tobytes()
                ).hexdigest(),
            }
        )

    result = {
        "schema_version": 1,
        "protocol": "direct_parquet_vs_lerobot_action_query_v1",
        "all_samples_exact": True,
        "sample_count": len(rows),
        "samples": rows,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "dataset_codebase_version": getattr(dataset.meta, "codebase_version", None),
        "action_schedule_sha256": action_digest,
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": sha256(config_path),
        "event_target_source": str(event_source),
        "event_target_source_sha256": sha256(event_source),
        "resolved_action_delta_timestamps": [
            float(value) for value in delta_timestamps["action"]
        ],
        "lerobot_commit": lerobot_commit,
        "script_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "video_decode_called": False,
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"all_samples_exact": True, "sample_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
