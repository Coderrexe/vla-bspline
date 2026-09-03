"""Compute fixed-duration controls from every training-frame action window.

This scanner reads only Parquet action/task columns. It exactly reproduces the
LeRobot future-action query (one window at every frame, final-action edge
padding plus ``action_is_pad``) without decoding observations or images.
Durations then pass through a pinned production ``event_targets.py`` source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_first_event_indices(source: Path):
    source = source.resolve()
    spec = importlib.util.spec_from_file_location(
        "duration_prior_event_targets", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load event target source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.first_event_indices, source


def discrete_median(histogram: Counter[int]) -> int:
    """Smallest integer duration whose cumulative mass reaches 50 percent."""

    total = sum(histogram.values())
    if total == 0:
        raise ValueError("cannot take the median of an empty histogram")
    threshold = (total + 1) // 2
    cumulative = 0
    for duration in sorted(histogram):
        cumulative += histogram[duration]
        if cumulative >= threshold:
            return duration
    raise AssertionError("unreachable")


def _task_strings(path: Path) -> list[str]:
    table = pd.read_parquet(path)
    if table.index.dtype == object:
        return [str(value) for value in table.index]
    if "task" in table.columns:
        return [str(value) for value in table["task"]]
    return [str(value) for value in table.iloc[:, 0]]


def _scalar_task(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"expected one task string, got {value!r}")
        value = value[0]
    return str(value)


def _episode_task_descriptions(root: Path) -> dict[int, str]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError("no episode metadata parquets found")
    result: dict[int, str] = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=["episode_index", "tasks"])
        for row in frame.itertuples(index=False):
            episode = int(row.episode_index)
            task = _scalar_task(row.tasks)
            if episode in result and result[episode] != task:
                raise ValueError(f"episode {episode} has conflicting task descriptions")
            result[episode] = task
    return result


def validate_episode_frames(episode: int, frames: list[int]) -> None:
    """Require one zero-based, contiguous frame schedule for an episode.

    ``future_windows`` indexes an episode array by its row offset.  That is
    equivalent to LeRobot's ``frame_index`` only under this invariant, so a
    gap or nonzero start must be fatal rather than silently shifting every
    subsequent training target.
    """

    if not frames:
        raise ValueError(f"episode {episode} has no frames")
    if frames[0] != 0:
        raise ValueError(
            f"episode {episode} frame indices must start at 0, got {frames[0]}"
        )
    for expected, observed in enumerate(frames):
        if observed != expected:
            raise ValueError(
                f"episode {episode} frame indices are not contiguous: "
                f"expected {expected}, got {observed}"
            )


def _load_action_episodes(root: Path) -> tuple[dict[int, np.ndarray], str, int]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError("no data parquets found")
    rows: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        frame = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "action"]
        )
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        for row in frame.itertuples(index=False):
            episode = int(row.episode_index)
            frame_index = int(row.frame_index)
            action = np.asarray(row.action, dtype="<f4")
            rows[episode].append((frame_index, action))
            digest.update(np.asarray([episode, frame_index], dtype="<i8").tobytes())
            digest.update(action.tobytes())
            total += 1
    episodes: dict[int, np.ndarray] = {}
    for episode, values in rows.items():
        values.sort(key=lambda item: item[0])
        frames = [item[0] for item in values]
        if len(frames) != len(set(frames)):
            raise ValueError(f"episode {episode} has duplicate frame indices")
        validate_episode_frames(episode, frames)
        episodes[episode] = np.stack([item[1] for item in values]).astype(np.float32)
    return episodes, digest.hexdigest(), total


def future_windows(
    actions: np.ndarray, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """All LeRobot-style future windows for one episode."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or len(actions) == 0:
        raise ValueError(f"expected nonempty (T,D) actions, got {actions.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(actions))
    padded = torch.cat([tensor, tensor[-1:].repeat(horizon - 1, 1)], dim=0)
    windows = padded.unfold(0, horizon, 1).permute(0, 2, 1).contiguous()
    anchor = torch.arange(len(actions)).unsqueeze(1)
    offset = torch.arange(horizon).unsqueeze(0)
    pad = anchor + offset >= len(actions)
    return windows, pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--event_target_source", type=Path, required=True)
    parser.add_argument(
        "--max_episodes", type=int, help="smoke only; output is inadmissible"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.ckpt.resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("type") != "smolvla_spline" or not config.get("predict_duration"):
        raise ValueError("duration prior requires a duration-enabled spline checkpoint")
    horizon = int(config["horizon_max"])
    minimum = int(config["min_seg"])
    pause_fraction = float(config["pause_frac"])
    layout = str(config.get("action_layout", "eef7"))
    pose_lo, grip_idx = (5, 11) if layout == "robocasa12" else (0, 6)
    first_event_indices, event_source = load_first_event_indices(
        args.event_target_source
    )

    root = args.dataset_root.expanduser().resolve()
    task_descriptions = _episode_task_descriptions(root)
    actions_by_episode, action_digest, frame_count = _load_action_episodes(root)
    if set(actions_by_episode) != set(task_descriptions):
        raise ValueError("data and episode metadata contain different episode sets")
    episode_ids = sorted(actions_by_episode)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episode_ids = episode_ids[: args.max_episodes]

    global_histogram: Counter[int] = Counter()
    task_histograms: dict[str, Counter[int]] = defaultdict(Counter)
    scanned_frames = 0
    for ordinal, episode in enumerate(episode_ids):
        windows, pad = future_windows(actions_by_episode[episode], horizon)
        durations = first_event_indices(
            windows,
            pad,
            pose_lo=pose_lo,
            grip_idx=grip_idx,
            min_seg=minimum,
            horizon_max=horizon,
            pause_frac=pause_fraction,
        ).tolist()
        global_histogram.update(int(value) for value in durations)
        task_histograms[task_descriptions[episode]].update(
            int(value) for value in durations
        )
        scanned_frames += len(durations)
        if ordinal % 200 == 0:
            print(f"episodes={ordinal + 1} frames={scanned_frames}", flush=True)

    complete = args.max_episodes is None and scanned_frames == frame_count
    original_strings = _task_strings(root / "meta" / "tasks.parquet")[:40]
    task_id = {task: index for index, task in enumerate(original_strings)}
    result = {
        "schema_version": 1,
        "protocol": "production_event_duration_training_prior_v2_parquet",
        "complete_training_scan": complete,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(root),
        "dataset_frames_expected": frame_count,
        "dataset_frames_scanned": scanned_frames,
        "dataset_episode_count": len(actions_by_episode),
        "action_schedule_sha256": action_digest,
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": sha256(config_path),
        "action_layout": layout,
        "horizon_max": horizon,
        "min_seg": minimum,
        "pause_frac": pause_fraction,
        "window_semantics": "every_frame_future_window_final_action_edge_pad_with_pad_mask",
        "median_definition": "smallest integer T with cumulative mass >= 0.5",
        "global_duration": discrete_median(global_histogram),
        "global_histogram": {
            str(key): global_histogram[key] for key in sorted(global_histogram)
        },
        "durations_by_task_description": {
            task: discrete_median(histogram)
            for task, histogram in sorted(task_histograms.items())
        },
        "durations_by_original_task_id": {
            str(task_id[task]): discrete_median(histogram)
            for task, histogram in sorted(task_histograms.items())
            if task in task_id
        },
        "task_histograms": {
            task: {str(key): histogram[key] for key in sorted(histogram)}
            for task, histogram in sorted(task_histograms.items())
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "event_target_source": str(event_source),
        "event_target_source_sha256": sha256(event_source),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete_training_scan",
                    "dataset_frames_scanned",
                    "global_duration",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
