"""Audit the realized frame-level language intervention in a LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_task(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0])
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original_tasks", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    tasks_path = root / "meta/tasks.parquet"
    episodes_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not tasks_path.exists() or not episodes_paths or not data_paths:
        raise FileNotFoundError("dataset is missing tasks, episode, or data parquet files")

    tasks_table = pd.read_parquet(tasks_path)
    if tasks_table.index.dtype == object:
        task_strings = [str(value) for value in tasks_table.index]
    else:
        task_strings = [str(value) for value in tasks_table.iloc[:, 0]]
    if len(task_strings) <= args.original_tasks:
        raise ValueError("dataset has no appended language-label tasks")
    original_id = {task: index for index, task in enumerate(task_strings[: args.original_tasks])}

    episodes = pd.concat([pd.read_parquet(path) for path in episodes_paths], ignore_index=True)
    original_by_episode = {}
    metadata_task_max = {}
    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        task = scalar_task(row["tasks"])
        if task not in original_id:
            raise ValueError(f"episode {episode} has unknown original task {task!r}")
        original_by_episode[episode] = original_id[task]
        if "stats/task_index/max" in episodes.columns:
            metadata_task_max[episode] = int(
                np.asarray(row["stats/task_index/max"]).reshape(-1)[0]
            )

    per_task = defaultdict(
        lambda: {
            "episodes": set(),
            "changed_episodes": set(),
            "frames": 0,
            "changed_frames": 0,
            "task_runs": 0,
            "granular_runs": 0,
        }
    )
    label_frame_counts = Counter()
    label_run_counts = Counter()
    total_frames = 0
    changed_frames = 0
    changed_episodes = set()

    for path in data_paths:
        frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "task_index"])
        for episode, group in frame.groupby("episode_index", sort=False):
            episode = int(episode)
            if episode not in original_by_episode:
                raise ValueError(f"data references episode {episode} absent from metadata")
            group = group.sort_values("frame_index")
            task_indices = group["task_index"].to_numpy(dtype=np.int64)
            original = original_by_episode[episode]
            changed = task_indices >= args.original_tasks
            starts = np.r_[True, task_indices[1:] != task_indices[:-1]]
            run_labels = task_indices[starts]

            stats = per_task[original]
            stats["episodes"].add(episode)
            stats["frames"] += len(task_indices)
            stats["changed_frames"] += int(changed.sum())
            stats["task_runs"] += len(run_labels)
            stats["granular_runs"] += int(np.sum(run_labels >= args.original_tasks))
            if changed.any():
                stats["changed_episodes"].add(episode)
                changed_episodes.add(episode)

            total_frames += len(task_indices)
            changed_frames += int(changed.sum())
            label_frame_counts.update(task_indices[changed].tolist())
            label_run_counts.update(run_labels[run_labels >= args.original_tasks].tolist())

    task_rows = []
    for task_id in range(args.original_tasks):
        stats = per_task[task_id]
        frames = stats["frames"]
        task_rows.append(
            {
                "task_id": task_id,
                "task": task_strings[task_id],
                "episodes": len(stats["episodes"]),
                "changed_episodes": len(stats["changed_episodes"]),
                "frames": frames,
                "changed_frames": stats["changed_frames"],
                "changed_frame_fraction": stats["changed_frames"] / frames if frames else 0.0,
                "task_runs": stats["task_runs"],
                "granular_runs": stats["granular_runs"],
            }
        )

    used_labels = set(label_frame_counts)
    stale_metadata_episodes = sum(
        metadata_task_max.get(episode, -1) < args.original_tasks
        for episode in changed_episodes
    )
    result = {
        "schema_version": 1,
        "dataset_root": str(root),
        "original_task_count": args.original_tasks,
        "appended_label_count": len(task_strings) - args.original_tasks,
        "used_appended_label_count": len(used_labels),
        "unused_appended_label_count": len(task_strings) - args.original_tasks - len(used_labels),
        "single_run_label_count": sum(count == 1 for count in label_run_counts.values()),
        "episode_count": len(original_by_episode),
        "changed_episode_count": len(changed_episodes),
        "changed_episode_fraction": len(changed_episodes) / len(original_by_episode),
        "total_frames": total_frames,
        "changed_frames": changed_frames,
        "changed_frame_fraction": changed_frames / total_frames,
        "stale_original_only_episode_metadata_count": stale_metadata_episodes,
        "tasks_with_any_changed_frames": [row["task_id"] for row in task_rows if row["changed_frames"]],
        "tasks": task_rows,
        "top_labels_by_runs": [
            {
                "task_index": index,
                "label": task_strings[index],
                "runs": runs,
                "frames": label_frame_counts[index],
            }
            for index, runs in label_run_counts.most_common(50)
        ],
        "metadata_sha256": {
            "tasks.parquet": sha256(tasks_path),
            **{
                str(path.relative_to(root)): sha256(path)
                for path in episodes_paths
            },
        },
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)

    print(
        f"changed {changed_frames}/{total_frames} frames "
        f"({100 * result['changed_frame_fraction']:.1f}%) in "
        f"{len(changed_episodes)}/{len(original_by_episode)} episodes"
    )
    for row in task_rows[:10]:
        print(
            f"t{row['task_id']}: episodes={row['changed_episodes']}/{row['episodes']} "
            f"frames={100 * row['changed_frame_fraction']:.1f}% "
            f"granular_runs={row['granular_runs']}"
        )
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
