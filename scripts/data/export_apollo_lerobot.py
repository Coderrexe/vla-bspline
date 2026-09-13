"""Lossless native Apollo -> LeRobot v3 numeric/video export.

Retained frame order and commands are unchanged. Five deterministically spaced
episodes are placed last to support LeRobot's per-task held-out split (.09).
Statistics use training episodes only. Source videos are hardlinked when possible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def numeric_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, .01, axis=0).tolist(),
            "q99": np.quantile(x, .99, axis=0).tolist(), "count": [len(x)]}


def export_task(source: Path, dest: Path, excluded_ids=()):
    if dest.exists():
        raise FileExistsError(f"Refusing to replace existing dataset {dest}")
    meta = json.loads((source / "manifest.json").read_text())
    paths = sorted((source / "episodes").glob("*/episode.json"))
    if len(paths) != meta["episodes"] or len(paths) < 10:
        raise ValueError("Unexpected episode count")
    heldout = {int(i) for i in np.linspace(4, len(paths) - 6, 5).round()}
    excluded_ids = set(excluded_ids)
    if excluded_ids - {p.parent.name for p in paths}:
        raise ValueError("An excluded episode ID was not found in the source")
    excluded = {i for i, p in enumerate(paths) if p.parent.name in excluded_ids}
    if excluded & heldout:
        raise ValueError("Do not silently change the established validation episodes")
    order = [i for i in range(len(paths)) if i not in heldout | excluded] + sorted(heldout)
    ntrain = len(order) - 5
    excluded_frames = sum(pq.read_metadata(paths[i].parent / 'frames.parquet').num_rows for i in excluded)
    features = copy.deepcopy(meta["features"])
    features.update({name: {"dtype": typ, "shape": [1], "names": None}
                     for name, typ in [("timestamp", "float32"), ("frame_index", "int64"),
                                       ("episode_index", "int64"), ("index", "int64"),
                                       ("task_index", "int64")]})
    for key in [k for k in features if k.startswith("observation.images.")]:
        features[key]["info"] = {"video.fps": meta["fps"], "video.height": 480,
                                 "video.width": 640, "video.channels": 3,
                                 "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                 "video.is_depth_map": False, "has_audio": False}
    for directory in [dest / "meta/episodes/chunk-000", dest / "data/chunk-000"]:
        directory.mkdir(parents=True)
    offset = 0
    tables, episodes, mapping, actions_train, states_train = [], [], [], [], []
    task_name = None
    for index, source_index in enumerate(order):
        ep_path = paths[source_index]
        ep = json.loads(ep_path.read_text())
        src_table = pq.read_table(ep_path.parent / "frames.parquet")
        n = len(src_table)
        task_names = set(src_table["task"].to_pylist())
        if len(task_names) != 1:
            raise ValueError("This export expects one task caption per episode")
        name = task_names.pop()
        task_name = task_name or name
        if name != task_name:
            raise ValueError("Task captions vary within this dataset")
        a = np.asarray(src_table["action"].to_pylist(), dtype=np.float32)
        s = np.asarray(src_table["observation.state"].to_pylist(), dtype=np.float32)
        if not (a.shape == (n, 16) and s.shape == (n, 32) and np.isfinite(a).all() and np.isfinite(s).all()):
            raise ValueError("Invalid action/state array")
        if not np.allclose(a[:, [7, 8, 9, 10, 11, 12, 13, 15]], 0, atol=1e-8):
            raise ValueError("A supposedly inactive arm or rail has nonzero commands")
        if not np.allclose(a[:, 14], 1, atol=1e-7):
            raise ValueError("View gripper is not constant open")
        if index < ntrain:
            actions_train.append(a); states_train.append(s)
        table = src_table.drop(["task"])
        for key, vals in [("episode_index", np.full(n, index)), ("task_index", np.zeros(n)),
                          ("index", np.arange(offset, offset + n))]:
            table = table.append_column(key, pa.array(vals, type=pa.int64()))
        tables.append(table)
        row = {"episode_index": index, "tasks": [task_name], "length": n,
               "dataset_from_index": offset, "dataset_to_index": offset + n,
               "data/chunk_index": 0, "data/file_index": 0,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
               "source_episode_id": ep_path.parent.name}
        for key, stats in ep["stats"].items():
            for stat, value in stats.items():
                row[f"stats/{key}/{stat}"] = value
        for camera in meta["cameras"]:
            key = f"observation.images.{camera}"
            video = ep_path.parent / "video" / f"{camera}.mp4"
            target = dest / "videos" / key / "chunk-000" / f"file-{index:03d}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(video, target)
            except OSError:
                shutil.copy2(video, target)
            row.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": index,
                        f"videos/{key}/from_timestamp": 0., f"videos/{key}/to_timestamp": n / meta["fps"]})
        episodes.append(row)
        mapping.append({"episode_index": index, "source_index": source_index,
                        "source_episode_id": ep_path.parent.name,
                        "split": "train" if index < ntrain else "validation",
                        "source_parquet_sha256": sha256(ep_path.parent / "frames.parquet")})
        offset += n
    whole = pa.concat_tables(tables)
    pq.write_table(whole, dest / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), dest / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="task")).to_parquet(dest / "meta/tasks.parquet")
    info = {"codebase_version": "v3.0", "robot_type": meta["robot_type"],
            "total_episodes": len(order), "total_frames": offset, "total_tasks": 1,
            "total_videos": 2 * len(order), "total_chunks": 1, "chunks_size": 1000,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": meta["fps"],
            "splits": {"train": f"0:{len(order)}"}, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    if offset != meta["frames"] - excluded_frames:
        raise ValueError("Total frame count differs from native manifest")
    stats = {"action": numeric_stats(np.concatenate(actions_train)),
             "observation.state": numeric_stats(np.concatenate(states_train))}
    # Visual normalization is IDENTITY; conventional image stats are recorded
    # solely for compatibility with LeRobot metadata readers.
    for key in [k for k in features if k.startswith("observation.images.")]:
        stats[key] = {"mean": [[[.5]], [[.5]], [[.5]]], "std": [[[.5]], [[.5]], [[.5]]],
                      "min": [[[0.]], [[0.]], [[0.]]], "max": [[[1.]], [[1.]], [[1.]]], "count": [1]}
    for name, payload in [("meta/info.json", info), ("meta/stats.json", stats),
                          ("apollo_manifest.json", meta),
                          ("export_manifest.json", {"source": str(source.resolve()),
                           "task": task_name, "train_episodes": ntrain, "validation_episodes": 5,
                           "validation_split_cli": .09, "episode_mapping": mapping,
                           "excluded_source_episode_ids": sorted(excluded_ids),
                           "excluded_source_frames": excluded_frames,
                           "action_transform": "identity", "image_transform": "identity",
                           "timing": "25Hz retained active frames; raw wallclock kept",
                           "normalization_source": "training episodes only"})]:
        (dest / name).write_text(json.dumps(payload, indent=2) + "\n")
    # Numeric round trip: each original non-language column is exactly preserved.
    recovered = pq.read_table(dest / "data/chunk-000/file-000.parquet")
    for m, ep in zip(mapping, episodes):
        original = pq.read_table(paths[m["source_index"]].parent / "frames.parquet").drop(["task"])
        actual = recovered.slice(ep["dataset_from_index"], ep["length"]).select(original.column_names)
        if not original.equals(actual):
            raise ValueError("Lossless numeric round trip failed")
    print(json.dumps({"dataset": str(dest), "episodes": len(order), "frames": offset,
                      "train": ntrain, "validation": 5, "roundtrip": "PASS"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-episode", action="append", default=[])
    args = p.parse_args()
    export_task(args.source, args.output, args.exclude_episode)
