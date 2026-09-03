#!/usr/bin/env python3
"""Build a symlink overlay with official per-frame RoboCasa language labels.

Only ``task_index`` and its task vocabulary are changed. All observations,
actions, rewards, episode metadata, and videos are either value-identical
rewrites or immutable symlinks to the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_action_signature(table: pa.Table) -> str:
    action = np.asarray(table["action"].to_pylist(), dtype="<f8")
    frame = np.asarray(table["frame_index"].to_pylist(), dtype="<i8")
    digest = hashlib.sha256()
    digest.update(action.tobytes())
    digest.update(frame.tobytes())
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule_manifest", type=Path, required=True)
    parser.add_argument("--action_signatures", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    for required in (
        source / "meta/info.json",
        source / "meta/tasks.parquet",
        source / "data",
        args.schedule,
        args.schedule_manifest,
        args.action_signatures,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    schedule_manifest = json.loads(args.schedule_manifest.read_text())
    if schedule_manifest.get("schema") not in {
        "robocasa_official_kettle_schedule_v1",
        "robocasa_official_phase_schedule_v1",
    }:
        raise RuntimeError("unexpected schedule manifest")
    if sha256(args.schedule) != schedule_manifest["schedule_sha256"]:
        raise RuntimeError("schedule hash mismatch")
    if sha256(args.action_signatures) != schedule_manifest["action_signature_manifest_sha256"]:
        raise RuntimeError("action-signature manifest hash mismatch")
    signatures = json.loads(args.action_signatures.read_text())
    if signatures.get("schema") != "canonical_action_frame_v1":
        raise RuntimeError("unexpected action-signature schema")
    episode_start, episode_end = map(int, schedule_manifest["combined_episode_range"])
    expected_episodes = int(schedule_manifest["episodes"])
    if episode_end - episode_start + 1 != expected_episodes:
        raise RuntimeError("schedule episode range/count mismatch")

    schedule = pq.read_table(args.schedule)
    required_schedule = {
        "episode_index", "frame_index", "instruction", "atomic_skill", "stage", "subtask_idx"
    }
    if set(schedule.column_names) != required_schedule:
        raise RuntimeError(f"unexpected schedule columns: {schedule.column_names}")
    if len(schedule) != int(schedule_manifest["frames"]):
        raise RuntimeError("schedule frame count mismatch")
    schedule_rows = schedule.to_pylist()
    schedule_by_key: dict[tuple[int, int], str] = {}
    for row in schedule_rows:
        key = (int(row["episode_index"]), int(row["frame_index"]))
        if key in schedule_by_key:
            raise RuntimeError(f"duplicate schedule key {key}")
        schedule_by_key[key] = str(row["instruction"])
    if len(schedule_by_key) != len(schedule):
        raise RuntimeError("schedule is not one-to-one")

    source_tasks = pq.read_table(source / "meta/tasks.parquet")
    task_rows = source_tasks.to_pylist()
    text_to_index = {str(row["task"]): int(row["task_index"]) for row in task_rows}
    next_index = max(text_to_index.values()) + 1
    for instruction in sorted(set(schedule["instruction"].to_pylist())):
        if instruction not in text_to_index:
            text_to_index[instruction] = next_index
            task_rows.append({"task_index": next_index, "task": instruction})
            next_index += 1
    output_tasks = pa.Table.from_pylist(task_rows, schema=source_tasks.schema)

    destination.mkdir(parents=True)
    (destination / "meta").mkdir()
    (destination / "data").mkdir()
    for item in source.iterdir():
        if item.name not in {"meta", "data"}:
            link(item, destination / item.name)
    for item in (source / "meta").iterdir():
        if item.name not in {"info.json", "tasks.parquet"}:
            link(item, destination / "meta" / item.name)
    pq.write_table(output_tasks, destination / "meta/tasks.parquet")
    info = json.loads((source / "meta/info.json").read_text())
    info["total_tasks"] = len(task_rows)
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")

    signature_by_source_episode = {
        int(row["episode_index"]) + episode_start: (int(row["frames"]), row["signature"])
        for row in signatures["episodes"]
    }
    if len(signature_by_source_episode) != expected_episodes:
        raise RuntimeError("signature episode count mismatch")
    rewritten_files: list[str] = []
    observed_keys: set[tuple[int, int]] = set()
    physical_columns: list[str] | None = None
    for source_file in sorted((source / "data").glob("chunk-*/*.parquet")):
        relative = source_file.relative_to(source / "data")
        output_file = destination / "data" / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        episode_column = pq.read_table(source_file, columns=["episode_index"])["episode_index"]
        episode_values = np.asarray(episode_column.to_pylist(), dtype=np.int64)
        affected = (episode_values >= episode_start) & (episode_values <= episode_end)
        if not bool(affected.any()):
            link(source_file, output_file)
            continue
        table = pq.read_table(source_file)
        if physical_columns is None:
            physical_columns = [name for name in table.column_names if name != "task_index"]
        frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        for row_index in np.flatnonzero(affected):
            key = (int(episode_values[row_index]), int(frame_values[row_index]))
            try:
                instruction = schedule_by_key[key]
            except KeyError as error:
                raise RuntimeError(f"source frame is absent from official schedule: {key}") from error
            observed_keys.add(key)
            task_values[row_index] = text_to_index[instruction]
        task_field_index = table.schema.get_field_index("task_index")
        replacement = pa.array(task_values, type=table.schema.field("task_index").type)
        output_table = table.set_column(task_field_index, "task_index", replacement)
        if not table.select(physical_columns).equals(output_table.select(physical_columns)):
            raise RuntimeError(f"non-language data changed in {source_file}")
        pq.write_table(output_table, output_file, compression="zstd")
        rewritten_files.append(str(relative))

    if observed_keys != set(schedule_by_key):
        missing = sorted(set(schedule_by_key) - observed_keys)[:5]
        raise RuntimeError(f"schedule/source coverage mismatch; missing examples={missing}")

    # Re-read exact source episodes and bind the official archive's physical stream.
    source_dataset = ds.dataset(source / "data", format="parquet")
    selected = source_dataset.to_table(
        columns=["action", "episode_index", "frame_index"],
        filter=(ds.field("episode_index") >= episode_start)
        & (ds.field("episode_index") <= episode_end),
    )
    for episode, expected in sorted(signature_by_source_episode.items()):
        mask = pc.equal(selected["episode_index"], episode)
        episode_table = selected.filter(mask)
        episode_table = episode_table.sort_by("frame_index")
        observed = (len(episode_table), canonical_action_signature(episode_table))
        if observed != expected:
            raise RuntimeError(f"official/source action mismatch for episode {episode}")

    manifest = {
        "schema": "robocasa_official_language_overlay_v2",
        "source_dataset": str(source),
        "destination": str(destination.resolve()),
        "source_info_sha256": sha256(source / "meta/info.json"),
        "source_tasks_sha256": sha256(source / "meta/tasks.parquet"),
        "schedule_sha256": sha256(args.schedule),
        "schedule_manifest_sha256": sha256(args.schedule_manifest),
        "action_signatures_sha256": sha256(args.action_signatures),
        "task": schedule_manifest.get("task", signatures.get("task")),
        "episode_range": [episode_start, episode_end],
        "episodes": expected_episodes,
        "frames": len(schedule),
        "physical_action_stream_matches_official": True,
        "changed_column": "task_index",
        "rewritten_data_files": rewritten_files,
        "added_task_strings": sorted(set(schedule["instruction"].to_pylist())),
        "output_info_sha256": sha256(destination / "meta/info.json"),
        "output_tasks_sha256": sha256(destination / "meta/tasks.parquet"),
    }
    manifest_path = destination / "official_language_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
