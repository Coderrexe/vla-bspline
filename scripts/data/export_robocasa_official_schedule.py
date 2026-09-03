#!/usr/bin/env python3
"""Export audited official RoboCasa annotations into a combined-data schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_signature(actions: list[list[float]], frame_index: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(actions, dtype="<f8").tobytes())
    digest.update(np.asarray(frame_index, dtype="<i8").tobytes())
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--combined_start", type=int, required=True)
    parser.add_argument("--source_archive_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.official_root.resolve()
    outputs = {
        "schedule": args.output_dir / f"robocasa_official_{args.task}_schedule.parquet",
        "signatures": args.output_dir / f"robocasa_official_{args.task}_action_signatures.json",
        "manifest": args.output_dir / f"robocasa_official_{args.task}_schedule_manifest.json",
    }
    if args.output_dir.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len(args.source_archive_sha256) != 64:
        raise ValueError("source archive SHA256 must contain 64 hexadecimal digits")

    vocabulary: dict[int, str] = {}
    with (root / "meta/tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            vocabulary[int(row["task_index"])] = str(row["task"])
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise RuntimeError("official dataset has no episode Parquets")

    schedule_rows: list[dict[str, object]] = []
    signatures: list[dict[str, object]] = []
    sequence_counts: Counter[tuple[tuple[int, str, str, str], ...]] = Counter()
    phase_durations: dict[str, list[int]] = defaultdict(list)
    toggle_offsets: list[int] = []
    exact_toggle_boundaries = 0
    total_semantic_boundaries = 0
    total_frames = 0
    for expected_episode, path in enumerate(files):
        table = pq.read_table(
            path,
            columns=[
                "episode_index", "frame_index", "action",
                "annotation.human.subtask", "annotation.human.subtask_name",
                "annotation.human.subtask_stage", "subtask_idx",
            ],
        )
        data = table.to_pydict()
        episode_values = {int(value) for value in data["episode_index"]}
        if episode_values != {expected_episode}:
            raise RuntimeError(f"non-contiguous episode mapping in {path}: {episode_values}")
        frame_index = [int(value) for value in data["frame_index"]]
        if frame_index != list(range(len(table))):
            raise RuntimeError(f"non-contiguous frames in {path}")
        combined_episode = args.combined_start + expected_episode
        sequence: list[tuple[int, str, str, str]] = []
        previous: tuple[int, str, str, str] | None = None
        starts: list[int] = []
        for row_index in range(len(table)):
            instruction = vocabulary[int(data["annotation.human.subtask"][row_index])]
            skill = vocabulary[int(data["annotation.human.subtask_name"][row_index])]
            stage = vocabulary[int(data["annotation.human.subtask_stage"][row_index])]
            subtask_idx = int(data["subtask_idx"][row_index])
            key = (subtask_idx, instruction, skill, stage)
            if key != previous:
                sequence.append(key)
                starts.append(row_index)
                previous = key
            schedule_rows.append(
                {
                    "episode_index": combined_episode,
                    "frame_index": row_index,
                    "instruction": instruction,
                    "atomic_skill": skill,
                    "stage": stage,
                    "subtask_idx": subtask_idx,
                }
            )
        sequence_counts[tuple(sequence)] += 1
        ends = starts[1:] + [len(table)]
        for phase_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            phase_durations[sequence[phase_index][1]].append(end - start)
        gripper = np.asarray([action[11] for action in data["action"]], dtype=np.float64)
        toggles = np.flatnonzero(np.sign(gripper[1:]) != np.sign(gripper[:-1])) + 1
        total_semantic_boundaries += len(starts) - 1
        for boundary in starts[1:]:
            if len(toggles):
                offset = int(np.min(np.abs(toggles - boundary)))
                toggle_offsets.append(offset)
                exact_toggle_boundaries += int(offset == 0)
        signatures.append(
            {
                "episode_index": expected_episode,
                "frames": len(table),
                "signature": action_signature(data["action"], frame_index),
            }
        )
        total_frames += len(table)

    if len(sequence_counts) != 1:
        raise RuntimeError(f"expected one semantic sequence, got {sequence_counts}")
    sequence, episode_count = next(iter(sequence_counts.items()))
    args.output_dir.mkdir(parents=True)
    schedule = pa.Table.from_pylist(schedule_rows)
    pq.write_table(schedule, outputs["schedule"], compression="zstd")
    signatures_payload = {
        "schema": "canonical_action_frame_v1",
        "task": args.task,
        "episodes": signatures,
    }
    outputs["signatures"].write_text(
        json.dumps(signatures_payload, indent=2, sort_keys=True) + "\n"
    )
    offsets = np.asarray(toggle_offsets, dtype=np.float64)
    manifest = {
        "schema": "robocasa_official_phase_schedule_v1",
        "task": args.task,
        "source_archive_sha256": args.source_archive_sha256,
        "episodes": len(files),
        "frames": total_frames,
        "combined_episode_range": [args.combined_start, args.combined_start + len(files) - 1],
        "semantic_sequence": [
            {"subtask_idx": idx, "instruction": text, "atomic_skill": skill, "stage": stage}
            for idx, text, skill, stage in sequence
        ],
        "sequence_episode_count": episode_count,
        "phase_duration_steps": {
            phase: quantiles(values) for phase, values in sorted(phase_durations.items())
        },
        "nearest_gripper_toggle_offset": {
            **quantiles(toggle_offsets),
            "semantic_boundaries": total_semantic_boundaries,
            "boundaries_with_any_toggle": len(toggle_offsets),
            "boundaries_without_any_toggle": total_semantic_boundaries - len(toggle_offsets),
            "exact_fraction_all_boundaries": exact_toggle_boundaries / total_semantic_boundaries,
            "within_one_fraction_all_boundaries": float(np.sum(offsets <= 1)) / total_semantic_boundaries,
        },
        "schedule_sha256": sha256(outputs["schedule"]),
        "action_signature_manifest_sha256": sha256(outputs["signatures"]),
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
