"""Build an immutable, coherent event-language overlay for LeRobot v3 data.

Unlike the historical ``build_lang_dataset.py`` recipe, this builder mixes at
the *episode* level.  A demonstration therefore uses either its original
whole-task instruction throughout or corrected event-aligned clauses
throughout; it never flips randomly between the two conditioning regimes at
every segment.  Only ``task_index`` and the task table are changed.

The input labels must carry the corrected production boundary version emitted
by :mod:`scripts.data.molmo_segment_labels`.  Each labeled episode must be a
gap-free, non-overlapping partition of its real ``frame_index`` range.  The
output is assembled in a sibling temporary directory and atomically renamed,
and a content-hashed provenance record is stored under ``meta/``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_BOUNDARY_VERSION = "event_targets_k_exclusive_window_quantile_v1"
PROVENANCE_SCHEMA = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no labels in {path}")
    seen = set()
    for line_number, row in enumerate(rows, start=1):
        required = {
            "episode_index", "seg_start", "seg_end", "task_index", "task",
            "label", "boundary_version", "boundary_config", "label_mode",
            "prompt_version", "model_id",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields {missing}")
        if row["boundary_version"] != EXPECTED_BOUNDARY_VERSION:
            raise ValueError(
                f"{path}:{line_number} boundary_version={row['boundary_version']!r}, "
                f"expected {EXPECTED_BOUNDARY_VERSION!r}"
            )
        key = (int(row["episode_index"]), int(row["seg_start"]))
        if key in seen:
            raise ValueError(f"duplicate episode/start key {key} in {path}")
        seen.add(key)
        if int(row["seg_end"]) <= int(row["seg_start"]):
            raise ValueError(f"non-positive interval at {path}:{line_number}")
        if not str(row["label"]).strip():
            raise ValueError(f"empty label at {path}:{line_number}")
    rows.sort(key=lambda row: (int(row["episode_index"]), int(row["seg_start"])))
    return rows


def choose_granular_episodes(
    rows: list[dict], keep_original_fraction: float, seed: int
) -> set[int]:
    """Choose a task-stratified set of fully granular episodes."""

    if not 0.0 <= keep_original_fraction <= 1.0:
        raise ValueError("keep_original_fraction must lie in [0, 1]")
    by_task: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_task[int(row["task_index"])].add(int(row["episode_index"]))
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for task_index in sorted(by_task):
        episodes = np.asarray(sorted(by_task[task_index]), dtype=np.int64)
        episodes = episodes[rng.permutation(len(episodes))]
        count = int(round((1.0 - keep_original_fraction) * len(episodes)))
        selected.update(map(int, episodes[:count]))
    return selected


def validate_episode_partition(rows: list[dict], frame_count: int, episode: int) -> None:
    intervals = [(int(row["seg_start"]), int(row["seg_end"])) for row in rows]
    expected_start = 0
    for start, end in intervals:
        if start != expected_start:
            raise ValueError(
                f"episode {episode} label partition gap/overlap: expected start "
                f"{expected_start}, found [{start}, {end})"
            )
        expected_start = end
    if expected_start != frame_count:
        raise ValueError(
            f"episode {episode} labels end at {expected_start}, data has {frame_count} frames"
        )


def _task_strings(tasks: pd.DataFrame) -> list[str]:
    if tasks.index.dtype == object:
        return list(map(str, tasks.index))
    return list(map(str, tasks.iloc[:, 0]))


def _write_tasks(path: Path, source: pd.DataFrame, strings: list[str]) -> None:
    if source.index.dtype == object:
        column = source.columns[0] if len(source.columns) else "task_index"
        output = pd.DataFrame(
            {column: range(len(strings))},
            index=pd.Index(strings, name=source.index.name),
        )
    else:
        output = pd.DataFrame({source.columns[0]: strings})
    output.to_parquet(path)


def build(args) -> dict:
    source = Path(args.root).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    temporary = output.with_name(output.name + f".building-{os.getpid()}")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {temporary}")
    if output.parent.stat().st_dev != source.stat().st_dev:
        raise ValueError("source and output must share a filesystem for hardlinks")

    labels = read_labels(labels_path)
    metadata_sets = {
        key: sorted({json.dumps(row[key], sort_keys=True) for row in labels})
        for key in (
            "boundary_version", "boundary_config", "label_mode", "prompt_version", "model_id"
        )
    }
    for key, values in metadata_sets.items():
        if len(values) != 1:
            raise ValueError(f"labels mix {key}: {values}")

    tasks_path = source / "meta" / "tasks.parquet"
    info_path = source / "meta" / "info.json"
    source_tasks = pd.read_parquet(tasks_path)
    original_strings = _task_strings(source_tasks)
    for row in labels:
        task_index = int(row["task_index"])
        if task_index >= len(original_strings) or original_strings[task_index] != row["task"]:
            raise ValueError(
                f"label task identity mismatch: index={task_index}, task={row['task']!r}"
            )

    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in labels:
        by_episode[int(row["episode_index"])].append(row)
    granular_episodes = choose_granular_episodes(
        labels, args.keep_original_fraction, args.seed
    )

    # Mirror with hardlinks first; only rewritten files are unlinked below.
    for directory, _, filenames in os.walk(source):
        relative = os.path.relpath(directory, source)
        target_dir = temporary / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_file = Path(directory) / filename
            target_file = target_dir / filename
            resolved = source_file.resolve()
            try:
                os.link(resolved, target_file)
            except OSError:
                shutil.copy2(resolved, target_file)

    unique_labels = sorted({str(row["label"]) for row in labels})
    label_id = {label: len(original_strings) + index for index, label in enumerate(unique_labels)}
    seen_selected = set()
    rewritten_frames = 0
    data_files = sorted(source.glob("data/**/*.parquet"))
    data_hashes = {}
    for source_file in data_files:
        relative = source_file.relative_to(source)
        data_hashes[str(relative)] = sha256_file(source_file)
        frame = pd.read_parquet(source_file)
        present = set(map(int, frame["episode_index"].unique())) & granular_episodes
        if not present:
            continue
        task_indices = frame["task_index"].to_numpy(copy=True)
        for episode in sorted(present):
            episode_rows = frame[frame["episode_index"] == episode]
            local_frames = episode_rows["frame_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(local_frames, np.arange(len(local_frames))):
                raise ValueError(f"episode {episode} frame_index is not contiguous from zero")
            partition = by_episode[episode]
            validate_episode_partition(partition, len(local_frames), episode)
            positions = np.flatnonzero(frame["episode_index"].to_numpy() == episode)
            rewritten = task_indices[positions].copy()
            for row in partition:
                start, end = int(row["seg_start"]), int(row["seg_end"])
                rewritten[start:end] = label_id[str(row["label"])]
            task_indices[positions] = rewritten
            rewritten_frames += len(positions)
            seen_selected.add(episode)
        target = temporary / relative
        target.unlink()
        frame["task_index"] = task_indices
        frame.to_parquet(target, index=False)
    if seen_selected != granular_episodes:
        raise ValueError(
            f"selected episodes missing from data: {sorted(granular_episodes - seen_selected)}"
        )

    all_strings = original_strings + unique_labels
    output_tasks = temporary / "meta" / "tasks.parquet"
    output_tasks.unlink()
    _write_tasks(output_tasks, source_tasks, all_strings)

    output_info = temporary / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    output_info.unlink()
    if "total_tasks" in info:
        info["total_tasks"] = len(all_strings)
    output_info.write_text(json.dumps(info, indent=2) + "\n")

    script_path = Path(__file__).resolve()
    per_task = defaultdict(lambda: {"labeled": set(), "granular": set()})
    for row in labels:
        task = int(row["task_index"])
        episode = int(row["episode_index"])
        per_task[task]["labeled"].add(episode)
        if episode in granular_episodes:
            per_task[task]["granular"].add(episode)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "source_root": str(source),
        "output_root": str(output),
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "builder_sha256": sha256_file(script_path),
        "source_info_sha256": sha256_file(info_path),
        "source_tasks_sha256": sha256_file(tasks_path),
        "source_data_sha256": data_hashes,
        "output_info_sha256": sha256_file(output_info),
        "output_tasks_sha256": sha256_file(output_tasks),
        "expected_boundary_version": EXPECTED_BOUNDARY_VERSION,
        "label_metadata": {key: json.loads(values[0]) for key, values in metadata_sets.items()},
        "mix": {
            "unit": "episode",
            "seed": args.seed,
            "keep_original_fraction_requested": args.keep_original_fraction,
            "labeled_episodes": len(by_episode),
            "granular_episodes": len(granular_episodes),
        },
        "per_source_task": {
            str(task): {
                "labeled_episodes": len(values["labeled"]),
                "granular_episodes": len(values["granular"]),
            }
            for task, values in sorted(per_task.items())
        },
        "label_records": len(labels),
        "unique_language_clauses": len(unique_labels),
        "rewritten_frames": rewritten_frames,
    }
    provenance_path = temporary / "meta" / "event_language_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    os.rename(temporary, output)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="exact source LeRobot dataset root")
    parser.add_argument("--labels", required=True, help="corrected seg_labels.jsonl")
    parser.add_argument("--out", required=True, help="new immutable output root")
    parser.add_argument("--keep_original_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260821)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
