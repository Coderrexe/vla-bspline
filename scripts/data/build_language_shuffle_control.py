"""Analyze or build a zero-copy semantic-shuffle control dataset.

The control changes only the string lookup in ``meta/tasks.parquet``.  Frame
``task_index`` values, episode boundaries, action/state observations, images,
and the temporal language-switch schedule remain byte-identical to the source
dataset.  Appended strings are permuted only among label ids having exactly the
same set of original-task supports.  Consequently every original task retains
its exact granular-caption vocabulary, while the caption assigned to an
eligible segment is wrong by construction.

This is deliberately a *legacy-language semantic-alignment* control.  It does
not repair the historical Molmo segment boundaries.

Examples (run ``analyze`` before allocating GPU training):

    python build_language_shuffle_control.py analyze \
      --source ~/scratch/vla_bspline/libero_l10gran \
      --seed 1701 --report /tmp/l10gran_shuffle_s1701_analysis.json

    python build_language_shuffle_control.py build \
      --source ~/scratch/vla_bspline/libero_l10gran \
      --out ~/scratch/vla_bspline/libero_l10gran_semshuffle_s1701 \
      --seed 1701

The derived root is an overlay of absolute symlinks to the immutable source;
only ``meta/tasks.parquet`` and the provenance manifest occupy new data blocks.
Moving or mutating the source invalidates the overlay, which is made explicit
in the manifest and checked by the training proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
CONTROL_KIND = "same-support-caption-derangement"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_task(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"episode has {len(value)} task strings; expected one")
        value = value[0]
    return str(value)


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _token_jaccard(left: str, right: str) -> float:
    a = set(_normalized_text(left).split())
    b = set(_normalized_text(right).split())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class TaskTable(NamedTuple):
    frame: pd.DataFrame
    strings: tuple[str, ...]
    storage: str


def read_task_table(path: Path) -> TaskTable:
    frame = pd.read_parquet(path)
    if frame.index.dtype == object:
        strings = tuple(str(value) for value in frame.index)
        storage = "index"
    elif "task" in frame.columns:
        strings = tuple(str(value) for value in frame["task"])
        storage = "task_column"
    else:
        object_columns = [name for name in frame.columns if frame[name].dtype == object]
        if len(object_columns) != 1:
            raise ValueError(
                "cannot locate task strings in tasks.parquet; expected an object index "
                "or a unique object-valued column"
            )
        storage = f"column:{object_columns[0]}"
        strings = tuple(str(value) for value in frame[object_columns[0]])
    if len(strings) != len(set(strings)):
        raise ValueError("task strings are not unique; a string permutation would collide")
    return TaskTable(frame=frame, strings=strings, storage=storage)


def write_task_table(table: TaskTable, strings: list[str], path: Path) -> None:
    if len(strings) != len(table.frame):
        raise ValueError("replacement task table has the wrong length")
    frame = table.frame.copy()
    if table.storage == "index":
        frame.index = pd.Index(strings, name=frame.index.name)
    elif table.storage == "task_column":
        frame.loc[:, "task"] = strings
    else:
        column = table.storage.split(":", 1)[1]
        frame.loc[:, column] = strings
    frame.to_parquet(path)


def _episode_task_map(root: Path, original_strings: tuple[str, ...]) -> dict[int, int]:
    episodes_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episodes_paths:
        raise FileNotFoundError("no episode metadata parquets found")
    original_id = {text: index for index, text in enumerate(original_strings)}
    result: dict[int, int] = {}
    for path in episodes_paths:
        frame = pd.read_parquet(path, columns=["episode_index", "tasks"])
        for row in frame.itertuples(index=False):
            episode = int(row.episode_index)
            task = _scalar_task(row.tasks)
            if task not in original_id:
                raise ValueError(f"episode {episode} has unknown original task {task!r}")
            if episode in result and result[episode] != original_id[task]:
                raise ValueError(f"episode {episode} has conflicting original tasks")
            result[episode] = original_id[task]
    return result


def scan_schedule(
    root: Path,
    original_task_count: int,
    task_strings: tuple[str, ...],
) -> dict[str, Any]:
    """Scan stale-safe episode/task support and canonical temporal runs."""
    episode_task = _episode_task_map(root, task_strings[:original_task_count])
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not data_paths:
        raise FileNotFoundError("no data parquets found")

    supports: dict[int, set[int]] = defaultdict(set)
    frame_by_label_task: dict[int, Counter[int]] = defaultdict(Counter)
    run_lengths_by_label_task: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_frames_by_task: Counter[int] = Counter()
    appended_frames_by_task: Counter[int] = Counter()
    appended_runs_by_task: Counter[int] = Counter()
    schedule_hash = hashlib.sha256()
    current: dict[int, tuple[int, int, int, int]] = {}
    # episode -> (last_frame, task_index, original_task, current_run_length)

    def flush(episode: int) -> None:
        _, label, original, length = current[episode]
        if label >= original_task_count:
            run_lengths_by_label_task[label][original].append(length)
            appended_runs_by_task[original] += 1

    for path in data_paths:
        frame = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "task_index"]
        )
        schedule_hash.update(str(path.relative_to(root)).encode() + b"\0")
        for episode, group in frame.groupby("episode_index", sort=False):
            episode = int(episode)
            if episode not in episode_task:
                raise ValueError(f"data references episode {episode} absent from metadata")
            original = episode_task[episode]
            group = group.sort_values("frame_index")
            frame_indices = group["frame_index"].to_numpy(dtype="<i8")
            task_indices = group["task_index"].to_numpy(dtype="<i8")
            if np.any(task_indices < 0) or np.any(task_indices >= len(task_strings)):
                raise ValueError(f"out-of-range task index in {path}")
            if len(frame_indices) and episode in current:
                last_frame = current[episode][0]
                if int(frame_indices[0]) <= last_frame:
                    raise ValueError(f"episode {episode} frames are duplicated or out of order")

            episode_column = np.full(len(group), episode, dtype="<i8")
            schedule_hash.update(episode_column.tobytes())
            schedule_hash.update(frame_indices.tobytes())
            schedule_hash.update(task_indices.tobytes())

            for frame_index, label in zip(frame_indices.tolist(), task_indices.tolist()):
                label = int(label)
                total_frames_by_task[original] += 1
                if label >= original_task_count:
                    supports[label].add(original)
                    frame_by_label_task[label][original] += 1
                    appended_frames_by_task[original] += 1
                prior = current.get(episode)
                if prior is None:
                    current[episode] = (int(frame_index), label, original, 1)
                elif label == prior[1] and int(frame_index) == prior[0] + 1:
                    current[episode] = (int(frame_index), label, original, prior[3] + 1)
                else:
                    flush(episode)
                    current[episode] = (int(frame_index), label, original, 1)

    for episode in list(current):
        flush(episode)

    return {
        "episode_count": len(episode_task),
        "data_parquet_count": len(data_paths),
        "schedule_sha256": schedule_hash.hexdigest(),
        "supports": {label: tuple(sorted(value)) for label, value in supports.items()},
        "frame_by_label_task": {
            label: dict(counts) for label, counts in frame_by_label_task.items()
        },
        "run_lengths_by_label_task": {
            label: {task: lengths for task, lengths in by_task.items()}
            for label, by_task in run_lengths_by_label_task.items()
        },
        "total_frames_by_task": dict(total_frames_by_task),
        "appended_frames_by_task": dict(appended_frames_by_task),
        "appended_runs_by_task": dict(appended_runs_by_task),
    }


def make_same_support_derangement(
    supports: dict[int, tuple[int, ...]],
    appended_ids: list[int],
    seed: int,
) -> tuple[dict[int, int], dict[tuple[int, ...], list[int]]]:
    """Return destination-id -> source-string-id, fixing unsupported/singletons."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for label in appended_ids:
        support = supports.get(label, ())
        groups[support].append(label)

    mapping = {label: label for label in appended_ids}
    rng = np.random.default_rng(seed)
    for support in sorted(groups):
        ids = sorted(groups[support])
        # Unused strings and singleton support classes have no rigorous
        # same-support counterfactual, so keep them unchanged and report them.
        if not support or len(ids) < 2:
            continue
        cycle = np.asarray(ids, dtype=np.int64)
        rng.shuffle(cycle)
        for index, destination in enumerate(cycle.tolist()):
            mapping[destination] = int(cycle[(index + 1) % len(cycle)])
    return mapping, groups


def analyze(source: Path, original_task_count: int, seed: int) -> dict[str, Any]:
    source = source.expanduser().resolve()
    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    table = read_task_table(tasks_path)
    if not 0 < original_task_count < len(table.strings):
        raise ValueError("original-task count is outside the task table")

    scan = scan_schedule(source, original_task_count, table.strings)
    appended_ids = list(range(original_task_count, len(table.strings)))
    mapping, groups = make_same_support_derangement(scan["supports"], appended_ids, seed)
    changed_ids = {label for label, source_id in mapping.items() if label != source_id}
    used_ids = set(scan["supports"])
    eligible_ids = changed_ids & used_ids
    fixed_used_ids = used_ids - eligible_ids

    eligible_frames = sum(
        sum(scan["frame_by_label_task"][label].values()) for label in eligible_ids
    )
    appended_frames = sum(scan["appended_frames_by_task"].values())
    eligible_runs = sum(
        len(lengths)
        for label in eligible_ids
        for lengths in scan["run_lengths_by_label_task"][label].values()
    )
    appended_runs = sum(scan["appended_runs_by_task"].values())

    pairs = []
    normalized_collisions = 0
    high_overlap = 0
    for destination in sorted(changed_ids):
        source_id = mapping[destination]
        destination_text = table.strings[destination]
        source_text = table.strings[source_id]
        similarity = _token_jaccard(destination_text, source_text)
        normalized_collision = _normalized_text(destination_text) == _normalized_text(source_text)
        normalized_collisions += int(normalized_collision)
        high_overlap += int(similarity >= 0.8)
        pairs.append(
            {
                "task_index": destination,
                "original_label": destination_text,
                "shuffled_label_source_index": source_id,
                "shuffled_label": source_text,
                "original_task_support": list(scan["supports"].get(destination, ())),
                "token_jaccard": similarity,
                "normalized_text_collision": normalized_collision,
            }
        )

    by_task = []
    for task in range(original_task_count):
        task_appended = int(scan["appended_frames_by_task"].get(task, 0))
        task_eligible = sum(
            counts.get(task, 0)
            for label, counts in scan["frame_by_label_task"].items()
            if label in eligible_ids
        )
        by_task.append(
            {
                "task_id": task,
                "task": table.strings[task],
                "total_frames": int(scan["total_frames_by_task"].get(task, 0)),
                "appended_frames": task_appended,
                "deranged_frames": task_eligible,
                "deranged_fraction_of_appended": (
                    task_eligible / task_appended if task_appended else 0.0
                ),
            }
        )

    # Formal invariants of the task-table-only intervention.
    mapped_values = [mapping[label] for label in appended_ids]
    if set(mapped_values) != set(appended_ids) or len(mapped_values) != len(set(mapped_values)):
        raise AssertionError("caption mapping is not bijective")
    for destination in changed_ids:
        source_id = mapping[destination]
        if destination == source_id:
            raise AssertionError("changed mapping contains a fixed point")
        if scan["supports"][destination] != scan["supports"][source_id]:
            raise AssertionError("mapping crosses an original-task support class")

    group_rows = [
        {
            "original_task_support": list(support),
            "label_count": len(ids),
            "used": bool(support),
            "deranged": bool(support) and len(ids) >= 2,
        }
        for support, ids in sorted(groups.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "control_kind": CONTROL_KIND,
        "source_root": str(source),
        "seed": seed,
        "original_task_count": original_task_count,
        "total_task_string_count": len(table.strings),
        "appended_label_count": len(appended_ids),
        "used_appended_label_count": len(used_ids),
        "deranged_used_label_count": len(eligible_ids),
        "fixed_used_label_count": len(fixed_used_ids),
        "appended_frame_count": appended_frames,
        "deranged_frame_count": eligible_frames,
        "deranged_fraction_of_appended_frames": (
            eligible_frames / appended_frames if appended_frames else 0.0
        ),
        "appended_run_count": appended_runs,
        "deranged_run_count": eligible_runs,
        "deranged_fraction_of_appended_runs": (
            eligible_runs / appended_runs if appended_runs else 0.0
        ),
        "normalized_text_collision_count": normalized_collisions,
        "high_token_overlap_pair_count": high_overlap,
        "task_index_schedule_sha256": scan["schedule_sha256"],
        "source_tasks_sha256": sha256_file(tasks_path),
        "source_episode_count": scan["episode_count"],
        "source_data_parquet_count": scan["data_parquet_count"],
        "invariants": {
            "task_index_schedule_unchanged": True,
            "frame_and_run_boundaries_unchanged": True,
            "appended_string_mapping_bijective": True,
            "changed_mapping_has_no_string_fixed_points": True,
            "mapping_stays_within_exact_original_task_support": True,
            "per_original_task_granular_vocabulary_preserved": True,
            "per_original_task_caption_frequency_and_run_length_multisets_preserved": True,
            "global_caption_vocabulary_preserved": True,
            "global_frame_run_length_and_frequency_multisets_preserved": True,
        },
        "interpretation": {
            "estimand": (
                "performance(correct legacy granular-caption alignment) minus "
                "performance(same-support shuffled-caption alignment), holding all "
                "frames and language-switch timing fixed"
            ),
            "scope": (
                "intention-to-treat over the full suite; singleton support classes "
                "remain correctly labeled and dilute the contrast"
            ),
            "does_not_test": (
                "production-exact event boundaries, caption quality, or B-spline "
                "specificity unless crossed with both policy heads"
            ),
        },
        "support_groups": group_rows,
        "per_original_task": by_task,
        "mapping": pairs,
        "fixed_used_labels": [
            {
                "task_index": label,
                "label": table.strings[label],
                "original_task_support": list(scan["supports"][label]),
            }
            for label in sorted(fixed_used_ids)
        ],
    }


def write_json_no_overwrite(payload: dict[str, Any], path: Path) -> None:
    path = path.expanduser()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_overlay(
    source: Path,
    output: Path,
    original_task_count: int,
    seed: int,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    requested_output = output.expanduser().absolute()
    output = requested_output.parent.resolve() / requested_output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    if source == output or source in output.parents:
        raise ValueError("output must not be inside the source dataset")
    if (source / "_semantic_shuffle_control").exists():
        raise ValueError("refusing to derive a shuffle from an existing shuffle overlay")
    output.parent.mkdir(parents=True, exist_ok=True)

    analysis = analyze(source, original_task_count, seed)
    table = read_task_table(source / "meta" / "tasks.parquet")
    replacement = list(table.strings)
    for pair in analysis["mapping"]:
        replacement[pair["task_index"]] = pair["shuffled_label"]

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        (temporary / "meta").mkdir()
        for child in source.iterdir():
            if child.name == "meta":
                continue
            os.symlink(str(child.resolve()), temporary / child.name, target_is_directory=child.is_dir())
        for child in (source / "meta").iterdir():
            if child.name == "tasks.parquet":
                continue
            os.symlink(
                str(child.resolve()),
                temporary / "meta" / child.name,
                target_is_directory=child.is_dir(),
            )

        tasks_out = temporary / "meta" / "tasks.parquet"
        write_task_table(table, replacement, tasks_out)
        manifest_dir = temporary / "_semantic_shuffle_control"
        manifest_dir.mkdir()
        analysis["derived_root"] = str(output)
        analysis["derived_tasks_sha256"] = sha256_file(tasks_out)
        analysis["overlay_symlink_targets"] = {
            child.name: str(child.resolve()) for child in source.iterdir() if child.name != "meta"
        }
        analysis["builder_sha256"] = sha256_file(Path(__file__).resolve())
        analysis["source_dependency_warning"] = (
            "This zero-copy overlay is valid only while source_root and its symlink "
            "targets remain immutable and available. Verify source_tasks_sha256 and "
            "task_index_schedule_sha256 before every training run."
        )
        manifest = manifest_dir / "manifest.json"
        manifest.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        tasks_out.chmod(0o444)
        manifest.chmod(0o444)
        temporary.rename(output)
    except BaseException:
        # The target is an explicitly generated, unique temporary sibling.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "build"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--source", type=Path, required=True)
        sub.add_argument("--original-tasks", type=int, default=40)
        sub.add_argument("--seed", type=int, default=1701)
        if command == "analyze":
            sub.add_argument("--report", type=Path, required=True)
        else:
            sub.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        result = analyze(args.source, args.original_tasks, args.seed)
        write_json_no_overwrite(result, args.report)
        print(
            f"deranged {result['deranged_frame_count']}/"
            f"{result['appended_frame_count']} appended frames "
            f"({100 * result['deranged_fraction_of_appended_frames']:.1f}%)"
        )
        print("analysis ->", args.report)
    else:
        result = build_overlay(args.source, args.out, args.original_tasks, args.seed)
        print(
            f"built {args.out}: deranged {result['deranged_frame_count']}/"
            f"{result['appended_frame_count']} appended frames"
        )


if __name__ == "__main__":
    main()
