"""Build paired original/clause LIBERO datasets with one causal switch.

This rescue builder deliberately does not use numeric dataset task IDs as
LIBERO suite IDs.  It resolves the two evaluator tasks by their exact task
descriptions, records that mapping, and fails if it is not one-to-one.

For every selected successful demonstration, the clause switch is the first
open command after a sustained closed-gripper run.  The release action remains
conditioned on clause 1; clause 2 starts at the following observation/action
row, matching the causal closed-loop event-clock evaluator.

Both outputs are new, immutable roots.  The paired-original root supplies the
``meta/tasks.parquet`` missing from the upstream snapshot but otherwise
hardlinks the source.  The clause root changes only language metadata and the
``task_index`` column.  A column-wise Arrow IPC audit proves that every selected
non-language data column is byte-identical after the Parquet round trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
BOUNDARY_VERSION = "causal_first_sustained_release_v1"
DEFAULT_MIN_CLOSED_RUN = 48

# The keys are evaluator suite task IDs, never dataset task_index values.
TARGETS: dict[int, dict[str, Any]] = {
    0: {
        "evaluator_task": "put both the alphabet soup and the tomato sauce in the basket",
        "clauses": [
            "put the alphabet soup in the basket",
            "put the tomato sauce in the basket",
        ],
    },
    4: {
        "evaluator_task": (
            "put the white mug on the left plate and put the yellow and white mug "
            "on the right plate"
        ),
        "clauses": [
            "put the white mug on the left plate",
            "put the yellow and white mug on the right plate",
        ],
    },
}


def load_targets(path: str | None) -> dict[int, dict[str, Any]]:
    """Load an optional exact-task specification without changing build semantics."""

    if path is None:
        return TARGETS
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text())
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError("target spec must contain a nonempty 'targets' object")
    targets: dict[int, dict[str, Any]] = {}
    for raw_task_id, raw_spec in raw_targets.items():
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid suite task ID: {raw_task_id!r}") from error
        if task_id < 0 or str(task_id) != str(raw_task_id):
            raise ValueError(f"suite task ID must be a canonical nonnegative integer: {raw_task_id!r}")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"target t{task_id} must be an object")
        evaluator_task = raw_spec.get("evaluator_task")
        clauses = raw_spec.get("clauses")
        if not isinstance(evaluator_task, str) or not evaluator_task.strip():
            raise ValueError(f"target t{task_id} has no evaluator_task")
        if (
            not isinstance(clauses, list)
            or len(clauses) != 2
            or not all(isinstance(clause, str) and clause.strip() for clause in clauses)
            or clauses[0] == clauses[1]
        ):
            raise ValueError(f"target t{task_id} must contain two distinct nonempty clauses")
        targets[task_id] = {
            "evaluator_task": evaluator_task,
            "clauses": list(clauses),
        }
    task_texts = [spec["evaluator_task"] for spec in targets.values()]
    clause_texts = [clause for spec in targets.values() for clause in spec["clauses"]]
    if len(set(task_texts)) != len(task_texts):
        raise ValueError("target evaluator_task strings must be unique")
    if len(set(clause_texts)) != len(clause_texts):
        raise ValueError("clause strings must be unique across targets")
    return targets


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_valid_release(gripper: np.ndarray, min_closed_run: int) -> dict[str, Any]:
    """Return the first release preceded by a sustained closed command run."""

    values = np.asarray(gripper).reshape(-1)
    if len(values) < 2:
        raise ValueError("an episode needs at least two actions")
    if min_closed_run < 1:
        raise ValueError("min_closed_run must be positive")
    releases = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0)) + 1
    candidates = []
    for release in releases:
        run_start = int(release) - 1
        while run_start > 0 and values[run_start - 1] > 0:
            run_start -= 1
        closed_run = int(release) - run_start
        candidates.append(
            {
                "release_action_index": int(release),
                "closed_run_start": run_start,
                "closed_run_actions": closed_run,
                "valid": closed_run >= min_closed_run,
            }
        )
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError(
            f"no close->open release after >= {min_closed_run} closed commands; "
            f"candidates={candidates}"
        )
    chosen = dict(valid[0])
    # action[release] opens the gripper under clause 1.  The next row is the
    # first observation/action pair conditioned on clause 2.
    chosen["switch_frame"] = chosen["release_action_index"] + 1
    chosen["release_candidates"] = candidates
    return chosen


def _episode_files(root: Path) -> list[Path]:
    files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata parquet under {root}")
    return files


def _read_episode_metadata(root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in _episode_files(root)]
    result = pd.concat(frames, ignore_index=True).sort_values("episode_index")
    if result["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index in metadata")
    return result


def _one_task(tasks: Any, episode: int) -> str:
    values = list(tasks) if not isinstance(tasks, str) else [tasks]
    if len(values) != 1:
        raise ValueError(f"source episode {episode} has tasks={values}, expected one")
    return str(values[0])


def resolve_task_mapping(root: Path) -> tuple[dict[int, str], pd.DataFrame, pd.DataFrame]:
    """Resolve dataset task IDs from exact episode task strings."""

    episodes = _read_episode_metadata(root)
    task_rows = (
        pads.dataset(str(root / "data"), format="parquet")
        .to_table(columns=["episode_index", "task_index"])
        .to_pandas()
    )
    per_episode = task_rows.groupby("episode_index")["task_index"].agg(
        lambda values: sorted(set(map(int, values)))
    )
    if any(len(values) != 1 for values in per_episode):
        raise ValueError("source episode mixes task_index values")
    ids = pd.DataFrame(
        {
            "episode_index": per_episode.index.astype(int),
            "task_index": [values[0] for values in per_episode],
        }
    )
    joined = episodes[["episode_index", "tasks", "length"]].merge(
        ids, on="episode_index", validate="one_to_one"
    )
    joined["task"] = [
        _one_task(tasks, int(episode))
        for tasks, episode in zip(joined["tasks"], joined["episode_index"])
    ]
    pairs = joined[["task_index", "task"]].drop_duplicates()
    if pairs["task_index"].duplicated().any() or pairs["task"].duplicated().any():
        raise ValueError("dataset task_index/task text mapping is not bijective")
    mapping = {int(row.task_index): str(row.task) for row in pairs.itertuples()}
    expected_ids = set(range(len(mapping)))
    if set(mapping) != expected_ids:
        raise ValueError(f"task indices are not contiguous: {sorted(mapping)}")
    return mapping, joined, episodes


def _mirror(source: Path, temporary: Path) -> None:
    if temporary.exists():
        raise FileExistsError(temporary)
    for directory, _, filenames in os.walk(source):
        relative = Path(directory).relative_to(source)
        target_directory = temporary / relative
        target_directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_file = (Path(directory) / filename).resolve()
            target_file = target_directory / filename
            try:
                os.link(source_file, target_file)
            except OSError:
                shutil.copy2(source_file, target_file)


def _write_tasks(path: Path, task_strings: list[str]) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    frame = pd.DataFrame(
        {"task_index": np.arange(len(task_strings), dtype=np.int64)},
        index=pd.Index(task_strings, name="task"),
    )
    frame.to_parquet(path)


def _replace_column(table: pa.Table, name: str, values: list[Any]) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"missing expected column {name}")
    array = pa.array(values, type=table.schema.field(index).type)
    return table.set_column(index, table.schema.field(index), array)


def _rewrite_episode_metadata(
    source: Path,
    output: Path,
    annotations: dict[int, dict[str, Any]],
) -> None:
    for source_path in _episode_files(source):
        relative = source_path.relative_to(source)
        target_path = output / relative
        table = pq.read_table(source_path)
        episode_ids = list(map(int, table["episode_index"].to_pylist()))
        if not set(episode_ids) & set(annotations):
            continue
        tasks = table["tasks"].to_pylist()
        replacements: dict[str, list[Any]] = {}
        for name in (
            "stats/task_index/min",
            "stats/task_index/max",
            "stats/task_index/mean",
            "stats/task_index/std",
            "stats/task_index/count",
        ):
            if name in table.column_names:
                replacements[name] = table[name].to_pylist()
        for row_index, episode in enumerate(episode_ids):
            if episode not in annotations:
                continue
            annotation = annotations[episode]
            clauses = annotation["clauses"]
            task_ids = annotation["clause_task_indices"]
            first_count = annotation["switch_frame"]
            second_count = annotation["frame_count"] - first_count
            expanded = np.asarray(
                [task_ids[0]] * first_count + [task_ids[1]] * second_count,
                dtype=np.int64,
            )
            tasks[row_index] = clauses
            stats = {
                "stats/task_index/min": int(expanded.min()),
                "stats/task_index/max": int(expanded.max()),
                "stats/task_index/mean": float(expanded.mean()),
                "stats/task_index/std": float(expanded.std()),
                "stats/task_index/count": int(len(expanded)),
            }
            for name, value in stats.items():
                if name in replacements:
                    replacements[name][row_index] = [value]
        table = _replace_column(table, "tasks", tasks)
        for name, values in replacements.items():
            table = _replace_column(table, name, values)
        target_path.unlink()
        pq.write_table(table, target_path)


def _rewrite_global_task_stats(output: Path) -> None:
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    values = (
        pads.dataset(str(output / "data"), format="parquet")
        .to_table(columns=["task_index"])["task_index"]
        .to_numpy()
        .astype(np.int64)
    )
    record = stats["task_index"]
    record.update(
        {
            "min": [int(values.min())],
            "max": [int(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [int(len(values))],
        }
    )
    for key, quantile in (("q01", .01), ("q10", .10), ("q50", .50), ("q90", .90), ("q99", .99)):
        if key in record:
            record[key] = [float(np.quantile(values, quantile))]
    stats_path.unlink()
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"normalization stats require a nonempty matrix, got {array.shape}")
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(len(array))],
        **{
            key: np.quantile(array, quantile, axis=0).tolist()
            for key, quantile in (
                ("q01", 0.01),
                ("q10", 0.10),
                ("q50", 0.50),
                ("q90", 0.90),
                ("q99", 0.99),
            )
        },
    }


def _write_selected_normalization_stats(
    output: Path,
    selected_data: pd.DataFrame,
    task_indices: np.ndarray,
) -> dict[str, Any]:
    if len(selected_data) != len(task_indices):
        raise ValueError("selected task-index vector has the wrong row count")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = _numeric_stats(np.stack(selected_data["action"].to_numpy()))
    stats["observation.state"] = _numeric_stats(
        np.stack(selected_data["observation.state"].to_numpy())
    )
    stats["task_index"] = _numeric_stats(np.asarray(task_indices, dtype=np.int64))
    stats_path.unlink()
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return {
        "stats_sha256": sha256_file(stats_path),
        "row_count": int(len(selected_data)),
        "features_recomputed": ["action", "observation.state", "task_index"],
        "image_stats_role": "retained_but_unused_visual_normalization_is_identity",
    }


def _ipc_update(digest: Any, field: pa.Field, column: pa.ChunkedArray) -> None:
    sink = pa.BufferOutputStream()
    schema = pa.schema([field])
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(pa.Table.from_arrays([column], schema=schema))
    digest.update(sink.getvalue().to_pybytes())


def _selected_table(path: Path, episodes: set[int]) -> pa.Table:
    table = pq.read_table(path)
    mask = np.fromiter(
        (int(value) in episodes for value in table["episode_index"].to_pylist()),
        dtype=bool,
        count=len(table),
    )
    return table.filter(pa.array(mask))


def verify_physical_columns(
    source: Path,
    clause: Path,
    episodes: set[int],
    rewritten_relatives: list[str],
) -> dict[str, Any]:
    source_schema = pads.dataset(str(source / "data"), format="parquet").schema
    columns = [name for name in source_schema.names if name != "task_index"]
    source_hashes = {name: hashlib.sha256() for name in columns}
    output_hashes = {name: hashlib.sha256() for name in columns}
    selected_rows = 0
    for relative in sorted(rewritten_relatives):
        left = _selected_table(source / relative, episodes)
        right = _selected_table(clause / relative, episodes)
        if len(left) != len(right):
            raise AssertionError(f"row count changed in {relative}")
        selected_rows += len(left)
        for name in columns:
            if not left[name].equals(right[name]):
                raise AssertionError(f"non-language column changed: {relative}:{name}")
            _ipc_update(source_hashes[name], left.schema.field(name), left[name])
            _ipc_update(output_hashes[name], right.schema.field(name), right[name])
    hashes = {
        name: {
            "source_sha256": source_hashes[name].hexdigest(),
            "output_sha256": output_hashes[name].hexdigest(),
            "byte_identical": source_hashes[name].digest() == output_hashes[name].digest(),
        }
        for name in columns
    }
    if not all(record["byte_identical"] for record in hashes.values()):
        raise AssertionError("physical-column IPC hashes differ")
    return {"selected_rows": selected_rows, "columns": hashes}


def _rewrite_data(
    source: Path,
    clause: Path,
    annotations: dict[int, dict[str, Any]],
) -> list[str]:
    episode_set = set(annotations)
    rewritten = []
    seen: set[int] = set()
    for source_path in sorted(source.glob("data/**/*.parquet")):
        table = pq.read_table(source_path)
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        present = sorted(set(map(int, episode_values)) & episode_set)
        if not present:
            continue
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        for episode in present:
            annotation = annotations[episode]
            positions = np.flatnonzero(episode_values == episode)
            local_frames = frame_values[positions]
            if not np.array_equal(local_frames, np.arange(annotation["frame_count"])):
                raise ValueError(f"episode {episode} frame_index is not contiguous from zero")
            switch = annotation["switch_frame"]
            first_id, second_id = annotation["clause_task_indices"]
            task_values[positions] = np.where(local_frames < switch, first_id, second_id)
            local_tasks = task_values[positions]
            changes = np.flatnonzero(local_tasks[1:] != local_tasks[:-1]) + 1
            if changes.tolist() != [switch]:
                raise AssertionError(
                    f"episode {episode} switches={changes.tolist()}, expected [{switch}]"
                )
            seen.add(episode)
        index = table.schema.get_field_index("task_index")
        table = table.set_column(
            index,
            table.schema.field(index),
            pa.array(task_values, type=table.schema.field(index).type),
        )
        relative = source_path.relative_to(source)
        target = clause / relative
        target.unlink()
        pq.write_table(table, target)
        rewritten.append(str(relative))
    if seen != episode_set:
        raise ValueError(f"selected episodes absent from data: {sorted(episode_set - seen)}")
    return rewritten


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.root).expanduser().resolve()
    original_out = Path(args.original_out).expanduser().resolve()
    clause_out = Path(args.clause_out).expanduser().resolve()
    for output in (original_out, clause_out):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        if output.parent.stat().st_dev != source.stat().st_dev:
            raise ValueError("source and outputs must share a filesystem for hardlinks")
    original_tmp = original_out.with_name(original_out.name + f".building-{os.getpid()}")
    clause_tmp = clause_out.with_name(clause_out.name + f".building-{os.getpid()}")
    if original_tmp.exists() or clause_tmp.exists():
        raise FileExistsError("staging path already exists")

    targets = load_targets(getattr(args, "target_spec", None))
    target_spec_path = (
        Path(args.target_spec).expanduser().resolve()
        if getattr(args, "target_spec", None)
        else None
    )
    mapping, joined, _ = resolve_task_mapping(source)
    reverse = {task: task_index for task_index, task in mapping.items()}
    missing = [
        spec["evaluator_task"]
        for spec in targets.values()
        if spec["evaluator_task"] not in reverse
    ]
    if missing:
        raise ValueError(f"evaluator tasks absent from source mapping: {missing}")
    task_strings = [mapping[index] for index in range(len(mapping))]
    clause_ids: dict[tuple[int, int], int] = {}
    for suite_task_id in sorted(targets):
        for clause_index, clause in enumerate(targets[suite_task_id]["clauses"]):
            if clause in task_strings:
                raise ValueError(f"clause collides with an existing task string: {clause!r}")
            clause_ids[(suite_task_id, clause_index)] = len(task_strings)
            task_strings.append(clause)

    data = pads.dataset(str(source / "data"), format="parquet").to_table(
        columns=[
            "episode_index",
            "frame_index",
            "task_index",
            "action",
            "observation.state",
        ]
    ).to_pandas()
    annotations: dict[int, dict[str, Any]] = {}
    mapping_rows = []
    for suite_task_id, spec in sorted(targets.items()):
        dataset_task_id = reverse[spec["evaluator_task"]]
        selected_meta = joined[joined["task"] == spec["evaluator_task"]]
        selected_ids = set(map(int, selected_meta["episode_index"]))
        selected_data = data[data["episode_index"].isin(selected_ids)]
        if set(map(int, selected_data["task_index"].unique())) != {dataset_task_id}:
            raise ValueError(f"task identity mismatch for suite task {suite_task_id}")
        for episode, frame in selected_data.groupby("episode_index"):
            frame = frame.sort_values("frame_index")
            local = frame["frame_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(local, np.arange(len(frame))):
                raise ValueError(f"episode {episode} frame coverage is not contiguous")
            actions = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)
            event = first_valid_release(actions[:, 6], args.min_closed_run)
            switch = int(event["switch_frame"])
            if switch < args.min_clause_frames or len(frame) - switch < args.min_clause_frames:
                raise ValueError(
                    f"episode {episode} has non-meaningful switch {switch}/{len(frame)}"
                )
            annotation = {
                "episode_index": int(episode),
                "suite_task_id": suite_task_id,
                "dataset_task_index": dataset_task_id,
                "evaluator_task": spec["evaluator_task"],
                "clauses": list(spec["clauses"]),
                "clause_task_indices": [
                    clause_ids[(suite_task_id, 0)], clause_ids[(suite_task_id, 1)]
                ],
                "frame_count": int(len(frame)),
                "release_action_index": event["release_action_index"],
                "closed_run_start": event["closed_run_start"],
                "closed_run_actions": event["closed_run_actions"],
                "release_candidates": event["release_candidates"],
                "switch_frame": switch,
                "boundary_version": BOUNDARY_VERSION,
                "boundary_convention": "clause1_[0,switch);_clause2_[switch,T)",
                "release_action_membership": "clause1",
            }
            annotations[int(episode)] = annotation
        mapping_rows.append(
            {
                "suite_task_id": suite_task_id,
                "evaluator_task": spec["evaluator_task"],
                "dataset_task_index": dataset_task_id,
                "episodes": len(selected_ids),
                "frames": int(len(selected_data)),
                "clauses": spec["clauses"],
            }
        )
    if len(annotations) != sum(row["episodes"] for row in mapping_rows):
        raise AssertionError("selected task episode sets overlap")

    mapping_payload = {
        "schema_version": SCHEMA_VERSION,
        "binding_key": "exact_evaluator_task_description",
        "targets": mapping_rows,
    }
    mapping_hash = sha256_bytes(canonical_json(mapping_payload))

    _mirror(source, original_tmp)
    _mirror(source, clause_tmp)
    _write_tasks(
        original_tmp / "meta" / "tasks.parquet",
        [mapping[index] for index in range(len(mapping))],
    )
    _write_tasks(clause_tmp / "meta" / "tasks.parquet", task_strings)
    info_path = clause_tmp / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(task_strings)
    info_path.unlink()
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    rewritten = _rewrite_data(source, clause_tmp, annotations)
    _rewrite_episode_metadata(source, clause_tmp, annotations)
    _rewrite_global_task_stats(clause_tmp)

    selected_data = data[data["episode_index"].isin(set(annotations))].sort_values(
        ["episode_index", "frame_index"]
    )
    original_task_indices = selected_data["task_index"].to_numpy(dtype=np.int64)
    clause_task_indices = np.empty(len(selected_data), dtype=np.int64)
    cursor = 0
    for episode, frame in selected_data.groupby("episode_index", sort=True):
        annotation = annotations[int(episode)]
        frame = frame.sort_values("frame_index")
        local_frames = frame["frame_index"].to_numpy(dtype=np.int64)
        count = len(frame)
        clause_task_indices[cursor : cursor + count] = np.where(
            local_frames < annotation["switch_frame"],
            annotation["clause_task_indices"][0],
            annotation["clause_task_indices"][1],
        )
        cursor += count
    if cursor != len(selected_data):
        raise AssertionError("normalization-stat task-index coverage mismatch")
    selected_stats = {
        "paired_original": _write_selected_normalization_stats(
            original_tmp, selected_data, original_task_indices
        ),
        "clause": _write_selected_normalization_stats(
            clause_tmp, selected_data, clause_task_indices
        ),
    }

    labels_path = clause_tmp / "meta" / "two_subgoal_labels.jsonl"
    with labels_path.open("w") as stream:
        for episode, annotation in sorted(annotations.items()):
            for clause_index, (start, end) in enumerate(
                ((0, annotation["switch_frame"]),
                 (annotation["switch_frame"], annotation["frame_count"]))
            ):
                stream.write(
                    json.dumps(
                        {
                            **annotation,
                            "clause_index": clause_index,
                            "seg_start": start,
                            "seg_end": end,
                            "label": annotation["clauses"][clause_index],
                            "task_index": annotation["clause_task_indices"][clause_index],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    verification = verify_physical_columns(
        source, clause_tmp, set(annotations), rewritten
    )
    expected_frames = sum(annotation["frame_count"] for annotation in annotations.values())
    if verification["selected_rows"] != expected_frames:
        raise AssertionError(
            f"verified {verification['selected_rows']} rows, expected {expected_frames}"
        )
    switch_by_task: dict[int, list[int]] = defaultdict(list)
    for annotation in annotations.values():
        switch_by_task[annotation["suite_task_id"]].append(annotation["switch_frame"])
    switch_summary = {
        str(task): {
            "count": len(values),
            "min": int(np.min(values)),
            "median": float(np.median(values)),
            "max": int(np.max(values)),
            "quantiles_10_25_50_75_90": [
                float(value) for value in np.quantile(values, [.1, .25, .5, .75, .9])
            ],
        }
        for task, values in sorted(switch_by_task.items())
    }
    selected_source_hashes = {
        relative: sha256_file(source / relative) for relative in sorted(rewritten)
    }
    script_path = Path(__file__).resolve()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source),
        "paired_original_root": str(original_out),
        "clause_root": str(clause_out),
        "source_info_sha256": sha256_file(source / "meta" / "info.json"),
        "source_episode_metadata_sha256": {
            str(path.relative_to(source)): sha256_file(path)
            for path in _episode_files(source)
        },
        "selected_source_data_sha256": selected_source_hashes,
        "builder_sha256": sha256_file(script_path),
        "target_spec": (
            {
                "path": str(target_spec_path),
                "sha256": sha256_file(target_spec_path),
            }
            if target_spec_path is not None
            else {"path": None, "sha256": None, "name": "builtin_t0_t4"}
        ),
        "mapping": mapping_payload,
        "mapping_sha256": mapping_hash,
        "boundary": {
            "version": BOUNDARY_VERSION,
            "min_closed_run_actions": args.min_closed_run,
            "min_clause_frames": args.min_clause_frames,
            "closed_definition": "action[6] > 0",
            "release_definition": "action[t-1,6] > 0 and action[t,6] <= 0",
            "switch_frame": "release_action_index + 1",
            "release_action_membership": "clause1",
        },
        "episode_count": len(annotations),
        "frame_count": expected_frames,
        "switches_per_episode": 1,
        "full_frame_coverage": True,
        "switch_summary": switch_summary,
        "labels_sha256": sha256_file(labels_path),
        "paired_original_tasks_sha256": sha256_file(
            original_tmp / "meta" / "tasks.parquet"
        ),
        "clause_tasks_sha256": sha256_file(clause_tmp / "meta" / "tasks.parquet"),
        "clause_info_sha256": sha256_file(clause_tmp / "meta" / "info.json"),
        "physical_non_language_verification": verification,
        "selected_normalization_stats": selected_stats,
        "rewritten_data_files": rewritten,
        "annotations": [annotations[index] for index in sorted(annotations)],
    }
    original_provenance = {
        "schema_version": SCHEMA_VERSION,
        "role": "paired_original_control",
        "source_root": str(source),
        "output_root": str(original_out),
        "data_files": "hardlinked_byte_identical_to_source",
        "tasks_sha256": sha256_file(original_tmp / "meta" / "tasks.parquet"),
        "mapping_sha256": mapping_hash,
        "episode_indices": sorted(annotations),
        "episode_count": len(annotations),
        "frame_count": expected_frames,
        "builder_sha256": sha256_file(script_path),
    }
    (clause_tmp / "meta" / "two_subgoal_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    (original_tmp / "meta" / "paired_original_provenance.json").write_text(
        json.dumps(original_provenance, indent=2, sort_keys=True) + "\n"
    )
    (clause_tmp / "meta" / "evaluator_dataset_task_mapping.json").write_text(
        json.dumps(
            {**mapping_payload, "mapping_sha256": mapping_hash},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # All validation has completed before either immutable destination appears.
    os.rename(clause_tmp, clause_out)
    os.rename(original_tmp, original_out)
    print(json.dumps({key: provenance[key] for key in (
        "episode_count", "frame_count", "mapping_sha256", "switch_summary",
        "labels_sha256", "builder_sha256")}, indent=2, sort_keys=True))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--original_out", required=True)
    parser.add_argument("--clause_out", required=True)
    parser.add_argument(
        "--target_spec",
        help="optional JSON task/clauses specification; defaults to the locked t0/t4 map",
    )
    parser.add_argument("--min_closed_run", type=int, default=DEFAULT_MIN_CLOSED_RUN)
    parser.add_argument("--min_clause_frames", type=int, default=8)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
