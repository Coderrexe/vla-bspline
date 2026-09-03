"""Audit the physical-data and language intervention in RoboCasa easy4.

The task-level and granular curriculum arms are only a meaningful language
comparison if they select exactly the same physical transitions.  This script
checks that contract over every selected frame and reports changed-language
coverage per source task.  It is read-only and writes one immutable JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


TASK_RANGES = {
    "KettleBoiling": (1010, 1510),
    "RinseSinkBasin": (3528, 4036),
    "ScrubCuttingBoard": (4037, 4540),
    "StackBowlsCabinet": (5546, 6060),
}
PHYSICAL_COLUMNS = [
    "observation.state",
    "action",
    "next.reward",
    "next.done",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
]
LANGUAGE_COLUMNS = [
    "annotation.human.task_description",
    "annotation.human.task_name",
    "task_index",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_filter():
    episode = ds.field("episode_index")
    filters = [
        (episode >= start) & (episode <= end)
        for start, end in TASK_RANGES.values()
    ]
    combined = filters[0]
    for expression in filters[1:]:
        combined = combined | expression
    return combined


def _task_map(root: Path) -> dict[int, str]:
    rows = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    return {int(row["task_index"]): str(row["task"]) for row in rows}


def _source_task(episode_index: int) -> str:
    for task, (start, end) in TASK_RANGES.items():
        if start <= episode_index <= end:
            return task
    raise ValueError(episode_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--granular", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    granular = Path(args.granular).resolve()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    columns = PHYSICAL_COLUMNS + LANGUAGE_COLUMNS
    selection = _selection_filter()
    base_table = ds.dataset(base / "data", format="parquet").to_table(
        columns=columns, filter=selection
    )
    granular_table = ds.dataset(granular / "data", format="parquet").to_table(
        columns=columns, filter=selection
    )
    base_table = base_table.sort_by("index")
    granular_table = granular_table.sort_by("index")
    if base_table.num_rows != granular_table.num_rows:
        raise RuntimeError(
            f"selected row-count mismatch: {base_table.num_rows} != "
            f"{granular_table.num_rows}"
        )

    physical_equal = base_table.select(PHYSICAL_COLUMNS).equals(
        granular_table.select(PHYSICAL_COLUMNS)
    )
    if not physical_equal:
        raise RuntimeError("task and granular arms differ in physical transitions")

    base_tasks = base_table["task_index"].combine_chunks()
    granular_tasks = granular_table["task_index"].combine_chunks()
    changed = pc.not_equal(base_tasks, granular_tasks)
    base_task_map = _task_map(base)
    granular_task_map = _task_map(granular)

    by_task = {
        task: {"frames": 0, "changed_frames": 0, "changed_fraction": 0.0}
        for task in TASK_RANGES
    }
    caption_pairs: dict[str, int] = {}
    episodes = base_table["episode_index"].combine_chunks().to_pylist()
    old_indices = base_tasks.to_pylist()
    new_indices = granular_tasks.to_pylist()
    changed_values = changed.to_pylist()
    for episode, old_index, new_index, is_changed in zip(
        episodes, old_indices, new_indices, changed_values, strict=True
    ):
        source_task = _source_task(int(episode))
        by_task[source_task]["frames"] += 1
        if is_changed:
            by_task[source_task]["changed_frames"] += 1
            pair = json.dumps(
                [base_task_map[int(old_index)], granular_task_map[int(new_index)]],
                ensure_ascii=False,
            )
            caption_pairs[pair] = caption_pairs.get(pair, 0) + 1

    for counts in by_task.values():
        counts["changed_fraction"] = counts["changed_frames"] / counts["frames"]

    result = {
        "schema": "robocasa_easy4_language_audit_v1",
        "base": str(base),
        "granular": str(granular),
        "selected_episode_ranges": TASK_RANGES,
        "selected_rows": base_table.num_rows,
        "physical_columns": PHYSICAL_COLUMNS,
        "physical_transitions_exactly_equal": physical_equal,
        "by_source_task": by_task,
        "changed_frames": int(pc.sum(pc.cast(changed, "int64")).as_py()),
        "caption_pair_counts": dict(
            sorted(caption_pairs.items(), key=lambda item: (-item[1], item[0]))
        ),
        "dataset_fingerprints": {
            "base_info": _sha256(base / "meta/info.json"),
            "base_tasks": _sha256(base / "meta/tasks.parquet"),
            "granular_info": _sha256(granular / "meta/info.json"),
            "granular_tasks": _sha256(granular / "meta/tasks.parquet"),
        },
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": _sha256(Path(__file__)),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = stream.name
    os.link(temporary, output)
    os.unlink(temporary)
    print(json.dumps({key: result[key] for key in (
        "selected_rows", "physical_transitions_exactly_equal", "changed_frames",
        "by_source_task",
    )}, indent=2))


if __name__ == "__main__":
    main()
