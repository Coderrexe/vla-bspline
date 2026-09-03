#!/usr/bin/env python3
"""Make Kettle phase language carry one burner goal across place and turn-on.

The official RoboCasa annotations name the selected burner only in the final
phase.  This transform copies that episode-level burner identity into the
placement instruction while leaving every boundary and physical datum intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PLACE = "place the kettle on the stove burner"
BURNER_RE = re.compile(
    r"^turn on the (center|front-center|front-left|front-right|left|rear-left|rear-right) "
    r"burner where the kettle is placed$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(table: pa.Table) -> tuple[pa.Table, dict[int, str], int]:
    required = {
        "episode_index", "frame_index", "instruction", "atomic_skill", "stage", "subtask_idx"
    }
    if set(table.column_names) != required:
        raise RuntimeError(f"unexpected schedule columns: {table.column_names}")

    rows = table.to_pylist()
    burners: dict[int, set[str]] = {}
    for row in rows:
        episode = int(row["episode_index"])
        match = BURNER_RE.fullmatch(str(row["instruction"]))
        if match:
            burners.setdefault(episode, set()).add(match.group(1))

    episodes = {int(row["episode_index"]) for row in rows}
    if set(burners) != episodes:
        missing = sorted(episodes - set(burners))[:5]
        raise RuntimeError(f"episodes without a burner-specific terminal phase: {missing}")
    ambiguous = {episode: values for episode, values in burners.items() if len(values) != 1}
    if ambiguous:
        raise RuntimeError(f"episodes with ambiguous burner goals: {ambiguous}")
    goal = {episode: next(iter(values)) for episode, values in burners.items()}

    changed = 0
    for row in rows:
        if row["instruction"] == PLACE:
            location = goal[int(row["episode_index"])]
            row["instruction"] = f"place the kettle on the {location} stove burner"
            changed += 1
    if changed <= 0:
        raise RuntimeError("no placement rows were rewritten")
    return pa.Table.from_pylist(rows, schema=table.schema), goal, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_manifest.exists():
        raise FileExistsError("refusing to overwrite a goal-consistent schedule artifact")

    source_manifest = json.loads(args.schedule_manifest.read_text())
    if sha256(args.schedule) != source_manifest["schedule_sha256"]:
        raise RuntimeError("source schedule hash mismatch")
    source = pq.read_table(args.schedule)
    transformed, goals, changed = transform(source)
    if len(transformed) != len(source):
        raise RuntimeError("row count changed")
    for column in set(source.column_names) - {"instruction"}:
        if not source[column].equals(transformed[column]):
            raise RuntimeError(f"non-language schedule column changed: {column}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(transformed, args.output, compression="zstd")
    counts: dict[str, int] = {}
    for value in goals.values():
        counts[value] = counts.get(value, 0) + 1
    payload = dict(source_manifest)
    payload.update(
        {
            "schema": "robocasa_official_phase_schedule_v1",
            "schedule_sha256": sha256(args.output),
            "source_schedule": str(args.schedule.resolve()),
            "source_schedule_sha256": sha256(args.schedule),
            "source_schedule_manifest": str(args.schedule_manifest.resolve()),
            "source_schedule_manifest_sha256": sha256(args.schedule_manifest),
            "language_transform": "episode_burner_goal_copied_into_place_phase_v1",
            "changed_instruction_rows": changed,
            "episode_burner_goal_counts": dict(sorted(counts.items())),
            "physical_or_boundary_columns_changed": False,
        }
    )
    args.output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
