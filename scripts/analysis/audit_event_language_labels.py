"""Quality gate for constrained Molmo2 event-phase labels.

The audit is deliberately label-only: it checks provenance, task coverage,
candidate/label identity, and temporal coherence without changing a label.  A
high adjacent phase-inversion rate is evidence that independent visual phase
classification is too noisy to train on and stops the dependent dataset build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.data.molmo_segment_labels import BOUNDARY_VERSION, PHASE_BANKS, PROMPT_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty label set")
    by_episode = defaultdict(list)
    phase_counts = Counter()
    task_counts = Counter()
    for index, row in enumerate(rows, start=1):
        if row.get("boundary_version") != BOUNDARY_VERSION:
            raise ValueError(f"row {index} has wrong boundary version")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"row {index} has wrong prompt version")
        if row.get("label_mode") != "phase_bank":
            raise ValueError(f"row {index} is not phase_bank mode")
        task = str(row["task"])
        if task not in PHASE_BANKS:
            raise ValueError(f"row {index} has unreviewed task {task!r}")
        phase_text = str(row.get("phase_id", ""))
        if len(phase_text) != 3 or not phase_text.startswith("P") or not phase_text[1:].isdigit():
            raise ValueError(f"row {index} has invalid phase_id {phase_text!r}")
        phase = int(phase_text[1:])
        if phase >= len(PHASE_BANKS[task]) or row["label"] != PHASE_BANKS[task][phase]:
            raise ValueError(f"row {index} phase/label identity mismatch")
        episode = int(row["episode_index"])
        by_episode[episode].append((int(row["seg_start"]), int(row["seg_end"]), phase, task))
        phase_counts[(task, phase)] += 1
        task_counts[task] += 1

    adjacent = inversions = gross_inversions = 0
    episode_rows = []
    for episode, records in sorted(by_episode.items()):
        records.sort()
        phases = [record[2] for record in records]
        task_set = {record[3] for record in records}
        if len(task_set) != 1:
            raise ValueError(f"episode {episode} mixes tasks")
        local_adjacent = max(0, len(phases) - 1)
        local_inversions = sum(right < left for left, right in zip(phases, phases[1:]))
        local_gross = sum(right < left - 1 for left, right in zip(phases, phases[1:]))
        adjacent += local_adjacent
        inversions += local_inversions
        gross_inversions += local_gross
        episode_rows.append(
            {
                "episode_index": episode,
                "task": next(iter(task_set)),
                "segments": len(records),
                "phase_sequence": phases,
                "adjacent_inversions": local_inversions,
                "gross_adjacent_inversions": local_gross,
            }
        )

    return {
        "records": len(rows),
        "episodes": len(by_episode),
        "tasks": {
            task: {
                "records": task_counts[task],
                "episodes": len({episode for episode, records in by_episode.items()
                                 if records[0][3] == task}),
                "phase_histogram": {
                    f"P{phase:02d}": phase_counts[(task, phase)]
                    for phase in range(len(PHASE_BANKS[task]))
                },
            }
            for task in sorted(task_counts)
        },
        "adjacent_pairs": adjacent,
        "adjacent_inversions": inversions,
        "gross_adjacent_inversions": gross_inversions,
        "adjacent_inversion_fraction": inversions / adjacent if adjacent else 0.0,
        "gross_adjacent_inversion_fraction": gross_inversions / adjacent if adjacent else 0.0,
        "episodes_with_any_inversion": sum(
            row["adjacent_inversions"] > 0 for row in episode_rows
        ),
        "episode_details": episode_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_inversion_fraction", type=float, default=0.15)
    parser.add_argument("--max_gross_inversion_fraction", type=float, default=0.05)
    parser.add_argument("--min_episodes_per_task", type=int, default=20)
    args = parser.parse_args()
    labels = Path(args.labels).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    report = audit(rows)
    report.update(
        {
            "labels_path": str(labels),
            "labels_sha256": sha256_file(labels),
            "auditor_sha256": sha256_file(Path(__file__).resolve()),
            "thresholds": {
                "max_inversion_fraction": args.max_inversion_fraction,
                "max_gross_inversion_fraction": args.max_gross_inversion_fraction,
                "min_episodes_per_task": args.min_episodes_per_task,
            },
        }
    )
    failures = []
    if report["adjacent_inversion_fraction"] > args.max_inversion_fraction:
        failures.append("adjacent inversion fraction")
    if report["gross_adjacent_inversion_fraction"] > args.max_gross_inversion_fraction:
        failures.append("gross adjacent inversion fraction")
    for task, task_report in report["tasks"].items():
        if task_report["episodes"] < args.min_episodes_per_task:
            failures.append(f"task {task!r} episode coverage")
    report["quality_gate_passed"] = not failures
    report["quality_gate_failures"] = failures
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "records", "episodes", "adjacent_inversion_fraction",
        "gross_adjacent_inversion_fraction", "quality_gate_passed",
        "quality_gate_failures")}, indent=2))
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
