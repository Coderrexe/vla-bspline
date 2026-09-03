"""Audit legacy Molmo language boundaries against production event semantics.

This is a read-only, CPU-only audit over a LeRobot v3 dataset.  It reconstructs
the exact boundary heuristic formerly used by ``molmo_segment_labels.py`` and
compares that partition with the corrected shared event-target implementation.
No images are decoded and the only output is one immutable JSON report.

Example (the locally cached HuggingFaceVLA/libero snapshot)::

    python scripts/analysis/legacy_molmo_boundary_audit.py \
      --root ~/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/REV \
      --out /tmp/libero_molmo_boundary_audit.json

The defaults (horizon 24, minimum segment 8, pause fraction 0.15) are the
defaults used to create the historical Molmo artifacts.  The audit intentionally
holds those hyperparameters fixed so that it isolates implementation semantics:
``k+1`` versus ``k`` toggle cuts, sign versus raw-value toggles, episode-global
nonzero-speed median versus fixed-window quantile, and real-tail versus padded
episode-end handling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import torch


SCHEMA_VERSION = 1
LEGACY_BOUNDARY_VERSION = "molmo_legacy_sign_kplus1_global_nonzero_median"


class Segment(NamedTuple):
    """Half-open action interval and the condition that ended it."""

    start: int
    end: int
    event_type: str


def _load_molmo_module():
    module_path = Path(__file__).resolve().parents[1] / "data" / "molmo_segment_labels.py"
    spec = importlib.util.spec_from_file_location("boundary_audit_molmo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load corrected segmenter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOLMO = _load_molmo_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_segments(
    actions: np.ndarray,
    horizon_max: int,
    min_seg: int,
    pause_frac: float,
) -> list[Segment]:
    """Frozen copy of the pre-migration Molmo boundary heuristic.

    Do not simplify this function: its asymmetric pause look-ahead and toggle
    ``k + 1`` cut are historical behavior being measured, not desired behavior.
    """

    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, D), got {actions.shape}")
    if actions.shape[1] < 7:
        raise ValueError(f"expected at least 7 action dimensions, got {actions.shape[1]}")
    if not 1 <= min_seg <= horizon_max:
        raise ValueError(f"min_seg={min_seg} must lie in [1, {horizon_max}]")
    if pause_frac < 0:
        raise ValueError(f"pause_frac must be nonnegative, got {pause_frac}")
    if len(actions) == 0:
        return []

    grip = actions[:, 6]
    speed = np.linalg.norm(actions[:, :6], axis=1)
    nonzero = speed > 1e-8
    threshold = pause_frac * np.median(speed[nonzero]) if nonzero.any() else 0.0
    segments: list[Segment] = []
    start = 0
    length = len(actions)
    while start < length:
        window_end = min(start + horizon_max, length)
        cut = window_end
        event_type = "episode_end" if window_end == length else "cap"
        for index in range(start + min_seg, window_end):
            if index > 0 and np.sign(grip[index]) != np.sign(grip[index - 1]):
                cut = index + 1
                event_type = "toggle"
                break
            if (
                threshold > 0
                and index + 1 < length
                and speed[index] < threshold
                and speed[index + 1] < threshold
            ):
                cut = index + 1
                event_type = "pause"
                break
        segments.append(Segment(start, cut, event_type))
        start = cut
    return segments


def corrected_segments(
    actions: np.ndarray,
    horizon_max: int,
    min_seg: int,
    pause_frac: float,
) -> list[Segment]:
    """Use the corrected extractor, which delegates to shared event_targets."""

    return [
        Segment(int(segment.start), int(segment.end), str(segment.event_type))
        for segment in _MOLMO.language_segments(
            actions,
            horizon_max=horizon_max,
            min_seg=min_seg,
            pause_frac=pause_frac,
        )
    ]


def _internal_boundaries(segments: Sequence[Segment]) -> list[int]:
    return [segment.end for segment in segments[:-1]]


def _nearest_distances(source: Sequence[int], target: Sequence[int]) -> list[int]:
    if not source:
        return []
    if not target:
        # No finite counterpart exists.  Keep this separate from the numeric
        # distribution instead of inventing a distance tied to episode length.
        return []
    target_array = np.asarray(target, dtype=np.int64)
    return [int(np.min(np.abs(target_array - value))) for value in source]


def _overlap(left: Segment, right: Segment) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def compare_partitions(legacy: Sequence[Segment], corrected: Sequence[Segment]) -> dict:
    """Return sufficient per-episode statistics for exact aggregation."""

    legacy_boundaries = _internal_boundaries(legacy)
    corrected_boundaries = _internal_boundaries(corrected)
    legacy_boundary_set = set(legacy_boundaries)
    corrected_boundary_set = set(corrected_boundaries)
    legacy_intervals = {(segment.start, segment.end) for segment in legacy}
    corrected_intervals = {(segment.start, segment.end) for segment in corrected}

    exact_intervals = legacy_intervals & corrected_intervals
    legacy_crossing = 0
    legacy_outside_dominant = 0
    legacy_zero_overlap = 0
    for segment in legacy:
        overlaps = [_overlap(segment, candidate) for candidate in corrected]
        best = max(overlaps, default=0)
        legacy_outside_dominant += (segment.end - segment.start) - best
        legacy_zero_overlap += int(best == 0)
        legacy_crossing += int(sum(value > 0 for value in overlaps) > 1)

    corrected_crossing = 0
    for segment in corrected:
        corrected_crossing += int(
            sum(_overlap(segment, candidate) > 0 for candidate in legacy) > 1
        )

    frame_count = legacy[-1].end if legacy else 0
    if corrected and corrected[-1].end != frame_count:
        raise ValueError("legacy and corrected partitions do not cover the same frame count")
    return {
        "frames": frame_count,
        "legacy_segments": len(legacy),
        "corrected_segments": len(corrected),
        "segment_count_delta": len(corrected) - len(legacy),
        "exact_partition": [
            (segment.start, segment.end) for segment in legacy
        ]
        == [(segment.start, segment.end) for segment in corrected],
        "exact_boundary_set": legacy_boundary_set == corrected_boundary_set,
        "legacy_boundaries": len(legacy_boundaries),
        "corrected_boundaries": len(corrected_boundaries),
        "exact_boundaries": len(legacy_boundary_set & corrected_boundary_set),
        "legacy_only_boundaries": len(legacy_boundary_set - corrected_boundary_set),
        "corrected_only_boundaries": len(corrected_boundary_set - legacy_boundary_set),
        "legacy_to_corrected_nearest": _nearest_distances(
            legacy_boundaries, corrected_boundaries
        ),
        "corrected_to_legacy_nearest": _nearest_distances(
            corrected_boundaries, legacy_boundaries
        ),
        "legacy_boundaries_without_counterpart": int(
            bool(legacy_boundaries) and not corrected_boundaries
        )
        * len(legacy_boundaries),
        "corrected_boundaries_without_counterpart": int(
            bool(corrected_boundaries) and not legacy_boundaries
        )
        * len(corrected_boundaries),
        # A historical caption is safely reusable only when its entire source
        # interval exists unchanged in the corrected partition.
        "exact_reusable_legacy_intervals": len(exact_intervals),
        "legacy_intervals_requiring_recaption": len(legacy) - len(exact_intervals),
        "legacy_intervals_crossing_corrected_boundary": legacy_crossing,
        "corrected_intervals_crossing_legacy_boundary": corrected_crossing,
        "legacy_intervals_with_zero_overlap": legacy_zero_overlap,
        # Assign each legacy label to its maximum-overlap corrected interval.
        # Residual frames quantify unavoidable interval-membership ambiguity.
        "legacy_label_frames_outside_dominant_corrected_interval": (
            legacy_outside_dominant
        ),
        "legacy_event_types": dict(Counter(segment.event_type for segment in legacy)),
        "corrected_event_types": dict(
            Counter(segment.event_type for segment in corrected)
        ),
    }


def _distance_summary(values: Sequence[int], missing: int) -> dict:
    array = np.asarray(values, dtype=np.int64)
    result = {
        "finite_count": int(array.size),
        "missing_counterpart_count": int(missing),
    }
    if not array.size:
        result.update(
            {
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "max": None,
                "exact_fraction": None,
                "within_1_fraction": None,
                "within_2_fraction": None,
                "within_4_fraction": None,
                "within_8_fraction": None,
                "histogram": {},
            }
        )
        return result
    result.update(
        {
            "mean": float(array.mean()),
            "median": float(np.quantile(array, 0.5)),
            "p90": float(np.quantile(array, 0.9)),
            "p95": float(np.quantile(array, 0.95)),
            "max": int(array.max()),
            "exact_fraction": float(np.mean(array == 0)),
            "within_1_fraction": float(np.mean(array <= 1)),
            "within_2_fraction": float(np.mean(array <= 2)),
            "within_4_fraction": float(np.mean(array <= 4)),
            "within_8_fraction": float(np.mean(array <= 8)),
            "histogram": {
                "0": int(np.sum(array == 0)),
                "1": int(np.sum(array == 1)),
                "2": int(np.sum(array == 2)),
                "3-4": int(np.sum((array >= 3) & (array <= 4))),
                "5-8": int(np.sum((array >= 5) & (array <= 8))),
                "9-16": int(np.sum((array >= 9) & (array <= 16))),
                "17+": int(np.sum(array >= 17)),
            },
        }
    )
    return result


def aggregate_episode_metrics(rows: Sequence[dict]) -> dict:
    additive = [
        "frames",
        "legacy_segments",
        "corrected_segments",
        "segment_count_delta",
        "legacy_boundaries",
        "corrected_boundaries",
        "exact_boundaries",
        "legacy_only_boundaries",
        "corrected_only_boundaries",
        "legacy_boundaries_without_counterpart",
        "corrected_boundaries_without_counterpart",
        "exact_reusable_legacy_intervals",
        "legacy_intervals_requiring_recaption",
        "legacy_intervals_crossing_corrected_boundary",
        "corrected_intervals_crossing_legacy_boundary",
        "legacy_intervals_with_zero_overlap",
        "legacy_label_frames_outside_dominant_corrected_interval",
    ]
    result = {key: int(sum(row[key] for row in rows)) for key in additive}
    result.update(
        {
            "episodes": len(rows),
            "episodes_with_exact_partition": sum(row["exact_partition"] for row in rows),
            "episodes_with_exact_boundary_set": sum(
                row["exact_boundary_set"] for row in rows
            ),
        }
    )
    denominator = result["legacy_boundaries"]
    result["legacy_boundary_exact_precision"] = (
        result["exact_boundaries"] / denominator if denominator else None
    )
    denominator = result["corrected_boundaries"]
    result["corrected_boundary_exact_recall"] = (
        result["exact_boundaries"] / denominator if denominator else None
    )
    denominator = result["legacy_segments"]
    result["exact_reusable_legacy_interval_fraction"] = (
        result["exact_reusable_legacy_intervals"] / denominator
        if denominator
        else None
    )
    result["legacy_interval_recaption_fraction"] = (
        result["legacy_intervals_requiring_recaption"] / denominator
        if denominator
        else None
    )
    denominator = result["frames"]
    result["legacy_label_frame_ambiguity_fraction"] = (
        result["legacy_label_frames_outside_dominant_corrected_interval"] / denominator
        if denominator
        else None
    )
    result["episode_exact_partition_fraction"] = (
        result["episodes_with_exact_partition"] / result["episodes"]
        if result["episodes"]
        else None
    )
    result["legacy_to_corrected_nearest_boundary_distance"] = _distance_summary(
        [value for row in rows for value in row["legacy_to_corrected_nearest"]],
        result["legacy_boundaries_without_counterpart"],
    )
    result["corrected_to_legacy_nearest_boundary_distance"] = _distance_summary(
        [value for row in rows for value in row["corrected_to_legacy_nearest"]],
        result["corrected_boundaries_without_counterpart"],
    )
    for name in ("legacy_event_types", "corrected_event_types"):
        counter: Counter[str] = Counter()
        for row in rows:
            counter.update(row[name])
        result[name] = dict(sorted(counter.items()))
    return result


def _task_strings(tasks_path: Path) -> list[str]:
    table = pd.read_parquet(tasks_path)
    if table.index.dtype == object:
        return [str(value) for value in table.index]
    return [str(value) for value in table.iloc[:, 0]]


def _canonical_action_digest(
    episode_rows: Sequence[tuple[int, int, int, np.ndarray]], digest: "hashlib._Hash"
) -> None:
    """Hash semantic values, independent of Parquet row-group encoding."""

    for episode, frame, task, actions in episode_rows:
        digest.update(np.asarray([episode, frame, task], dtype="<i8").tobytes())
        digest.update(np.asarray(actions, dtype="<f4").tobytes())


def _git_revision(repo: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def audit_dataset(
    root: Path,
    *,
    horizon_max: int,
    min_seg: int,
    pause_frac: float,
    task_ids: set[int] | None = None,
    episode_limit: int | None = None,
) -> dict:
    root = root.expanduser().resolve()
    tasks_path = root / "meta" / "tasks.parquet"
    info_path = root / "meta" / "info.json"
    episode_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not tasks_path.is_file() or not info_path.is_file() or not episode_paths or not data_paths:
        raise FileNotFoundError(
            f"{root} is missing meta/tasks.parquet, meta/info.json, episode metadata, or data parquets"
        )

    task_strings = _task_strings(tasks_path)
    episodes_meta = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "length", "tasks"]) for path in episode_paths],
        ignore_index=True,
    )
    expected_lengths = {
        int(row.episode_index): int(row.length) for row in episodes_meta.itertuples()
    }

    by_episode: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for path in data_paths:
        table = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "task_index", "action"]
        )
        if task_ids is not None:
            table = table[table["task_index"].isin(task_ids)]
        for row in table.itertuples(index=False):
            by_episode[int(row.episode_index)].append(
                (int(row.frame_index), int(row.task_index), np.asarray(row.action))
            )

    selected_episode_ids = sorted(by_episode)
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive")
        selected_episode_ids = selected_episode_ids[:episode_limit]

    semantic_digest = hashlib.sha256()
    episode_results: list[dict] = []
    per_task_rows: dict[int, list[dict]] = defaultdict(list)
    for episode in selected_episode_ids:
        rows = sorted(by_episode[episode], key=lambda item: item[0])
        frames = [item[0] for item in rows]
        if frames != list(range(len(rows))):
            raise ValueError(f"episode {episode} has non-contiguous or duplicate frame indices")
        if expected_lengths.get(episode) != len(rows):
            raise ValueError(
                f"episode {episode} has {len(rows)} data rows but metadata length {expected_lengths.get(episode)}"
            )
        task_values = {item[1] for item in rows}
        if len(task_values) != 1:
            raise ValueError(f"source episode {episode} has multiple task indices: {task_values}")
        task_id = next(iter(task_values))
        if not 0 <= task_id < len(task_strings):
            raise ValueError(f"episode {episode} has invalid task index {task_id}")
        actions = np.stack([item[2] for item in rows]).astype(np.float32, copy=False)
        _canonical_action_digest(
            [
                (episode, frame, task, action)
                for frame, task, action in rows
            ],
            semantic_digest,
        )

        legacy = legacy_segments(actions, horizon_max, min_seg, pause_frac)
        corrected = corrected_segments(actions, horizon_max, min_seg, pause_frac)
        metrics = compare_partitions(legacy, corrected)
        metrics.update(
            {
                "episode_index": episode,
                "task_id": task_id,
                "task": task_strings[task_id],
            }
        )
        episode_results.append(metrics)
        per_task_rows[task_id].append(metrics)

    repo = Path(__file__).resolve().parents[2]
    molmo_source = Path(_MOLMO.__file__).resolve()
    event_source = repo / "policy" / "smolvla_spline" / "event_targets.py"
    metadata_files = [info_path, tasks_path, *episode_paths]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "legacy Molmo label partition versus corrected production event partition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "horizon_max": horizon_max,
            "min_seg": min_seg,
            "pause_frac": pause_frac,
            "pose_slice": [0, 6],
            "gripper_index": 6,
            "task_ids": sorted(task_ids) if task_ids is not None else None,
            "episode_limit": episode_limit,
        },
        "boundary_versions": {
            "legacy": LEGACY_BOUNDARY_VERSION,
            "corrected": getattr(_MOLMO, "BOUNDARY_VERSION", "unknown"),
        },
        "dataset": {
            "root": str(root),
            "data_parquet_count": len(data_paths),
            "selected_episode_count": len(episode_results),
            "selected_frame_count": sum(row["frames"] for row in episode_results),
            "semantic_action_sha256": semantic_digest.hexdigest(),
            "metadata_sha256": {
                str(path.relative_to(root)): sha256_file(path) for path in metadata_files
            },
        },
        "provenance": {
            "git": _git_revision(repo),
            "source_sha256": {
                str(Path(__file__).resolve().relative_to(repo)): sha256_file(
                    Path(__file__).resolve()
                ),
                str(molmo_source.relative_to(repo)): sha256_file(molmo_source),
                str(event_source.relative_to(repo)): sha256_file(event_source),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "overall": aggregate_episode_metrics(episode_results),
        "tasks": [
            {
                "task_id": task_id,
                "task": task_strings[task_id],
                **aggregate_episode_metrics(per_task_rows[task_id]),
            }
            for task_id in sorted(per_task_rows)
        ],
        # Retain episode-level sufficient statistics so every aggregate can be
        # independently recomputed without the source dataset.  Boundary arrays
        # are summarized, not serialized, keeping the report small.
        "episodes": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "legacy_to_corrected_nearest",
                    "corrected_to_legacy_nearest",
                    "legacy_event_types",
                    "corrected_event_types",
                    "task",
                }
            }
            for row in episode_results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon-max", type=int, default=24)
    parser.add_argument("--min-seg", type=int, default=8)
    parser.add_argument("--pause-frac", type=float, default=0.15)
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        default=None,
        help="repeat to restrict the audit; default is every task",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="deterministically audit only the lowest selected episode indices",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not 1 <= args.min_seg <= args.horizon_max:
        raise ValueError("min-seg must lie in [1, horizon-max]")
    if args.pause_frac < 0 or not math.isfinite(args.pause_frac):
        raise ValueError("pause-frac must be finite and nonnegative")

    result = audit_dataset(
        args.root,
        horizon_max=args.horizon_max,
        min_seg=args.min_seg,
        pause_frac=args.pause_frac,
        task_ids=set(args.task_id) if args.task_id is not None else None,
        episode_limit=args.episode_limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)

    overall = result["overall"]
    print(
        f"audited {overall['episodes']} episodes / {overall['frames']} frames; "
        f"exact partitions={100 * overall['episode_exact_partition_fraction']:.2f}%"
    )
    print(
        "legacy intervals requiring recaption: "
        f"{overall['legacy_intervals_requiring_recaption']}/"
        f"{overall['legacy_segments']} "
        f"({100 * overall['legacy_interval_recaption_fraction']:.2f}%)"
    )
    print(
        "legacy-label frame ambiguity: "
        f"{100 * overall['legacy_label_frame_ambiguity_fraction']:.2f}%"
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
