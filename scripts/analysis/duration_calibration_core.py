"""Pure calibration utilities for the spline duration head.

This module intentionally has no LeRobot dependency.  The GPU inference job
emits raw, pre-rounding duration predictions; the functions below fit and
audit monotone scalar maps that can be applied to duration without touching
the sampled B-spline shape channels.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


FOLDS = ("fit", "select", "audit")
FAMILIES = ("identity", "log_affine", "snap", "isotonic", "event_isotonic_snap")


def _stable_score(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def assign_episode_folds(
    episode_tasks: Mapping[int, str], salt: str
) -> dict[int, str]:
    """Assign whole episodes to balanced, task-stratified folds.

    Assignment never looks at actions, event labels, or model predictions.
    Within each task, episodes are ordered by a salted hash and dealt into the
    three folds round-robin, so every sufficiently represented task appears in
    every fold without leaking frames from one episode across folds.
    """

    if not salt:
        raise ValueError("fold salt must be nonempty")
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode, task in episode_tasks.items():
        if episode < 0 or not task:
            raise ValueError("episode ids must be nonnegative and tasks nonempty")
        by_task[str(task)].append(int(episode))
    result: dict[int, str] = {}
    for task, episodes in sorted(by_task.items()):
        ordered = sorted(episodes, key=lambda ep: _stable_score(salt, task, ep))
        for ordinal, episode in enumerate(ordered):
            result[episode] = FOLDS[ordinal % len(FOLDS)]
    return result


def deterministic_frame_sample(
    rows: Iterable[dict], *, per_task_fold: int, salt: str
) -> list[dict]:
    """Uniform deterministic frame sample within each task/fold.

    Rows are ranked only by coordinates and the supplied salt.  In particular,
    duration target and event type are deliberately absent from the score, so
    the retained sample estimates the natural training-frame distribution.
    """

    if per_task_fold <= 0:
        raise ValueError("per_task_fold must be positive")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        fold = str(row["fold"])
        if fold not in FOLDS:
            raise ValueError(f"invalid fold {fold!r}")
        task = str(row["task_description"])
        groups[(task, fold)].append(dict(row))
    selected: list[dict] = []
    for (task, fold), group in sorted(groups.items()):
        if len(group) < per_task_fold:
            raise ValueError(
                f"task/fold {(task, fold)!r} has only {len(group)} eligible frames; "
                f"requested {per_task_fold}"
            )
        ranked = sorted(
            group,
            key=lambda row: _stable_score(
                salt,
                task,
                fold,
                int(row["episode_index"]),
                int(row["frame_index"]),
            ),
        )
        selected.extend(ranked[:per_task_fold])
    return sorted(
        selected,
        key=lambda row: (
            FOLDS.index(str(row["fold"])),
            str(row["task_description"]),
            int(row["episode_index"]),
            int(row["frame_index"]),
        ),
    )


def _as_vectors(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x_array) == 0 or len(x_array) != len(y_array):
        raise ValueError("x and y must be nonempty vectors of equal length")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("x and y must be finite")
    if np.any(x_array <= 0):
        raise ValueError("duration predictions must be positive")
    return x_array, y_array


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> dict:
    """Weighted PAVA with a compact piecewise-constant serialization."""

    order = np.argsort(x, kind="mergesort")
    x_sorted, y_sorted = x[order], y[order]
    unique, first, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    sums = np.add.reduceat(y_sorted, first)
    blocks: list[dict[str, float]] = []
    for value, count, total in zip(unique, counts, sums):
        blocks.append(
            {
                "x_min": float(value),
                "x_max": float(value),
                "weight": float(count),
                "sum": float(total),
            }
        )
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            left_mean = left["sum"] / left["weight"]
            right_mean = right["sum"] / right["weight"]
            if left_mean <= right_mean:
                break
            blocks[-2:] = [
                {
                    "x_min": left["x_min"],
                    "x_max": right["x_max"],
                    "weight": left["weight"] + right["weight"],
                    "sum": left["sum"] + right["sum"],
                }
            ]
    values = [block["sum"] / block["weight"] for block in blocks]
    thresholds = [
        0.5 * (blocks[index]["x_max"] + blocks[index + 1]["x_min"])
        for index in range(len(blocks) - 1)
    ]
    return {
        "thresholds": [float(value) for value in thresholds],
        "values": [float(value) for value in values],
        "n_blocks": len(blocks),
    }


def _best_snap_threshold(
    x: np.ndarray, y: np.ndarray, *, minimum: int, cap: int
) -> float | None:
    unique = np.unique(x)
    candidates = [float("inf")]
    if len(unique) == 1:
        candidates.append(float(unique[0]))
    else:
        candidates.extend(float(0.5 * (a + b)) for a, b in zip(unique[:-1], unique[1:]))
        candidates.extend((float(unique[0]), float(unique[-1])))
    best: tuple[float, float, float] | None = None
    best_threshold: float | None = None
    for threshold in candidates:
        raw = np.where(x >= threshold, cap, x)
        prediction = np.clip(np.rint(raw), minimum, cap)
        mae = float(np.mean(np.abs(prediction - y)))
        # Prefer no snap, then a higher/more conservative threshold, on ties.
        no_snap_penalty = 0.0 if math.isinf(threshold) else 1.0
        key = (mae, no_snap_penalty, -threshold)
        if best is None or key < best:
            best = key
            best_threshold = None if math.isinf(threshold) else threshold
    return best_threshold


def fit_mapping(
    family: str,
    x: Sequence[float],
    y: Sequence[float],
    *,
    minimum: int,
    cap: int,
) -> dict:
    """Fit one predeclared global monotone duration calibrator."""

    if family not in FAMILIES:
        raise ValueError(f"unknown calibration family {family!r}")
    if not 1 <= minimum <= cap:
        raise ValueError("invalid duration bounds")
    x_array, y_array = _as_vectors(x, y)
    if np.any((y_array < minimum) | (y_array > cap)):
        raise ValueError("targets outside duration bounds")
    mapping: dict = {
        "family": family,
        "minimum": int(minimum),
        "cap": int(cap),
        "n_fit_draws": int(len(x_array)),
    }
    if family == "identity":
        mapping["parameters"] = {}
    elif family == "log_affine":
        lx, ly = np.log(x_array), np.log(y_array)
        variance = float(np.sum((lx - lx.mean()) ** 2))
        slope = 0.0 if variance == 0 else float(
            np.sum((lx - lx.mean()) * (ly - ly.mean())) / variance
        )
        slope = max(0.0, slope)
        intercept = float(ly.mean() - slope * lx.mean())
        mapping["parameters"] = {"slope": slope, "intercept": intercept}
    elif family == "snap":
        mapping["parameters"] = {
            "threshold": _best_snap_threshold(
                x_array, y_array, minimum=minimum, cap=cap
            )
        }
    elif family == "isotonic":
        mapping["parameters"] = _fit_isotonic(x_array, y_array)
    else:
        event = y_array < cap
        event_model = _fit_isotonic(x_array[event], y_array[event]) if event.any() else None
        mapping["parameters"] = {
            "event_model": event_model,
            "threshold": _best_snap_threshold(
                x_array, y_array, minimum=minimum, cap=cap
            ),
        }
    return mapping


def _apply_isotonic(parameters: Mapping, x: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters["values"], dtype=np.float64)
    thresholds = np.asarray(parameters["thresholds"], dtype=np.float64)
    if len(values) != len(thresholds) + 1 or len(values) == 0:
        raise ValueError("invalid isotonic serialization")
    indices = np.searchsorted(thresholds, x, side="right")
    return values[indices]


def apply_mapping(mapping: Mapping, x: Sequence[float], *, rounded: bool = True) -> np.ndarray:
    """Apply a serialized mapping; shape/action channels are not inputs."""

    raw = np.asarray(x, dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("duration predictions must be finite and positive")
    family = str(mapping["family"])
    parameters = mapping["parameters"]
    minimum, cap = int(mapping["minimum"]), int(mapping["cap"])
    if family == "identity":
        calibrated = raw
    elif family == "log_affine":
        calibrated = np.exp(
            float(parameters["slope"]) * np.log(raw) + float(parameters["intercept"])
        )
    elif family == "snap":
        threshold = parameters.get("threshold")
        calibrated = raw if threshold is None else np.where(raw >= float(threshold), cap, raw)
    elif family == "isotonic":
        calibrated = _apply_isotonic(parameters, raw)
    elif family == "event_isotonic_snap":
        event_model = parameters.get("event_model")
        calibrated = raw if event_model is None else _apply_isotonic(event_model, raw)
        threshold = parameters.get("threshold")
        if threshold is not None:
            calibrated = np.where(raw >= float(threshold), cap, calibrated)
    else:
        raise ValueError(f"unknown calibration family {family!r}")
    calibrated = np.clip(calibrated, minimum, cap)
    return np.rint(calibrated).astype(np.int64) if rounded else calibrated


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def cap_confusion(y_true: Sequence[float], y_pred: Sequence[float], cap: int) -> dict:
    truth = np.asarray(y_true).reshape(-1) >= cap
    prediction = np.asarray(y_pred).reshape(-1) >= cap
    if len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("truth and prediction must be nonempty and aligned")
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    balanced = 0.5 * (recall + specificity) if np.isfinite([recall, specificity]).all() else float("nan")
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": balanced,
        "cap_recall": recall,
        "cap_precision": precision,
        "event_specificity": specificity,
    }


def fit_cap_threshold(
    x: Sequence[float], y: Sequence[float], *, cap: int
) -> tuple[float, dict]:
    """Fit a raw-duration threshold for cap-vs-event discrimination.

    Balanced accuracy is the primary criterion; ties prefer the higher, more
    conservative threshold.  This diagnostic is separate from the MAE-optimal
    snap threshold used by the candidate calibrator.
    """

    x_array, y_array = _as_vectors(x, y)
    truth = y_array >= cap
    if truth.all() or (~truth).all():
        raise ValueError("cap-threshold fitting requires both cap and event targets")
    unique = np.unique(x_array)
    if len(unique) == 1:
        candidates = [float(unique[0])]
    else:
        candidates = [float(unique[0])]
        candidates.extend(float(0.5 * (a + b)) for a, b in zip(unique[:-1], unique[1:]))
        candidates.append(float(unique[-1]))
    best_threshold = candidates[0]
    best_confusion = cap_confusion(y_array, np.where(x_array >= best_threshold, cap, cap - 1), cap)
    for threshold in candidates[1:]:
        confusion = cap_confusion(y_array, np.where(x_array >= threshold, cap, cap - 1), cap)
        key = (confusion["balanced_accuracy"], threshold)
        best_key = (best_confusion["balanced_accuracy"], best_threshold)
        if key > best_key:
            best_threshold, best_confusion = threshold, confusion
    return best_threshold, best_confusion


def calibration_metrics(
    y_true: Sequence[float], y_pred: Sequence[float], *, minimum: int, cap: int
) -> dict:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("truth and prediction must be nonempty and aligned")
    error = prediction - truth
    pearson = float("nan")
    spearman = float("nan")
    if np.std(truth) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
        spearman = float(np.corrcoef(_rank_average(truth), _rank_average(prediction))[0, 1])
    event = truth < cap
    histogram_truth = {str(value): int(np.sum(truth == value)) for value in range(minimum, cap + 1)}
    histogram_pred = {
        str(value): int(np.sum(prediction == value)) for value in range(minimum, cap + 1)
    }
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "true_mean": float(np.mean(truth)),
        "predicted_mean": float(np.mean(prediction)),
        "event_mae": float(np.mean(np.abs(error[event]))) if event.any() else float("nan"),
        "cap_mae": float(np.mean(np.abs(error[~event]))) if (~event).any() else float("nan"),
        "cap_confusion": cap_confusion(truth, prediction, cap),
        "true_histogram": histogram_truth,
        "predicted_histogram": histogram_pred,
    }


def choose_candidate(
    fit_x: Sequence[float],
    fit_y: Sequence[float],
    select_x: Sequence[float],
    select_y: Sequence[float],
    *,
    minimum: int,
    cap: int,
) -> tuple[str, dict[str, dict], dict[str, dict]]:
    """Predeclared selection rule, using the selection fold only once.

    A non-identity map must improve overall MAE by both 0.25 steps and 5%,
    while worsening neither non-cap MAE by >0.25 nor cap balanced accuracy by
    >0.02.  These guardrails prevent a majority-cap shortcut from winning.
    """

    fx, fy = _as_vectors(fit_x, fit_y)
    sx, sy = _as_vectors(select_x, select_y)
    mappings: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    repeated = {family: index for index, family in enumerate(FAMILIES)}
    for family in FAMILIES:
        mapping = fit_mapping(family, fx, fy, minimum=minimum, cap=cap)
        prediction = apply_mapping(mapping, sx)
        mappings[family] = mapping
        scores[family] = calibration_metrics(sy, prediction, minimum=minimum, cap=cap)
    baseline = scores["identity"]
    eligible = []
    for family in FAMILIES[1:]:
        score = scores[family]
        enough_gain = (
            score["mae"] <= baseline["mae"] - 0.25
            and score["mae"] <= 0.95 * baseline["mae"]
        )
        event_safe = score["event_mae"] <= baseline["event_mae"] + 0.25
        cap_safe = (
            score["cap_confusion"]["balanced_accuracy"]
            >= baseline["cap_confusion"]["balanced_accuracy"] - 0.02
        )
        if enough_gain and event_safe and cap_safe:
            eligible.append(family)
    chosen = min(
        eligible,
        key=lambda family: (scores[family]["mae"], repeated[family]),
        default="identity",
    )
    for family in FAMILIES:
        scores[family]["eligible"] = family == "identity" or family in eligible
        scores[family]["selected"] = family == chosen
    return chosen, mappings, scores
