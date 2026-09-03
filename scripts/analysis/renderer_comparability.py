#!/usr/bin/env python3
"""Compare repeated EGL and OSMesa LIBERO renderer probes.

The input artifacts are produced by ``libero_renderer_probe.py``.  At least two
replicates per backend are required so cross-backend distances are interpreted
against EGL's own replay variation rather than against an assumed exact GPU
renderer.  This analyzer is intentionally conservative: it tests equivalence,
not merely failure to find a significant difference.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL = "libero_renderer_comparability_probe_v1"
ANALYSIS_PROTOCOL = "libero_egl_osmesa_comparability_analysis_v1"
DEFAULT_THRESHOLDS = {
    "raw_cross_p95_mae_uint8": 1.0,
    "raw_cross_p05_psnr_db": 40.0,
    "raw_cross_p05_ssim": 0.995,
    "processed_cross_p95_mae": 0.01,
    "processed_cross_p05_cosine": 0.999,
    "relative_to_egl_within_multiplier": 2.0,
    "success_equivalence_margin": 0.05,
    "per_task_success_point_margin": 0.10,
    "cross_backend_discord_excess_margin": 0.05,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(np.percentile(np.asarray(materialized, dtype=np.float64), percentile))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_comparisons": len(records)}
    if not records:
        return result
    excluded = {"coordinate", "camera", "left", "right", "kind"}
    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key not in excluded and isinstance(value, (int, float)) and value is not None
        }
    )
    for key in numeric_keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        result[key] = {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "p05": _percentile(values, 5),
            "p95": _percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }
    return result


def _to_image(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"SSIM expects one 3-D image; got {value.shape}")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    return value


def _ssim(left: np.ndarray, right: np.ndarray, data_range: float) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    first = _to_image(left)
    second = _to_image(right)
    channel_axis = -1 if first.shape[-1] in (3, 4) else None
    return float(
        structural_similarity(
            first,
            second,
            channel_axis=channel_axis,
            data_range=data_range,
        )
    )


def array_metrics(left: np.ndarray, right: np.ndarray, *, raw: bool) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"array shape mismatch: {left.shape} != {right.shape}")
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    delta = np.abs(first - second)
    data_range = 255.0 if raw else float(max(np.ptp(first), np.ptp(second), 1e-12))
    mse = float(np.mean(np.square(first - second)))
    denominator = float(np.linalg.norm(first.ravel()) * np.linalg.norm(second.ravel()))
    cosine = (
        float(np.dot(first.ravel(), second.ravel()) / denominator)
        if denominator
        else float(np.array_equal(first, second))
    )
    return {
        "mae": float(np.mean(delta)),
        "rmse": math.sqrt(mse),
        "max_abs": float(np.max(delta)),
        "exact_fraction": float(np.mean(delta == 0)),
        "greater_than_1_fraction": float(np.mean(delta > 1.0)) if raw else None,
        "greater_than_2_fraction": float(np.mean(delta > 2.0)) if raw else None,
        "greater_than_5_fraction": float(np.mean(delta > 5.0)) if raw else None,
        "psnr_db": (
            # A finite sentinel keeps the JSON RFC-compliant while preserving
            # the useful meaning of an exact comparison.
            999.0 if mse == 0 else 20.0 * math.log10(data_range / math.sqrt(mse))
        ),
        "cosine": cosine,
        "ssim": _ssim(first, second, data_range) if raw else None,
    }


def _load_probe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("protocol") != PROTOCOL:
        raise ValueError(f"not a renderer probe: {path}")
    backend = payload.get("renderer_backend")
    if backend not in {"egl", "osmesa"}:
        raise ValueError(f"invalid renderer backend in {path}: {backend}")
    arrays_path = Path(payload["arrays_file"])
    if not arrays_path.is_absolute():
        arrays_path = path.parent / arrays_path
    arrays_path = arrays_path.resolve()
    if _sha256(arrays_path) != payload.get("arrays_sha256"):
        raise ValueError(f"array sidecar hash mismatch: {arrays_path}")
    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    arrays_file.close()
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for episode in payload["episodes"]:
        coordinate = (int(episode["task_id"]), int(episode["state_id"]))
        if coordinate in records:
            raise ValueError(f"duplicate coordinate {coordinate} in {path}")
        for collection in ("raw_cameras", "processed_cameras"):
            for metadata in episode[collection].values():
                key = metadata["array_key"]
                if key not in arrays:
                    raise ValueError(f"missing sidecar array {key} in {path}")
                array = arrays[key]
                if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                    raise ValueError(f"array metadata mismatch for {key} in {path}")
                digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
                if digest != metadata["sha256"]:
                    raise ValueError(f"array content hash mismatch for {key} in {path}")
        records[coordinate] = episode
    payload["_path"] = str(path.resolve())
    payload["_arrays"] = arrays
    payload["_records"] = records
    return payload


def _identity(probe: dict[str, Any]) -> dict[str, Any]:
    determinism = probe["runtime"]["determinism"]
    return {
        "checkpoint_config_sha256": probe["checkpoint_config_sha256"],
        "model_sha256": probe["model_sha256"],
        "suite": probe["suite"],
        "task_ids": probe["task_ids"],
        "state_ids": probe["state_ids"],
        "seed_base": probe["seed_base"],
        "control_frequency_hz": probe["control_frequency_hz"],
        "source_identity": probe["source"],
        "python_version": probe["runtime"]["python_version"],
        "software_versions": probe["runtime"]["software_versions"],
        "numpy_version": determinism["numpy_version"],
        "torch_version": determinism["torch_version"],
        "torch_cuda_version": determinism["torch_cuda_version"],
        "cudnn_version": determinism["cudnn_version"],
        # UUID and PCI location may differ across otherwise identical L40S
        # allocations. Device model/capability/memory may not.
        "cuda_devices": determinism["cuda_devices"],
    }


def _episode_array(
    probe: dict[str, Any], episode: dict[str, Any], collection: str, camera: str
) -> np.ndarray:
    return probe["_arrays"][episode[collection][camera]["array_key"]]


def _compare_probe_pair(
    left: dict[str, Any], right: dict[str, Any], *, category: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_coordinates = set(left["_records"])
    right_coordinates = set(right["_records"])
    if left_coordinates != right_coordinates:
        raise ValueError("probe coordinate grids differ")
    raw_records: list[dict[str, Any]] = []
    processed_records: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    physics_equal = True
    model_equal = True
    for coordinate in sorted(left_coordinates):
        first = left["_records"][coordinate]
        second = right["_records"][coordinate]
        physics_equal = physics_equal and (
            first["initial_mujoco_integration_state_sha256"]
            == second["initial_mujoco_integration_state_sha256"]
        )
        model_equal = model_equal and (
            first["compiled_model_xml_sha256"] == second["compiled_model_xml_sha256"]
        )
        for collection, destination, raw in (
            ("raw_cameras", raw_records, True),
            ("processed_cameras", processed_records, False),
        ):
            cameras = set(first[collection])
            if cameras != set(second[collection]):
                raise ValueError(f"camera schema mismatch at {coordinate}: {collection}")
            for camera in sorted(cameras):
                record = {
                    "coordinate": list(coordinate),
                    "camera": camera,
                    "left": left["_path"],
                    "right": right["_path"],
                    "kind": category,
                    **array_metrics(
                        _episode_array(left, first, collection, camera),
                        _episode_array(right, second, collection, camera),
                        raw=raw,
                    ),
                }
                destination.append(record)
        outcomes.append(
            {
                "task_id": coordinate[0],
                "state_id": coordinate[1],
                "left_success": bool(first["success"]),
                "right_success": bool(second["success"]),
                "difference": int(second["success"]) - int(first["success"]),
                "action_trace_equal": (
                    first["executed_action_trace_sha256"]
                    == second["executed_action_trace_sha256"]
                ),
                "first_action_equal": (
                    first["first_action_sha256"] == second["first_action_sha256"]
                ),
            }
        )
    return raw_records, processed_records, {
        "category": category,
        "left": left["_path"],
        "right": right["_path"],
        "initial_mujoco_integration_state_all_equal": physics_equal,
        "compiled_model_xml_all_equal": model_equal,
        "n_coordinates": len(outcomes),
        "left_success_rate": float(np.mean([row["left_success"] for row in outcomes])),
        "right_success_rate": float(np.mean([row["right_success"] for row in outcomes])),
        "success_difference_right_minus_left": float(
            np.mean([row["difference"] for row in outcomes])
        ),
        "outcome_disagreement_fraction": float(
            np.mean([row["left_success"] != row["right_success"] for row in outcomes])
        ),
        "first_action_agreement_fraction": float(
            np.mean([row["first_action_equal"] for row in outcomes])
        ),
        "action_trace_agreement_fraction": float(
            np.mean([row["action_trace_equal"] for row in outcomes])
        ),
        "outcomes": outcomes,
    }


def _stratified_bootstrap_ci(
    coordinate_effects: dict[tuple[int, int], float], *, seed: int, draws: int = 20_000
) -> list[float]:
    by_task: dict[int, list[float]] = defaultdict(list)
    for (task_id, _), effect in coordinate_effects.items():
        by_task[task_id].append(effect)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    task_ids = sorted(by_task)
    for draw in range(draws):
        task_means = []
        for task_id in task_ids:
            task_values = np.asarray(by_task[task_id], dtype=np.float64)
            sample = rng.choice(task_values, size=len(task_values), replace=True)
            task_means.append(float(np.mean(sample)))
        values[draw] = float(np.mean(task_means))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze(paths: list[Path], *, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    probes = [_load_probe(path.resolve()) for path in paths]
    groups = {
        backend: sorted(
            [probe for probe in probes if probe["renderer_backend"] == backend],
            key=lambda probe: probe["replicate_index"],
        )
        for backend in ("egl", "osmesa")
    }
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("at least two EGL and two OSMesa replicates are required")
    for backend, group in groups.items():
        indices = [probe["replicate_index"] for probe in group]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate {backend} replicate indices")
    reference_identity = _identity(probes[0])
    identity_mismatches = [
        {"path": probe["_path"], "identity": _identity(probe)}
        for probe in probes
        if _identity(probe) != reference_identity
    ]
    if identity_mismatches:
        raise ValueError(f"probe scientific identities differ: {identity_mismatches}")

    all_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_processed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcome_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for backend, group in groups.items():
        category = f"within_{backend}"
        for left, right in itertools.combinations(group, 2):
            raw, processed, outcomes = _compare_probe_pair(left, right, category=category)
            all_raw[category].extend(raw)
            all_processed[category].extend(processed)
            outcome_pairs[category].append(outcomes)
    for egl in groups["egl"]:
        for osmesa in groups["osmesa"]:
            raw, processed, outcomes = _compare_probe_pair(
                egl, osmesa, category="cross_backend"
            )
            all_raw["cross_backend"].extend(raw)
            all_processed["cross_backend"].extend(processed)
            outcome_pairs["cross_backend"].append(outcomes)

    summaries = {
        category: {
            "raw_camera": {
                **_summary(all_raw[category]),
                "by_camera": {
                    camera: _summary(
                        [row for row in all_raw[category] if row["camera"] == camera]
                    )
                    for camera in sorted({row["camera"] for row in all_raw[category]})
                },
            },
            "processed_camera": {
                **_summary(all_processed[category]),
                "by_camera": {
                    camera: _summary(
                        [
                            row
                            for row in all_processed[category]
                            if row["camera"] == camera
                        ]
                    )
                    for camera in sorted(
                        {row["camera"] for row in all_processed[category]}
                    )
                },
            },
            "closed_loop_pairs": outcome_pairs[category],
        }
        for category in ("within_egl", "within_osmesa", "cross_backend")
    }

    coordinates = sorted(probes[0]["_records"])
    coordinate_effects: dict[tuple[int, int], float] = {}
    per_task: dict[str, list[float]] = defaultdict(list)
    for coordinate in coordinates:
        egl_mean = float(
            np.mean([probe["_records"][coordinate]["success"] for probe in groups["egl"]])
        )
        osmesa_mean = float(
            np.mean(
                [probe["_records"][coordinate]["success"] for probe in groups["osmesa"]]
            )
        )
        effect = osmesa_mean - egl_mean
        coordinate_effects[coordinate] = effect
        per_task[str(coordinate[0])].append(effect)
    success_effect = float(np.mean(list(coordinate_effects.values())))
    success_ci = _stratified_bootstrap_ci(coordinate_effects, seed=20_260_821)
    per_task_effects = {
        task_id: float(np.mean(values)) for task_id, values in sorted(per_task.items())
    }

    raw_cross = summaries["cross_backend"]["raw_camera"]
    raw_egl = summaries["within_egl"]["raw_camera"]
    processed_cross = summaries["cross_backend"]["processed_camera"]
    processed_egl = summaries["within_egl"]["processed_camera"]
    raw_osmesa = summaries["within_osmesa"]["raw_camera"]
    processed_osmesa = summaries["within_osmesa"]["processed_camera"]

    def statistic(summary: dict[str, Any], key: str, field: str) -> float:
        value = summary.get(key, {}).get(field)
        if value is None:
            raise ValueError(f"missing required metric {key}.{field}")
        return float(value)

    raw_cross_mae = statistic(raw_cross, "mae", "p95")
    raw_egl_mae = statistic(raw_egl, "mae", "p95")
    processed_cross_mae = statistic(processed_cross, "mae", "p95")
    processed_egl_mae = statistic(processed_egl, "mae", "p95")
    ssim_available = raw_cross.get("ssim", {}).get("p05") is not None
    initial_physics_equal = all(
        pair["initial_mujoco_integration_state_all_equal"]
        and pair["compiled_model_xml_all_equal"]
        for pairs in outcome_pairs.values()
        for pair in pairs
    )
    osmesa_outcomes_exact = all(
        pair["outcome_disagreement_fraction"] == 0
        and pair["first_action_agreement_fraction"] == 1
        and pair["action_trace_agreement_fraction"] == 1
        for pair in outcome_pairs["within_osmesa"]
    )
    osmesa_arrays_exact = (
        statistic(raw_osmesa, "exact_fraction", "min") == 1
        and statistic(processed_osmesa, "exact_fraction", "min") == 1
    )
    opengl_consistent_within_backend = all(
        all(probe["opengl"] == group[0]["opengl"] for probe in group[1:])
        for group in groups.values()
    )
    egl_discord = max(
        pair["outcome_disagreement_fraction"] for pair in outcome_pairs["within_egl"]
    )
    cross_discord = max(
        pair["outcome_disagreement_fraction"] for pair in outcome_pairs["cross_backend"]
    )
    gates = {
        "scientific_identity_exact": True,
        "initial_physics_and_model_exact": initial_physics_equal,
        "opengl_implementation_consistent_within_backend": (
            opengl_consistent_within_backend
        ),
        "osmesa_repeat_arrays_actions_outcomes_exact": (
            osmesa_arrays_exact and osmesa_outcomes_exact
        ),
        "raw_pixel_absolute": (
            raw_cross_mae <= limits["raw_cross_p95_mae_uint8"]
            and statistic(raw_cross, "psnr_db", "p05")
            >= limits["raw_cross_p05_psnr_db"]
            and (
                not ssim_available
                or statistic(raw_cross, "ssim", "p05")
                >= limits["raw_cross_p05_ssim"]
            )
        ),
        "raw_pixel_no_more_than_egl_variability": raw_cross_mae
        <= max(
            limits["raw_cross_p95_mae_uint8"],
            limits["relative_to_egl_within_multiplier"] * raw_egl_mae,
        ),
        "processed_tensor_absolute": (
            processed_cross_mae <= limits["processed_cross_p95_mae"]
            and statistic(processed_cross, "cosine", "p05")
            >= limits["processed_cross_p05_cosine"]
        ),
        "processed_no_more_than_egl_variability": processed_cross_mae
        <= max(
            limits["processed_cross_p95_mae"],
            limits["relative_to_egl_within_multiplier"] * processed_egl_mae,
        ),
        "closed_loop_overall_equivalence_95pct": (
            success_ci[0] >= -limits["success_equivalence_margin"]
            and success_ci[1] <= limits["success_equivalence_margin"]
        ),
        "closed_loop_per_task_point_margin": all(
            abs(effect) <= limits["per_task_success_point_margin"]
            for effect in per_task_effects.values()
        ),
        "closed_loop_discord_not_above_egl_variability": cross_discord
        <= egl_discord + limits["cross_backend_discord_excess_margin"],
    }
    gates["renderer_equivalence_for_mixed_backend_claims"] = all(gates.values())
    return {
        "schema_version": 1,
        "protocol": ANALYSIS_PROTOCOL,
        "decision": (
            "PASS_EQUIVALENCE_GATE"
            if gates["renderer_equivalence_for_mixed_backend_claims"]
            else "FAIL_EQUIVALENCE_GATE_DO_NOT_MIX_BACKENDS"
        ),
        "warning": (
            "Even a pass does not make historical EGL and OSMesa rows a matched "
            "experiment. Re-evaluate every claimed comparator under the same "
            "renderer whenever feasible."
        ),
        "thresholds_preregistered_before_probe": limits,
        "ssim_available": ssim_available,
        "scientific_identity": reference_identity,
        "probe_paths": {
            backend: [probe["_path"] for probe in group]
            for backend, group in groups.items()
        },
        "opengl_implementations": {
            backend: [probe["opengl"] for probe in group]
            for backend, group in groups.items()
        },
        "n_coordinates": len(coordinates),
        "metric_summaries": summaries,
        "success_renderer_effect_osmesa_minus_egl": {
            "point_difference": success_effect,
            "fixed_suite_state_stratified_bootstrap_95pct_ci": success_ci,
            "per_task_point_differences": per_task_effects,
            "within_egl_max_outcome_disagreement": egl_discord,
            "cross_backend_max_outcome_disagreement": cross_discord,
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.probes)
    _atomic_write_json_new(args.out, result)
    print(result["decision"])
    print(json.dumps(result["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
