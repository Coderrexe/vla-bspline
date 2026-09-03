"""Normalization-stat provenance and compatibility checks for the spline head.

The source-tree JSON is an input to *new training runs*, not a runtime
dependency of a saved policy.  :func:`resolve_spline_stats` copies the exact
validated payload into the policy config on first construction.  Subsequent
checkpoint loads use that embedded payload even if the package JSON has moved
or changed; the model state dict contains the same mean/std as persistent
buffers as a second, tensor-native copy.

Older checkpoints predate the embedded payload.  They retain their historical
behaviour (load the configured JSON) while receiving explicit warnings for
metadata fields that old JSONs did not record.
"""

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any


STATS_CONTRACT_VERSION = 1
EVENT_BOUNDARY_SEMANTICS = "production_event_index_k_exclusive"
FIXED_BOUNDARY_SEMANTICS = "fixed_horizon_actions_0_to_h_minus_1"

ACTION_LAYOUTS = {
    "eef7": {
        "pose_dims": [0, 1, 2, 3, 4, 5],
        "grip_idx": 6,
        "pass_dims": [],
    },
    "robocasa12": {
        "pose_dims": [5, 6, 7, 8, 9, 10],
        "grip_idx": 11,
        "pass_dims": [0, 1, 2, 3, 4],
    },
}


def _legacy_warning(origin: str, field: str, assumed: Any) -> None:
    warnings.warn(
        f"Legacy spline stats from {origin} do not record {field!r}; "
        f"assuming {assumed!r}. Regenerate the stats to make this contract explicit.",
        UserWarning,
        stacklevel=3,
    )


def _require_equal(stats: Mapping[str, Any], key: str, expected: Any, origin: str) -> None:
    if key not in stats:
        raise ValueError(f"spline stats from {origin} are missing required field {key!r}")
    actual = stats[key]
    if isinstance(expected, float):
        matches = isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"spline stats from {origin} have {key}={actual!r}, but the policy "
            f"requires {expected!r}"
        )


def _validate_vector(value: Any, length: int, name: str, origin: str, *, positive=False) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(
            f"spline stats from {origin} field {name!r} must have length {length}"
        )
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"spline stats from {origin} field {name!r} contains a non-finite value")
        if positive and float(item) <= 0:
            raise ValueError(f"spline stats from {origin} field {name!r} must be strictly positive")


def _validate_matrix(
    value: Any,
    rows: int,
    cols: int,
    name: str,
    origin: str,
    *,
    positive=False,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(
            f"spline stats from {origin} field {name!r} must have shape ({rows}, {cols})"
        )
    for row in value:
        _validate_vector(row, cols, name, origin, positive=positive)


def validate_spline_stats(stats: Mapping[str, Any], cfg: Any, origin: str) -> dict[str, Any]:
    """Validate and canonicalize one stats payload against a spline config.

    Legacy-only metadata omissions are inferred from the policy config with a
    warning, preserving old checkpoint loading without silently accepting an
    explicitly wrong value.
    """

    if not isinstance(stats, Mapping):
        raise ValueError(f"spline stats from {origin} must be a JSON object")
    result = copy.deepcopy(dict(stats))
    if "stats_contract_version" in result:
        _require_equal(result, "stats_contract_version", STATS_CONTRACT_VERSION, origin)

    n_ctrl = int(cfg.n_ctrl)
    degree = int(cfg.spline_degree)
    _require_equal(result, "n_ctrl", n_ctrl, origin)
    if "degree" in result:
        _require_equal(result, "degree", degree, origin)
    else:
        # The first RoboCasa stats generator omitted the already-fixed cubic
        # degree. Preserve those checkpoints while making every re-save explicit.
        _legacy_warning(origin, "degree", degree)
        result["degree"] = degree

    predict_duration = bool(cfg.predict_duration)
    if predict_duration:
        horizon_fields = [key for key in ("horizon_max", "h_max") if key in result]
        if not horizon_fields:
            raise ValueError(
                f"spline stats from {origin} are missing 'horizon_max' (legacy alias 'h_max')"
            )
        horizon_values = {int(result[key]) for key in horizon_fields}
        if len(horizon_values) != 1:
            raise ValueError(f"spline stats from {origin} contain conflicting horizon fields")
        actual_horizon = horizon_values.pop()
        if actual_horizon != int(cfg.horizon_max):
            raise ValueError(
                f"spline stats from {origin} have horizon_max={actual_horizon}, but the policy "
                f"requires {cfg.horizon_max}"
            )
        _require_equal(result, "min_seg", int(cfg.min_seg), origin)
        _require_equal(result, "pause_frac", float(cfg.pause_frac), origin)
        result["horizon_max"] = actual_horizon
        expected_boundary = EVENT_BOUNDARY_SEMANTICS

        expected_weight = getattr(cfg, "fit_end_weight", None)
        if "fit_end_weight" in result:
            actual_weight = result["fit_end_weight"]
            weight_matches = (
                actual_weight is None and expected_weight is None
            ) or (
                actual_weight is not None
                and expected_weight is not None
                and math.isclose(
                    float(actual_weight),
                    float(expected_weight),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if not weight_matches:
                raise ValueError(
                    f"spline stats from {origin} have fit_end_weight={actual_weight!r}, "
                    f"but the policy requires {expected_weight!r}"
                )
        elif expected_weight is not None:
            raise ValueError(
                f"spline stats from {origin} do not record the policy's non-default "
                f"fit_end_weight={expected_weight!r}"
            )
    else:
        _require_equal(result, "horizon", int(cfg.horizon), origin)
        expected_boundary = FIXED_BOUNDARY_SEMANTICS

    expected_speed_aug = getattr(cfg, "speed_aug", None)
    actual_speed_aug = result.get("speed_aug")
    if expected_speed_aug is None and actual_speed_aug is not None:
        raise ValueError(
            f"spline stats from {origin} were generated with speed_aug={actual_speed_aug!r}, "
            "but the policy has speed_aug=None"
        )
    if expected_speed_aug is not None:
        if actual_speed_aug is None:
            raise ValueError(
                f"spline stats from {origin} do not record the policy's speed_aug="
                f"{expected_speed_aug!r}"
            )
        expected_speeds = [float(value) for value in expected_speed_aug]
        actual_speeds = [float(value) for value in actual_speed_aug]
        if actual_speeds != expected_speeds:
            raise ValueError(
                f"spline stats from {origin} have speed_aug={actual_speeds!r}, but the "
                f"policy requires {expected_speeds!r}"
            )

    layout = str(getattr(cfg, "action_layout", "eef7"))
    if layout not in ACTION_LAYOUTS:
        raise ValueError(f"cannot validate unknown action layout {layout!r}")
    if "action_layout" in result:
        _require_equal(result, "action_layout", layout, origin)
    else:
        _legacy_warning(origin, "action_layout", layout)
        result["action_layout"] = layout

    spec = ACTION_LAYOUTS[layout]
    for key, expected in spec.items():
        if key in result:
            _require_equal(result, key, expected, origin)
        else:
            result[key] = copy.deepcopy(expected)

    # ``segmentation`` was emitted by the first shared RoboCasa generator.
    boundary_keys = [key for key in ("boundary_semantics", "segmentation") if key in result]
    if not boundary_keys:
        _legacy_warning(origin, "boundary_semantics", expected_boundary)
    else:
        for key in boundary_keys:
            _require_equal(result, key, expected_boundary, origin)
    result["boundary_semantics"] = expected_boundary
    result["stats_contract_version"] = STATS_CONTRACT_VERSION

    _validate_matrix(result.get("pose_ctrl_mean"), n_ctrl, 6, "pose_ctrl_mean", origin)
    _validate_matrix(
        result.get("pose_ctrl_std"), n_ctrl, 6, "pose_ctrl_std", origin, positive=True
    )
    _validate_vector(result.get("grip_ctrl_mean"), n_ctrl, "grip_ctrl_mean", origin)
    _validate_vector(
        result.get("grip_ctrl_std"), n_ctrl, "grip_ctrl_std", origin, positive=True
    )
    if predict_duration:
        _validate_vector([result.get("logT_mean")], 1, "logT_mean", origin)
        _validate_vector([result.get("logT_std")], 1, "logT_std", origin, positive=True)

    n_pass = len(spec["pass_dims"])
    have_pass_mean = "pass_ctrl_mean" in result
    have_pass_std = "pass_ctrl_std" in result
    if have_pass_mean != have_pass_std:
        raise ValueError(
            f"spline stats from {origin} must provide both pass_ctrl_mean and pass_ctrl_std"
        )
    if have_pass_mean:
        _validate_matrix(result["pass_ctrl_mean"], n_ctrl, n_pass, "pass_ctrl_mean", origin)
        _validate_matrix(
            result["pass_ctrl_std"], n_ctrl, n_pass, "pass_ctrl_std", origin, positive=True
        )

    return result


def resolve_spline_stats(cfg: Any, module_dir: str) -> tuple[dict[str, Any], str]:
    """Return validated stats, preferring the checkpoint-embedded payload.

    On a fresh/legacy config, the source JSON is read once and its canonical
    payload is attached to ``cfg.embedded_spline_stats`` so ``save_pretrained``
    serializes it into ``config.json``.
    """

    embedded = getattr(cfg, "embedded_spline_stats", None)
    if embedded is not None:
        origin = "checkpoint-embedded config"
        return validate_spline_stats(embedded, cfg, origin), origin

    stats_name = cfg.spline_stats_file_v2 if cfg.predict_duration else cfg.spline_stats_file
    stats_path = os.path.join(module_dir, stats_name)
    with open(stats_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    origin = stats_path
    normalized = validate_spline_stats(raw, cfg, origin)
    cfg.embedded_spline_stats = copy.deepcopy(normalized)
    return normalized, origin


def consume_legacy_normalization_missing_keys(
    state_dict: Mapping[str, Any], prefix: str, missing_keys: list[str]
) -> bool:
    """Suppress exactly the missing-buffer pair used by legacy checkpoints.

    Returns ``True`` only when *both* buffers are absent. A partial pair is
    intentionally left untouched so strict PyTorch loading reports corruption.
    """

    mean_key = prefix + "_tgt_mean"
    std_key = prefix + "_tgt_std"
    if mean_key in state_dict or std_key in state_dict:
        return False
    for key in (mean_key, std_key):
        if key in missing_keys:
            missing_keys.remove(key)
    return True
