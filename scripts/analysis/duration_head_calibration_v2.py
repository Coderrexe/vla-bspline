"""Leakage-controlled duration-head calibration on LIBERO training demos.

The command has three deliberately separate stages:

``prepare``
    Read action Parquets only, reproduce production event targets, split whole
    episodes into fit/select/audit folds, and write a locked frame manifest.
    No image or video is decoded.

``infer``
    Run the checkpoint on exactly those manifested observations.  Model
    predictions necessarily require the policy's camera inputs, so this is the
    only stage that decodes video.  The output contains scalar duration values
    only; frames are never copied or saved.

``analyze``
    Fit predeclared global monotone maps on ``fit``, choose one on ``select``,
    open ``audit`` once, and emit a deployment mapping refit on fit+select.
    The audit fold is never used for model-family selection or refitting.

All folds are sampled from the original training demonstrations.  They measure
post-hoc calibration, not generalization.  Simulator initial states remain
untouched for the predeclared closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from scripts.analysis.duration_calibration_core import (
        FAMILIES,
        FOLDS,
        apply_mapping,
        assign_episode_folds,
        calibration_metrics,
        cap_confusion,
        choose_candidate,
        deterministic_frame_sample,
        fit_cap_threshold,
        fit_mapping,
    )
    from scripts.analysis.duration_training_prior import (
        _episode_task_descriptions,
        _load_action_episodes,
        _task_strings,
        future_windows,
        sha256,
    )
except ModuleNotFoundError:  # flat staging in an isolated cluster directory
    from duration_calibration_core import (  # type: ignore
        FAMILIES,
        FOLDS,
        apply_mapping,
        assign_episode_folds,
        calibration_metrics,
        cap_confusion,
        choose_candidate,
        deterministic_frame_sample,
        fit_cap_threshold,
        fit_mapping,
    )
    from duration_training_prior import (  # type: ignore
        _episode_task_descriptions,
        _load_action_episodes,
        _task_strings,
        future_windows,
        sha256,
    )


PROTOCOL = "libero_duration_head_calibration_train_only_v2"
DEFAULT_SPLIT_SALT = "icra27-duration-calibration-v2-20260821"
SELECTION_RULE = {
    "primary_metric": "draw-level selection-fold MAE",
    "candidate_families": list(FAMILIES),
    "minimum_absolute_mae_gain_steps": 0.25,
    "minimum_relative_mae_gain": 0.05,
    "maximum_event_mae_regression_steps": 0.25,
    "maximum_cap_balanced_accuracy_regression": 0.02,
    "tie_break": "lower MAE then simpler predeclared family order",
    "audit_use": "selected-versus-identity estimate only; never select/refit on audit",
}
EVAL_PRECOMMITMENT = {
    "status": "must_be_frozen_before_any_calibrated_closed_loop_result_is_opened",
    "benchmark": "LIBERO-Long (libero_10)",
    "tasks": list(range(10)),
    "initial_state_ids": list(range(50)),
    "control_frequency_hz": 20,
    "primary_outcome": "paired success difference: calibrated duration minus raw predicted duration",
    "secondary_outcomes": ["environment steps", "policy calls", "executed duration distribution"],
    "required_control": "same checkpoint, shape tokens, initial states, RNG protocol, and cadence",
    "tuning_rule": "no threshold or mapping changes after the first closed-loop calibrated arm starts",
    "scope": "one checkpoint is mechanistic; any paper capability claim requires three training seeds",
    "engineering_note": "task 2 may gate implementation but was selected after prior pilot evidence and is not claim-grade",
}


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math_isfinite(value):
        return None
    return value


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _load_event_module(source: Path):
    source = source.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("duration_calibration_event_targets", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import event source {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("first_event_indices", "first_event_types"):
        if not hasattr(module, name):
            raise AttributeError(f"event target source lacks {name}")
    return module, source


def _model_files(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("model*.safetensors"))
    if not files:
        candidate = checkpoint / "model.safetensors"
        raise FileNotFoundError(candidate)
    return files


def _model_digest(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for path in _model_files(checkpoint):
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _episode_bounds(root: Path) -> dict[int, tuple[int, int]]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError("no episode metadata parquets found")
    result: dict[int, tuple[int, int]] = {}
    for path in paths:
        table = pd.read_parquet(
            path, columns=["episode_index", "dataset_from_index", "dataset_to_index"]
        )
        for row in table.itertuples(index=False):
            episode = int(row.episode_index)
            bounds = (int(row.dataset_from_index), int(row.dataset_to_index))
            if episode in result and result[episode] != bounds:
                raise ValueError(f"conflicting bounds for episode {episode}")
            result[episode] = bounds
    return result


def _event_kwargs(config: dict) -> dict:
    layout = str(config.get("action_layout", "eef7"))
    pose_lo, grip_idx = (5, 11) if layout == "robocasa12" else (0, 6)
    return {
        "pose_lo": pose_lo,
        "grip_idx": grip_idx,
        "min_seg": int(config["min_seg"]),
        "horizon_max": int(config["horizon_max"]),
        "pause_frac": float(config["pause_frac"]),
    }


def _train_dataset_repo(train_config: dict) -> str | None:
    dataset = train_config.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("repo_id"), str):
        return dataset["repo_id"]
    return None


def prepare(args: argparse.Namespace) -> None:
    started = time.time()
    checkpoint = args.ckpt.expanduser().resolve()
    root = args.dataset_root.expanduser().resolve()
    event_module, event_source = _load_event_module(args.event_target_source)
    config_path = checkpoint / "config.json"
    train_config_path = checkpoint / "train_config.json"
    config = json.loads(config_path.read_text())
    if config.get("type") != "smolvla_spline" or not config.get("predict_duration"):
        raise ValueError("checkpoint is not a duration-enabled spline policy")
    if not train_config_path.is_file():
        raise FileNotFoundError(
            "train_config.json is required to bind the calibration dataset repo"
        )
    train_config = json.loads(train_config_path.read_text())
    trained_repo = _train_dataset_repo(train_config)
    if trained_repo != args.dataset_repo_id:
        raise ValueError(
            f"checkpoint trained on repo {trained_repo!r}, requested {args.dataset_repo_id!r}"
        )
    tasks_path, info_path = root / "meta" / "tasks.parquet", root / "meta" / "info.json"
    observed_tasks_hash = sha256(tasks_path)
    if observed_tasks_hash != args.expected_tasks_sha256:
        raise ValueError(
            "task-language metadata does not match the predeclared training dataset: "
            f"observed={observed_tasks_hash}, expected={args.expected_tasks_sha256}. "
            "Do not use a language-augmented overlay for checkpoint calibration."
        )

    episode_tasks = _episode_task_descriptions(root)
    bounds = _episode_bounds(root)
    actions, action_digest, total_frames = _load_action_episodes(root)
    if set(actions) != set(episode_tasks) or set(actions) != set(bounds):
        raise ValueError("action, task, and episode-bound metadata disagree")
    task_strings = _task_strings(tasks_path)
    if len(task_strings) != args.expected_task_count:
        raise ValueError(
            f"expected {args.expected_task_count} tasks, dataset exposes {len(task_strings)}"
        )
    task_to_id = {task: index for index, task in enumerate(task_strings)}
    if len(task_to_id) != len(task_strings):
        raise ValueError("duplicate task descriptions are unsupported")
    if any(task not in task_to_id for task in episode_tasks.values()):
        raise ValueError("episode metadata contains a task absent from tasks.parquet")
    folds = assign_episode_folds(episode_tasks, args.split_salt)
    kwargs = _event_kwargs(config)

    candidates: list[dict] = []
    eligible_counts: Counter[tuple[str, str]] = Counter()
    for ordinal, episode in enumerate(sorted(actions)):
        action = actions[episode]
        start, end = bounds[episode]
        if end - start != len(action):
            raise ValueError(f"episode {episode} bounds/action length mismatch")
        windows, padding = future_windows(action, kwargs["horizon_max"])
        duration = event_module.first_event_indices(windows, padding, **kwargs).tolist()
        kinds = event_module.first_event_types(windows, padding, **kwargs)
        task = episode_tasks[episode]
        fold = folds[episode]
        for frame, (target, kind) in enumerate(zip(duration, kinds)):
            candidates.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "absolute_index": start + frame,
                    "task_id": task_to_id[task],
                    "task_description": task,
                    "fold": fold,
                    "duration_target": int(target),
                    "event_type": str(kind),
                    "pad_count": int(padding[frame].sum().item()),
                }
            )
            eligible_counts[(task, fold)] += 1
        if ordinal % 200 == 0:
            print(f"prepared labels for {ordinal + 1}/{len(actions)} episodes", flush=True)
    selected = deterministic_frame_sample(
        candidates,
        per_task_fold=args.samples_per_task_fold,
        salt=args.frame_salt,
    )
    coordinates = {(row["episode_index"], row["frame_index"]) for row in selected}
    if len(coordinates) != len(selected):
        raise AssertionError("sample contains duplicate coordinates")
    model_digest = _model_digest(checkpoint)
    distribution: dict[str, dict] = {}
    for fold in FOLDS:
        rows = [row for row in selected if row["fold"] == fold]
        distribution[fold] = {
            "n_frames": len(rows),
            "n_episodes": len({row["episode_index"] for row in rows}),
            "target_histogram": dict(sorted(Counter(row["duration_target"] for row in rows).items())),
            "event_type_histogram": dict(sorted(Counter(row["event_type"] for row in rows).items())),
        }
    manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "stage": "locked_train_frame_manifest",
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256(config_path),
        "train_config_sha256": sha256(train_config_path),
        "model_sha256": model_digest,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(root),
        "dataset_revision_hint": root.name,
        "tasks_sha256": observed_tasks_hash,
        "info_sha256": sha256(info_path),
        "action_schedule_sha256": action_digest,
        "dataset_frames": total_frames,
        "dataset_episodes": len(actions),
        "dataset_tasks": len(task_strings),
        "event_target_source": str(event_source),
        "event_target_source_sha256": sha256(event_source),
        "event_parameters": kwargs,
        "split_unit": "whole_episode",
        "split_method": "within-task salted-hash order then round-robin",
        "split_salt": args.split_salt,
        "frame_sampling": "uniform salted coordinate rank within task/fold; labels unused",
        "frame_salt": args.frame_salt,
        "samples_per_task_fold": args.samples_per_task_fold,
        "sample_distribution": distribution,
        "selection_rule": SELECTION_RULE,
        "closed_loop_evaluation_precommitment": EVAL_PRECOMMITMENT,
        "leakage_scope": {
            "source": "training demonstrations only",
            "audit_generalization": False,
            "simulator_initial_states_used": False,
            "audit_fold_used_for_selection": False,
        },
        "rows": selected,
        "script_sha256": sha256(Path(__file__).resolve()),
        "wall_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(args.out, manifest)
    print(json.dumps({"frames": len(selected), "distribution": distribution}, indent=2))


def _verify_manifest(manifest: dict, path: Path) -> None:
    if manifest.get("protocol") != PROTOCOL or manifest.get("stage") != "locked_train_frame_manifest":
        raise ValueError(f"{path} is not a locked v2 calibration manifest")
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError("manifest has no rows")
    episode_fold: dict[int, str] = {}
    coordinates = set()
    for row in rows:
        episode, frame, fold = int(row["episode_index"]), int(row["frame_index"]), str(row["fold"])
        if fold not in FOLDS:
            raise ValueError(f"invalid fold {fold}")
        if episode in episode_fold and episode_fold[episode] != fold:
            raise ValueError(f"episode {episode} leaks across folds")
        episode_fold[episode] = fold
        coordinate = (episode, frame)
        if coordinate in coordinates:
            raise ValueError(f"duplicate coordinate {coordinate}")
        coordinates.add(coordinate)


def _runtime_manifest(policy_source: Path) -> dict:
    cuda = torch.cuda.is_available()
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_device_uuid": (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if cuda
            else None
        ),
        "cudnn": torch.backends.cudnn.version(),
        "policy_source": str(policy_source),
        "policy_source_sha256": sha256(policy_source),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def infer(args: argparse.Namespace) -> None:
    started = time.time()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    _verify_manifest(manifest, manifest_path)
    checkpoint = Path(manifest["checkpoint"]).resolve()
    root = Path(manifest["dataset_root"]).resolve()
    config_path = checkpoint / "config.json"
    if sha256(config_path) != manifest["checkpoint_config_sha256"]:
        raise ValueError("checkpoint config changed after manifest creation")
    if _model_digest(checkpoint) != manifest["model_sha256"]:
        raise ValueError("checkpoint weights changed after manifest creation")
    for relative, key in (("meta/tasks.parquet", "tasks_sha256"), ("meta/info.json", "info_sha256")):
        if sha256(root / relative) != manifest[key]:
            raise ValueError(f"dataset metadata changed: {relative}")
    event_module, event_source = _load_event_module(Path(manifest["event_target_source"]))
    if sha256(event_source) != manifest["event_target_source_sha256"]:
        raise ValueError("event target source changed after manifest creation")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference is required; refusing a silent CPU run")

    # Imports stay inside the video-dependent stage so `prepare` remains a
    # lightweight action-only scanner.
    importlib_import = __import__
    importlib_import("lerobot.policies.smolvla_spline.configuration_smolvla_spline")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_config.pretrained_path = str(checkpoint)
    metadata = LeRobotDatasetMetadata(manifest["dataset_repo_id"], root=root)
    deltas = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(
        manifest["dataset_repo_id"],
        root=root,
        delta_timestamps=deltas,
        download_videos=False,
    )
    expected_offsets = list(range(int(manifest["event_parameters"]["horizon_max"])))
    observed_offsets = list(dataset.reader.delta_indices.get("action", []))
    if observed_offsets != expected_offsets:
        raise ValueError(f"action delta offsets changed: {observed_offsets}")
    rows = manifest["rows"]
    indices = [int(row["absolute_index"]) for row in rows]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    policy_source = Path(inspect.getfile(policy.__class__)).resolve()
    result_rows: list[dict] = []
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch_count = len(batch[ACTION])
            expected = rows[cursor : cursor + batch_count]
            raw_action = torch.as_tensor(batch[ACTION]).cpu()
            raw_pad = torch.as_tensor(batch["action_is_pad"]).bool().cpu()
            target = event_module.first_event_indices(
                raw_action, raw_pad, **manifest["event_parameters"]
            ).tolist()
            kinds = event_module.first_event_types(
                raw_action, raw_pad, **manifest["event_parameters"]
            )
            observed_indices = torch.as_tensor(batch["index"]).reshape(-1).tolist()
            for local, row in enumerate(expected):
                if int(observed_indices[local]) != int(row["absolute_index"]):
                    raise AssertionError("LeRobot absolute-index mapping disagrees with manifest")
                if int(target[local]) != int(row["duration_target"]) or str(kinds[local]) != str(row["event_type"]):
                    raise AssertionError("LeRobot action query disagrees with manifested event target")
            processed = preprocessor(batch)
            images, image_masks = policy.prepare_images(processed)
            state = policy.prepare_state(processed)
            language_tokens = processed[OBS_LANGUAGE_TOKENS]
            language_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
            raw_draws: list[np.ndarray] = []
            production_draws: list[np.ndarray] = []
            for draw in range(args.draws):
                torch.manual_seed(args.seed + batch_index * args.draws + draw)
                torch.cuda.manual_seed_all(args.seed + batch_index * args.draws + draw)
                tokens = policy.model.sample_actions(
                    images, image_masks, language_tokens, language_masks, state
                )
                unnormalized = tokens[:, :, : policy._N_OUT] * policy._tgt_std + policy._tgt_mean
                raw_duration = torch.exp(unnormalized[..., 7].mean(dim=1))
                production = policy._predicted_T_batch(unnormalized)
                raw_draws.append(raw_duration.detach().cpu().numpy())
                production_draws.append(production.detach().cpu().numpy())
            raw_array = np.stack(raw_draws, axis=1)
            production_array = np.stack(production_draws, axis=1)
            for local, row in enumerate(expected):
                result_rows.append(
                    {
                        **row,
                        "raw_duration_draws": [float(value) for value in raw_array[local]],
                        "production_duration_draws": [int(value) for value in production_array[local]],
                    }
                )
            cursor += batch_count
            if batch_index % 10 == 0:
                print(f"inference frames={cursor}/{len(rows)}", flush=True)
    if cursor != len(rows):
        raise AssertionError("inference did not consume the complete manifest")
    output = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "stage": "raw_duration_predictions",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": manifest["checkpoint_config_sha256"],
        "model_sha256": manifest["model_sha256"],
        "dataset_repo_id": manifest["dataset_repo_id"],
        "dataset_root": str(root),
        "tasks_sha256": manifest["tasks_sha256"],
        "event_target_source_sha256": manifest["event_target_source_sha256"],
        "draws_per_frame": args.draws,
        "inference_seed": args.seed,
        "batch_size": args.batch_size,
        "duration_snap_threshold_in_checkpoint": getattr(policy.config, "duration_snap_threshold", None),
        "runtime": _runtime_manifest(policy_source),
        "rows": result_rows,
        "wall_seconds": time.time() - started,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    _atomic_json(args.out, output)
    print(json.dumps({"frames": len(result_rows), "draws": args.draws}, indent=2))


def _fold_arrays(rows: list[dict], fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = [row for row in rows if row["fold"] == fold]
    raw = np.asarray([row["raw_duration_draws"] for row in subset], dtype=np.float64)
    target = np.asarray([row["duration_target"] for row in subset], dtype=np.float64)
    return raw, target, np.asarray(subset, dtype=object)


def _draw_vectors(raw: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return raw.reshape(-1), np.repeat(target, raw.shape[1])


def _paired_bootstrap(
    raw: np.ndarray,
    target: np.ndarray,
    selected_mapping: dict,
    identity_mapping: dict,
    *,
    seed: int = 20260821,
    replicates: int = 5000,
) -> dict:
    identity = apply_mapping(identity_mapping, raw)
    selected = apply_mapping(selected_mapping, raw)
    identity_error = np.mean(np.abs(identity - target[:, None]), axis=1)
    selected_error = np.mean(np.abs(selected - target[:, None]), axis=1)
    improvement = identity_error - selected_error
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(improvement), size=(replicates, len(improvement)))
    bootstrap = improvement[draws].mean(axis=1)
    return {
        "unit": "frame (all stochastic draws kept within frame)",
        "replicates": replicates,
        "mae_improvement_steps": float(improvement.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "probability_improvement_positive": float(np.mean(bootstrap > 0)),
    }


def analyze(args: argparse.Namespace) -> None:
    prediction_path = args.predictions.expanduser().resolve()
    prediction = json.loads(prediction_path.read_text())
    if prediction.get("protocol") != PROTOCOL or prediction.get("stage") != "raw_duration_predictions":
        raise ValueError("input is not a v2 raw prediction artifact")
    manifest_path = Path(prediction["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    _verify_manifest(manifest, manifest_path)
    if sha256(manifest_path) != prediction["manifest_sha256"]:
        raise ValueError("prediction artifact is not bound to the current manifest")
    rows = prediction["rows"]
    if len(rows) != len(manifest["rows"]):
        raise ValueError("prediction and manifest row counts differ")
    for observed, expected in zip(rows, manifest["rows"]):
        fields = ("episode_index", "frame_index", "absolute_index", "fold", "duration_target", "event_type")
        if any(observed[field] != expected[field] for field in fields):
            raise ValueError("prediction rows are not exactly aligned to the manifest")
    minimum = int(manifest["event_parameters"]["min_seg"])
    cap = int(manifest["event_parameters"]["horizon_max"])
    arrays = {fold: _fold_arrays(rows, fold) for fold in FOLDS}
    fit_x, fit_y = _draw_vectors(arrays["fit"][0], arrays["fit"][1])
    select_x, select_y = _draw_vectors(arrays["select"][0], arrays["select"][1])
    audit_x, audit_y = _draw_vectors(arrays["audit"][0], arrays["audit"][1])
    chosen, mappings, selection_scores = choose_candidate(
        fit_x, fit_y, select_x, select_y, minimum=minimum, cap=cap
    )
    selected_fit_mapping = mappings[chosen]
    identity_mapping = mappings["identity"]
    audit_identity = calibration_metrics(
        audit_y, apply_mapping(identity_mapping, audit_x), minimum=minimum, cap=cap
    )
    audit_selected = calibration_metrics(
        audit_y, apply_mapping(selected_fit_mapping, audit_x), minimum=minimum, cap=cap
    )
    # The deployment mapping reuses the already-selected family and fit+select
    # only.  Audit remains untouched by both family selection and parameter fit.
    deployment_mapping = fit_mapping(
        chosen,
        np.concatenate([fit_x, select_x]),
        np.concatenate([fit_y, select_y]),
        minimum=minimum,
        cap=cap,
    )

    fit_cap_threshold_value, fit_cap_score = fit_cap_threshold(fit_x, fit_y, cap=cap)
    cap_threshold_scores = {"fit": fit_cap_score}
    for fold, x, y in (("select", select_x, select_y), ("audit", audit_x, audit_y)):
        classified = np.where(x >= fit_cap_threshold_value, cap, cap - 1)
        cap_threshold_scores[fold] = cap_confusion(y, classified, cap)

    task_audit = {}
    event_audit = {}
    audit_rows = [row for row in rows if row["fold"] == "audit"]
    for field, destination in (("task_description", task_audit), ("event_type", event_audit)):
        values = sorted({str(row[field]) for row in audit_rows})
        for value in values:
            subset = [row for row in audit_rows if str(row[field]) == value]
            raw = np.asarray([row["raw_duration_draws"] for row in subset], dtype=np.float64)
            target = np.asarray([row["duration_target"] for row in subset], dtype=np.float64)
            x, y = _draw_vectors(raw, target)
            destination[value] = {
                "n_frames": len(subset),
                "identity": calibration_metrics(
                    y, apply_mapping(identity_mapping, x), minimum=minimum, cap=cap
                ),
                "selected": calibration_metrics(
                    y, apply_mapping(selected_fit_mapping, x), minimum=minimum, cap=cap
                ),
            }

    distribution = {}
    frame_mean = {}
    for fold in FOLDS:
        raw, target, _ = arrays[fold]
        x, y = _draw_vectors(raw, target)
        identity = apply_mapping(identity_mapping, x)
        distribution[fold] = calibration_metrics(y, identity, minimum=minimum, cap=cap)
        frame_mean[fold] = calibration_metrics(
            target,
            np.clip(np.rint(raw.mean(axis=1)), minimum, cap),
            minimum=minimum,
            cap=cap,
        )
        frame_mean[fold]["mean_within_frame_raw_std"] = float(np.mean(np.std(raw, axis=1)))

    calibrator = {
        "schema_version": 1,
        "protocol": "duration_decode_calibrator_v1",
        "mapping": deployment_mapping,
        "selected_family": chosen,
        "fit_folds": ["fit", "select"],
        "excluded_fold": "audit",
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": sha256(prediction_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "checkpoint_config_sha256": manifest["checkpoint_config_sha256"],
        "model_sha256": manifest["model_sha256"],
        "dataset_action_schedule_sha256": manifest["action_schedule_sha256"],
        "dataset_tasks_sha256": manifest["tasks_sha256"],
        "event_target_source_sha256": manifest["event_target_source_sha256"],
        "selection_rule": SELECTION_RULE,
        "shape_channels_modified": False,
        "closed_loop_evaluation_precommitment": manifest["closed_loop_evaluation_precommitment"],
    }
    _atomic_json(args.calibrator_out, calibrator)
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "stage": "calibration_analysis",
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "selected_family": chosen,
        "selection_rule": SELECTION_RULE,
        "selection_scores": selection_scores,
        "fit_only_candidate_parameters": mappings,
        "audit_opened_once": {
            "identity": audit_identity,
            "selected": audit_selected,
            "paired_cluster_bootstrap": _paired_bootstrap(
                arrays["audit"][0],
                arrays["audit"][1],
                selected_fit_mapping,
                identity_mapping,
            ),
        },
        "raw_identity_distribution_by_fold": distribution,
        "frame_mean_identity_by_fold": frame_mean,
        "cap_event_raw_threshold": {
            "threshold_fit_on_fit_fold": fit_cap_threshold_value,
            "scores": cap_threshold_scores,
        },
        "audit_by_event_type": event_audit,
        "audit_by_task": task_audit,
        "deployment_calibrator": str(args.calibrator_out.resolve()),
        "deployment_calibrator_sha256": sha256(args.calibrator_out.resolve()),
        "leakage_statement": (
            "Every parameter came from original training demonstrations. The audit fold was "
            "excluded from family selection and refit. These are calibration diagnostics, not "
            "held-out generalization results; closed-loop simulator states remain untouched."
        ),
        "closed_loop_evaluation_precommitment": manifest["closed_loop_evaluation_precommitment"],
    }
    _atomic_json(args.out, report)
    print(
        json.dumps(
            {
                "selected_family": chosen,
                "audit_identity_mae": audit_identity["mae"],
                "audit_selected_mae": audit_selected["mae"],
                "calibrator": str(args.calibrator_out),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="action-only locked manifest")
    prep.add_argument("--ckpt", type=Path, required=True)
    prep.add_argument("--dataset_root", type=Path, required=True)
    prep.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    prep.add_argument("--expected_tasks_sha256", required=True)
    prep.add_argument("--expected_task_count", type=int, default=40)
    prep.add_argument("--event_target_source", type=Path, required=True)
    prep.add_argument("--samples_per_task_fold", type=int, default=32)
    prep.add_argument("--split_salt", default=DEFAULT_SPLIT_SALT)
    prep.add_argument("--frame_salt", default=DEFAULT_SPLIT_SALT + "-frames")
    prep.add_argument("--out", type=Path, required=True)
    prep.set_defaults(func=prepare)

    inference = subparsers.add_parser("infer", help="selected-frame GPU inference")
    inference.add_argument("--manifest", type=Path, required=True)
    inference.add_argument("--draws", type=int, default=4)
    inference.add_argument("--batch_size", type=int, default=16)
    inference.add_argument("--num_workers", type=int, default=8)
    inference.add_argument("--seed", type=int, default=1701)
    inference.add_argument("--out", type=Path, required=True)
    inference.set_defaults(func=infer)

    analysis = subparsers.add_parser("analyze", help="fit/select/audit mappings")
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--calibrator_out", type=Path, required=True)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "draws", 1) <= 0:
        raise ValueError("draws must be positive")
    if getattr(args, "batch_size", 1) <= 0 or getattr(args, "num_workers", 0) < 0:
        raise ValueError("invalid data-loader settings")
    args.func(args)


if __name__ == "__main__":
    main()

