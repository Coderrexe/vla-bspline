"""Analyze the locked A/C x language LIBERO-Long factorial.

Inputs are explicit ``NAME=PATH`` bindings rather than output-directory globs,
so reruns cannot silently enter the paper table. Required canonical names are
``Abase_s1000``, ``Alang_s1000``, ``Cbase_s1000``, ``Clang_s1000`` and the same
for seeds 1001 and 1002. Required cadence-matched names are
``Cbase_s{seed}_nas10`` and ``Clang_s{seed}_nas10`` for all three seeds.
The unsuffixed C cells are deployment-cadence (five actions/replan) results;
the matched C ``nas10`` cells are required for the cadence-matched primary
interaction with A (ten actions/replan). There are 18 required evaluations.

The analysis fails closed on evaluation provenance and, by default, training
data provenance. Pass one ``--train_manifest NAME=PATH`` for every canonical
cell. A manifest has the schema written by
``cluster/train_language_factorial.sbatch``. The escape hatch
``--allow_unverified_training_data`` is exploratory only. Claim-bearing output
also requires one ``--runtime_manifest NAME=PATH`` per evaluation, binding its
Slurm job to GPU, CUDA, cuDNN, and Torch identities, plus one immutable
``--evaluation_manifest NAME=PATH`` per evaluation to bind source config,
config-only cadence variant, and result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEEDS = (1000, 1001, 1002)
HEADS = ("A", "C")
LANGS = ("base", "lang")
N_TASKS = 10
STATES_PER_TASK = 50
EXPECTED_ACTION_STEPS = {"A": 10, "C": 5}
EXPLORATORY_REGRESSION_MARGIN_PP = 5.0


def canonical_names() -> set[str]:
    return {f"{head}{lang}_s{seed}" for head in HEADS for lang in LANGS for seed in SEEDS}


def cadence_matched_c_names() -> set[str]:
    return {f"C{lang}_s{seed}_nas10" for lang in LANGS for seed in SEEDS}


def _parse_canonical_name(name: str) -> tuple[str, str, int]:
    for head in HEADS:
        for lang in LANGS:
            prefix = f"{head}{lang}_s"
            if name.startswith(prefix):
                raw_seed = name[len(prefix) :]
                if raw_seed.isdigit() and int(raw_seed) in SEEDS:
                    return head, lang, int(raw_seed)
    raise ValueError(f"not a canonical cell name: {name}")


def _require_fields(payload: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(payload))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cell(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    validate_cell_integrity(data, str(path))
    return data


def validate_cell_integrity(cell: dict[str, Any], label: str) -> None:
    """Validate one artifact without relying on any other factorial cell."""

    required = {
        "schema_version", "protocol", "checkpoint_config_sha256", "model_sha256",
        "policy_type", "n_action_steps", "suite", "control_frequency_hz", "state_ids",
        "seed_base", "n_tasks", "n_episodes", "n_successes", "success_rate",
        "policy_source_sha256", "env_source_sha256", "lerobot_commit", "stats_sha256",
        "evaluator_sha256", "slurm_job_id", "tasks",
    }
    _require_fields(cell, required, label)
    if cell["schema_version"] != 1 or cell["protocol"] != "libero_explicit_init_state_v1":
        raise ValueError(f"{label} is not a locked-eval v1 artifact")
    if cell["n_tasks"] != N_TASKS or cell["n_episodes"] != N_TASKS * STATES_PER_TASK:
        raise ValueError(f"{label} must contain 10 tasks and 500 episodes")
    if cell["state_ids"] != list(range(STATES_PER_TASK)):
        raise ValueError(f"{label} must evaluate exact state ids 0..49")
    if cell["suite"] != "libero_10" or cell["control_frequency_hz"] != 20:
        raise ValueError(f"{label} has unexpected suite or control frequency")
    for field in (
        "checkpoint_config_sha256", "model_sha256", "policy_source_sha256",
        "env_source_sha256", "evaluator_sha256",
    ):
        if not _is_sha256(cell[field]):
            raise ValueError(f"{label}.{field} is not a SHA-256 digest")
    if not isinstance(cell["lerobot_commit"], str) or not cell["lerobot_commit"]:
        raise ValueError(f"{label}.lerobot_commit is absent")
    if not isinstance(cell["slurm_job_id"], str) or not cell["slurm_job_id"]:
        raise ValueError(f"{label}.slurm_job_id is absent")
    if len(cell["tasks"]) != N_TASKS:
        raise ValueError(f"{label} has {len(cell['tasks'])} task records, expected 10")

    mapping = episode_map(cell)
    expected_grid = {(task, state) for task in range(N_TASKS) for state in range(STATES_PER_TASK)}
    if set(mapping) != expected_grid:
        missing = sorted(expected_grid - set(mapping))
        extra = sorted(set(mapping) - expected_grid)
        raise ValueError(f"{label} has a noncanonical task/state grid: missing={missing}, extra={extra}")

    task_ids: set[int] = set()
    counted_successes = 0
    for task in cell["tasks"]:
        task_id = int(task["task_id"])
        if task_id in task_ids:
            raise ValueError(f"{label} has duplicate task id {task_id}")
        task_ids.add(task_id)
        if not isinstance(task.get("task_description"), str) or not task["task_description"]:
            raise ValueError(f"{label} task {task_id} has no task description")
        if len(task.get("episodes", [])) != STATES_PER_TASK:
            raise ValueError(f"{label} task {task_id} does not have 50 episodes")
        task_successes = 0
        for episode in task["episodes"]:
            state_id = int(episode["state_id"])
            expected_seed = int(cell["seed_base"]) + task_id * 1_000 + state_id
            if int(episode.get("seed", -1)) != expected_seed:
                raise ValueError(
                    f"{label} task/state {(task_id, state_id)} has seed "
                    f"{episode.get('seed')}, expected {expected_seed}"
                )
            if not isinstance(episode.get("success"), bool):
                raise ValueError(f"{label} task/state {(task_id, state_id)} success is not bool")
            task_successes += int(episode["success"])
        counted_successes += task_successes
        if "success_rate" in task and not math.isclose(
            float(task["success_rate"]), task_successes / STATES_PER_TASK, abs_tol=1e-12
        ):
            raise ValueError(f"{label} task {task_id} success_rate is inconsistent with episodes")
    if task_ids != set(range(N_TASKS)):
        raise ValueError(f"{label} has task ids {sorted(task_ids)}, expected 0..9")
    if int(cell["n_successes"]) != counted_successes:
        raise ValueError(f"{label}.n_successes is inconsistent with episodes")
    if not math.isclose(
        float(cell["success_rate"]), counted_successes / (N_TASKS * STATES_PER_TASK), abs_tol=1e-12
    ):
        raise ValueError(f"{label}.success_rate is inconsistent with episodes")


def episode_map(cell: dict[str, Any]) -> dict[tuple[int, int], bool]:
    result: dict[tuple[int, int], bool] = {}
    for task in cell["tasks"]:
        task_id = int(task["task_id"])
        for episode in task["episodes"]:
            key = (task_id, int(episode["state_id"]))
            if key in result:
                raise ValueError(f"duplicate task/state key: {key}")
            result[key] = bool(episode["success"])
    return result


def validate_evaluation_provenance(cells: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Fail closed on pairing, evaluator identity, model identity, and cadence."""

    required = canonical_names()
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"missing required cells: {missing}")
    cadence_names = cadence_matched_c_names()
    missing_cadence = sorted(cadence_names - set(cells))
    if missing_cadence:
        raise ValueError(f"missing required cadence-matched C cells: {missing_cadence}")
    unexpected = sorted(set(cells) - required - cadence_names)
    if unexpected:
        raise ValueError(f"unexpected cells: {unexpected}")

    for name, cell in cells.items():
        validate_cell_integrity(cell, name)

    global_equal = (
        "protocol", "suite", "control_frequency_hz", "state_ids", "seed_base", "n_tasks",
        "n_episodes", "evaluator_sha256", "env_source_sha256", "lerobot_commit",
    )
    reference_name = sorted(required)[0]
    reference = cells[reference_name]
    for field in global_equal:
        mismatched = sorted(name for name, cell in cells.items() if cell[field] != reference[field])
        if mismatched:
            raise ValueError(f"evaluation provenance mismatch for {field}: {mismatched}")
    reference_tasks = {
        int(task["task_id"]): str(task["task_description"]) for task in reference["tasks"]
    }
    for name, cell in cells.items():
        task_descriptions = {
            int(task["task_id"]): str(task["task_description"]) for task in cell["tasks"]
        }
        if task_descriptions != reference_tasks:
            raise ValueError(f"evaluation task descriptions differ for {name}")

    policy_types: dict[str, str] = {}
    for head in HEADS:
        head_names = sorted(name for name in required if name.startswith(head))
        head_reference = cells[head_names[0]]
        policy_types[head] = str(head_reference["policy_type"])
        for field in (
            "policy_type",
            "policy_source_sha256",
            "stats_sha256",
            "checkpoint_config_sha256",
        ):
            mismatched = [name for name in head_names if cells[name][field] != head_reference[field]]
            if mismatched:
                raise ValueError(f"{head}-head provenance mismatch for {field}: {mismatched}")
        for name in head_names:
            if int(cells[name]["n_action_steps"]) != EXPECTED_ACTION_STEPS[head]:
                raise ValueError(
                    f"{name} cadence is {cells[name]['n_action_steps']}, "
                    f"expected {EXPECTED_ACTION_STEPS[head]}"
                )
    if policy_types["A"] == policy_types["C"]:
        raise ValueError("A and C unexpectedly have the same policy type")
    a_reference = cells[sorted(name for name in required if name.startswith("A"))[0]]
    if a_reference["stats_sha256"] is not None:
        raise ValueError("A head unexpectedly uses spline statistics")
    c_reference = cells[sorted(name for name in required if name.startswith("C"))[0]]
    if not _is_sha256(c_reference["stats_sha256"]):
        raise ValueError("C head is missing a valid spline-statistics hash")
    if a_reference["checkpoint_config_sha256"] == c_reference["checkpoint_config_sha256"]:
        raise ValueError("A and C unexpectedly have the same checkpoint config")

    model_hashes = [str(cells[name]["model_sha256"]) for name in sorted(required)]
    if len(set(model_hashes)) != len(model_hashes):
        duplicates = sorted({digest for digest in model_hashes if model_hashes.count(digest) > 1})
        raise ValueError(f"canonical factorial cells reuse model weights: {duplicates}")

    nas10_config_hashes: set[str] = set()
    for lang in LANGS:
        for seed in SEEDS:
            canonical = cells[f"C{lang}_s{seed}"]
            control = cells[f"C{lang}_s{seed}_nas10"]
            if control["model_sha256"] != canonical["model_sha256"]:
                raise ValueError(f"C{lang} seed {seed} cadence control changed model weights")
            if control["checkpoint_config_sha256"] == canonical["checkpoint_config_sha256"]:
                raise ValueError(
                    f"C{lang} seed {seed} cadence control did not change the checkpoint config"
                )
            if int(control["n_action_steps"]) != 10:
                raise ValueError(f"C{lang} seed {seed} cadence control is not n_action_steps=10")
            for field in ("policy_type", "policy_source_sha256", "stats_sha256"):
                if control[field] != canonical[field]:
                    raise ValueError(f"C{lang} seed {seed} cadence control changed {field}")
            nas10_config_hashes.add(str(control["checkpoint_config_sha256"]))
    if len(nas10_config_hashes) != 1:
        raise ValueError(
            "cadence-matched C cells do not share one n_action_steps=10 checkpoint config"
        )

    return {
        "status": "validated", "evaluator_sha256": reference["evaluator_sha256"],
        "env_source_sha256": reference["env_source_sha256"],
        "lerobot_commit": reference["lerobot_commit"], "policy_types": policy_types,
        "canonical_models_unique": True,
        "cadence_matched_c_cells": "validated_all_three_seeds",
    }


def load_training_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_evaluation_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    manifest["_artifact_sha256"] = _sha256_file(path)
    return manifest


def validate_evaluation_manifests(
    cells: Mapping[str, dict[str, Any]], manifests: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bind source checkpoint -> config-only variant -> locked result."""

    required = canonical_names() | cadence_matched_c_names()
    missing = sorted(required - set(manifests))
    extra = sorted(set(manifests) - required)
    if missing or extra:
        raise ValueError(
            f"evaluation manifests must match all 18 eval cells; missing={missing}, extra={extra}"
        )
    fields = {
        "schema_version",
        "source_checkpoint",
        "variant_checkpoint",
        "source_config_sha256",
        "variant_config_sha256",
        "model_sha256",
        "original_n_action_steps",
        "evaluated_n_action_steps",
        "policy_type",
        "suite",
        "state_start",
        "states_per_task",
        "control_frequency_hz",
        "initial_states",
        "slurm_job_id",
        "source_config",
        "variant_config",
        "_artifact_sha256",
    }
    source_bindings: dict[str, tuple[str, str]] = {}
    for name in sorted(required):
        cell = cells[name]
        manifest = manifests[name]
        _require_fields(manifest, fields, f"evaluation manifest {name}")
        if manifest["schema_version"] != 2:
            raise ValueError(f"evaluation manifest {name} is not provenance schema v2")
        for field in (
            "source_config_sha256",
            "variant_config_sha256",
            "model_sha256",
            "_artifact_sha256",
        ):
            if not _is_sha256(manifest[field]):
                raise ValueError(f"evaluation manifest {name}.{field} is not a SHA-256 digest")
        if cell.get("evaluation_manifest_sha256") != manifest["_artifact_sha256"]:
            raise ValueError(f"eval result {name} does not bind the supplied evaluation manifest")
        if manifest["variant_config_sha256"] != cell["checkpoint_config_sha256"]:
            raise ValueError(f"evaluation manifest {name} does not bind the variant config")
        if manifest["model_sha256"] != cell["model_sha256"]:
            raise ValueError(f"evaluation manifest {name} does not bind the evaluated model")
        if str(manifest["variant_checkpoint"]) != str(cell.get("checkpoint")):
            raise ValueError(f"evaluation manifest {name} does not bind the variant checkpoint path")
        if str(manifest["slurm_job_id"]) != str(cell["slurm_job_id"]):
            raise ValueError(f"evaluation manifest {name} does not bind the eval Slurm job")
        if manifest["policy_type"] != cell["policy_type"]:
            raise ValueError(f"evaluation manifest {name} has the wrong policy type")
        if (
            manifest["suite"] != cell["suite"]
            or int(manifest["control_frequency_hz"]) != int(cell["control_frequency_hz"])
            or int(manifest["state_start"]) != 0
            or int(manifest["states_per_task"]) != STATES_PER_TASK
            or manifest["initial_states"] != list(range(STATES_PER_TASK))
        ):
            raise ValueError(f"evaluation manifest {name} has the wrong eval schedule")
        if int(manifest["evaluated_n_action_steps"]) != int(cell["n_action_steps"]):
            raise ValueError(f"evaluation manifest {name} has the wrong evaluated cadence")

        source_config = manifest["source_config"]
        variant_config = manifest["variant_config"]
        if not isinstance(source_config, dict) or not isinstance(variant_config, dict):
            raise ValueError(f"evaluation manifest {name} lacks semantic config payloads")
        if source_config.get("n_action_steps") != manifest["original_n_action_steps"]:
            raise ValueError(f"evaluation manifest {name} original cadence is inconsistent")
        if variant_config.get("n_action_steps") != manifest["evaluated_n_action_steps"]:
            raise ValueError(f"evaluation manifest {name} variant cadence is inconsistent")
        source_without_cadence = dict(source_config)
        variant_without_cadence = dict(variant_config)
        source_without_cadence.pop("n_action_steps", None)
        variant_without_cadence.pop("n_action_steps", None)
        if source_without_cadence != variant_without_cadence:
            differing_keys = sorted(
                key
                for key in set(source_without_cadence) | set(variant_without_cadence)
                if source_without_cadence.get(key) != variant_without_cadence.get(key)
            )
            raise ValueError(
                f"evaluation manifest {name} variant changes fields beyond n_action_steps: "
                f"{differing_keys}"
            )
        source_bindings[name] = (
            str(manifest["source_config_sha256"]),
            str(manifest["source_checkpoint"]),
        )

    # The two C cadence evaluations for each trained checkpoint must share one
    # immutable source checkpoint/config, not merely equal model bytes.
    for lang in LANGS:
        for seed in SEEDS:
            native = f"C{lang}_s{seed}"
            matched = f"{native}_nas10"
            if source_bindings[native] != source_bindings[matched]:
                raise ValueError(
                    f"{native} NAS5/NAS10 evaluations do not share one source checkpoint/config"
                )
    return {
        "status": "validated",
        "n_bound_manifests": len(required),
        "allowed_variant_config_delta": ["n_action_steps"],
    }


def validate_training_provenance(
    cells: Mapping[str, dict[str, Any]],
    manifests: Mapping[str, dict[str, Any]],
    evaluation_manifests: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate checkpoint treatment/seed and data identity across the factorial."""

    required = canonical_names()
    missing = sorted(required - set(manifests))
    extra = sorted(set(manifests) - required)
    if missing or extra:
        raise ValueError(f"training manifests must match canonical cells; missing={missing}, extra={extra}")
    fields = {
        "arm", "seed", "dataset_info_sha256", "dataset_tasks_sha256", "policy_type",
        "n_action_steps", "lerobot_commit", "training_source_hashes", "model_sha256",
        "final_policy_config_sha256", "train_config_sha256", "saved_train_seed",
        "dataset_content_hashes",
    }
    dataset_ids: dict[str, set[tuple[str, str]]] = {lang: set() for lang in LANGS}
    commits: dict[str, set[str]] = {head: set() for head in HEADS}
    source_hash_sets: dict[str, set[str]] = {head: set() for head in HEADS}
    dataset_content_sets: dict[str, set[str]] = {lang: set() for lang in LANGS}
    train_config_hashes: list[str] = []
    for name in sorted(required):
        manifest = manifests[name]
        _require_fields(manifest, fields, f"training manifest {name}")
        head, lang, seed = _parse_canonical_name(name)
        if manifest["arm"] != head or int(manifest["seed"]) != seed:
            raise ValueError(f"training manifest {name} has wrong arm or seed")
        if manifest["policy_type"] != cells[name]["policy_type"]:
            raise ValueError(f"training manifest {name} has wrong policy type")
        if int(manifest["n_action_steps"]) != EXPECTED_ACTION_STEPS[head]:
            raise ValueError(f"training manifest {name} has wrong training cadence")
        if int(manifest["saved_train_seed"]) != seed:
            raise ValueError(f"training manifest {name} saved train-config seed is wrong")
        if manifest["model_sha256"] != cells[name]["model_sha256"]:
            raise ValueError(f"training manifest {name} does not bind the evaluated model weights")
        evaluation_manifest = evaluation_manifests.get(name)
        if evaluation_manifest is None:
            raise ValueError(f"training manifest {name} has no paired evaluation manifest")
        if (
            manifest["final_policy_config_sha256"]
            != evaluation_manifest["source_config_sha256"]
        ):
            raise ValueError(
                f"training manifest {name} does not bind the evaluation source config"
            )
        if not _is_sha256(manifest["train_config_sha256"]):
            raise ValueError(f"training manifest {name}.train_config_sha256 is invalid")
        train_config_hashes.append(str(manifest["train_config_sha256"]))
        for field in ("dataset_info_sha256", "dataset_tasks_sha256"):
            if not _is_sha256(manifest[field]):
                raise ValueError(f"training manifest {name}.{field} is not a SHA-256 digest")
        dataset_ids[lang].add(
            (str(manifest["dataset_info_sha256"]), str(manifest["dataset_tasks_sha256"]))
        )
        content_hashes = manifest["dataset_content_hashes"]
        required_content_hashes = {
            "source_dataset_identity_sha256",
            "non_language_parquet_sha256",
            "task_index_schedule_sha256",
            "episode_metadata_sha256",
            "video_content_manifest_sha256",
        }
        if not isinstance(content_hashes, dict):
            raise ValueError(f"training manifest {name} has no dataset-content hash map")
        _require_fields(
            content_hashes, required_content_hashes, f"training manifest {name} dataset content"
        )
        if set(content_hashes) != required_content_hashes:
            raise ValueError(f"training manifest {name} has unexpected dataset-content hashes")
        if any(not _is_sha256(value) for value in content_hashes.values()):
            raise ValueError(f"training manifest {name} has an invalid dataset-content hash")
        dataset_content_sets[lang].add(json.dumps(content_hashes, sort_keys=True))
        if not isinstance(manifest["lerobot_commit"], str) or not manifest["lerobot_commit"]:
            raise ValueError(f"training manifest {name} has no source commit")
        commits[head].add(str(manifest["lerobot_commit"]))
        source_hashes = manifest["training_source_hashes"]
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError(f"training manifest {name} has no source-file hash map")
        for source_name, source_hash in source_hashes.items():
            if not isinstance(source_name, str) or not source_name or not _is_sha256(source_hash):
                raise ValueError(f"training manifest {name} has an invalid source-file hash map")
        if source_hashes.get("policy_source") != cells[name]["policy_source_sha256"]:
            raise ValueError(
                f"training/evaluation policy source differs for {name}; "
                "the checkpoint cannot be interpreted under mutated policy code"
            )
        # Canonical serialization lets path/hash maps be compared without relying
        # on a git commit. This is essential because the spline policy directory
        # was historically untracked and could change under the same HEAD.
        source_hash_sets[head].add(json.dumps(source_hashes, sort_keys=True))
        if head == "C":
            if not _is_sha256(manifest.get("spline_stats_sha256")):
                raise ValueError(f"training manifest {name} has no spline-statistics hash")
            if manifest["spline_stats_sha256"] != cells[name]["stats_sha256"]:
                raise ValueError(f"training/evaluation spline statistics differ for {name}")

    for lang, identities in dataset_ids.items():
        if len(identities) != 1:
            raise ValueError(f"{lang} cells do not share one dataset identity: {sorted(identities)}")
    base_identity = next(iter(dataset_ids["base"]))
    lang_identity = next(iter(dataset_ids["lang"]))
    if base_identity == lang_identity or base_identity[1] == lang_identity[1]:
        raise ValueError("base and language treatments do not have distinct task-table identities")
    for lang, identities in dataset_content_sets.items():
        if len(identities) != 1:
            raise ValueError(
                f"{lang} cells do not share one full dataset-content identity"
            )
    base_content = json.loads(next(iter(dataset_content_sets["base"])))
    lang_content = json.loads(next(iter(dataset_content_sets["lang"])))
    invariant_content_fields = (
        "source_dataset_identity_sha256",
        "non_language_parquet_sha256",
        "episode_metadata_sha256",
        "video_content_manifest_sha256",
    )
    changed_non_language = [
        field
        for field in invariant_content_fields
        if base_content[field] != lang_content[field]
    ]
    if changed_non_language:
        raise ValueError(
            "base/language datasets differ outside the language task-index overlay: "
            f"{changed_non_language}"
        )
    if (
        base_content["task_index_schedule_sha256"]
        == lang_content["task_index_schedule_sha256"]
    ):
        raise ValueError("base/language datasets have the same task-index schedule")
    for head, identities in commits.items():
        if len(identities) != 1:
            raise ValueError(f"{head} base/language training source commits differ: {sorted(identities)}")
    for head, identities in source_hash_sets.items():
        if len(identities) != 1:
            raise ValueError(
                f"{head} base/language training source-file hashes differ; "
                "the head x language effect is code-confounded"
            )
    if len(set(train_config_hashes)) != len(train_config_hashes):
        raise ValueError("canonical cells unexpectedly reuse a saved training-config artifact")
    return {
        "status": "validated",
        "dataset_identity_by_treatment": {
            "base": {
                "info_sha256": base_identity[0],
                "tasks_sha256": base_identity[1],
                "content_hashes": base_content,
            },
            "lang": {
                "info_sha256": lang_identity[0],
                "tasks_sha256": lang_identity[1],
                "content_hashes": lang_content,
            },
        },
        "training_commit_by_head": {head: next(iter(values)) for head, values in commits.items()},
        "training_source_hashes_by_head": {
            head: json.loads(next(iter(values))) for head, values in source_hash_sets.items()
        },
    }


def validate_runtime_provenance(
    cells: Mapping[str, dict[str, Any]], manifests: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bind every eval artifact to one verified software/hardware runtime."""

    required = canonical_names() | cadence_matched_c_names()
    missing = sorted(required - set(manifests))
    extra = sorted(set(manifests) - required)
    if missing or extra:
        raise ValueError(f"runtime manifests must match all 18 eval cells; missing={missing}, extra={extra}")
    fields = {
        "slurm_job_id",
        "gpu_model",
        "cuda_version",
        "cudnn_version",
        "torch_version",
    }
    runtime_identities: set[tuple[str, str, str, str]] = set()
    for name in sorted(required):
        manifest = manifests[name]
        _require_fields(manifest, fields, f"runtime manifest {name}")
        if str(manifest["slurm_job_id"]) != str(cells[name]["slurm_job_id"]):
            raise ValueError(f"runtime manifest {name} does not bind the eval Slurm job")
        values = tuple(
            str(manifest[field])
            for field in ("gpu_model", "cuda_version", "cudnn_version", "torch_version")
        )
        if any(not value for value in values):
            raise ValueError(f"runtime manifest {name} has an empty hardware/software field")
        runtime_identities.add(values)
    if len(runtime_identities) != 1:
        raise ValueError(
            "evaluation hardware/software identities differ across factorial cells: "
            f"{sorted(runtime_identities)}"
        )
    identity = next(iter(runtime_identities))
    return {
        "status": "validated",
        "gpu_model": identity[0],
        "cuda_version": identity[1],
        "cudnn_version": identity[2],
        "torch_version": identity[3],
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= positive total")
    proportion = successes / total
    denom = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denom
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denom
    return center - radius, center + radius


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    if left.shape != right.shape:
        raise ValueError("paired arrays differ in shape")
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {"left_only": left_only, "right_only": right_only, "discordant": discordant, "p_value": p_value}


def holm(p_values: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def ordered_arrays(
    left: dict[tuple[int, int], bool], right: dict[tuple[int, int], bool]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"state manifests differ: missing_left={missing_left}, missing_right={missing_right}"
        )
    expected = {(task, state) for task in range(N_TASKS) for state in range(STATES_PER_TASK)}
    if set(left) != expected:
        raise ValueError("paired comparison requires the exact 10 x 50 task/state grid")
    keys = sorted(left)
    return keys, np.array([left[k] for k in keys]), np.array([right[k] for k in keys])


def comparison(left_cell: dict[str, Any], right_cell: dict[str, Any]) -> dict[str, Any]:
    keys, left, right = ordered_arrays(episode_map(left_cell), episode_map(right_cell))
    result: dict[str, Any] = {
        "left_rate": float(left.mean()), "right_rate": float(right.mean()),
        "delta_pp": float(100 * (right.mean() - left.mean())),
        "overall_mcnemar": exact_mcnemar(left, right), "per_task": [],
    }
    task_p_values = []
    for task_id in range(N_TASKS):
        mask = np.array([key[0] == task_id for key in keys])
        test = exact_mcnemar(left[mask], right[mask])
        row = {
            "task_id": task_id, "left_rate": float(left[mask].mean()),
            "right_rate": float(right[mask].mean()),
            "delta_pp": float(100 * (right[mask].mean() - left[mask].mean())), **test,
        }
        result["per_task"].append(row)
        task_p_values.append(float(test["p_value"]))
    for row, corrected in zip(result["per_task"], holm(task_p_values), strict=True):
        row["holm_within_cell_p_value"] = corrected
    return result


def to_cube(cells: Mapping[str, dict[str, Any]], *, c_n_action_steps: int = 10) -> np.ndarray:
    """Create a paired factorial cube at A@10 and the requested C cadence."""

    if c_n_action_steps not in (5, 10):
        raise ValueError("C cube cadence must be 5 or 10 actions/replan")
    cube = np.empty((2, 2, 3, N_TASKS, STATES_PER_TASK), dtype=float)
    expected = {(task, state) for task in range(N_TASKS) for state in range(STATES_PER_TASK)}
    for head_ix, head in enumerate(HEADS):
        for lang_ix, lang in enumerate(LANGS):
            for seed_ix, seed in enumerate(SEEDS):
                key = f"{head}{lang}_s{seed}"
                if head == "C" and c_n_action_steps == 10:
                    key += "_nas10"
                mapping = episode_map(cells[key])
                if set(mapping) != expected:
                    raise ValueError(f"{key} does not contain exact task/state grid 10 x 0..49")
                cube[head_ix, lang_ix, seed_ix] = [
                    [mapping[(task, state)] for state in range(STATES_PER_TASK)]
                    for task in range(N_TASKS)
                ]
    return cube


def hierarchical_interaction_ci(
    cube: np.ndarray, samples: int = 20_000, seed: int = 20260821
) -> tuple[float, float]:
    expected_shape = (2, 2, len(SEEDS), N_TASKS, STATES_PER_TASK)
    if cube.shape != expected_shape or samples <= 0:
        raise ValueError(f"expected cube shape {expected_shape} and positive samples")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for sample in range(samples):
        seed_ids = rng.integers(0, len(SEEDS), len(SEEDS))
        task_ids = rng.integers(0, N_TASKS, N_TASKS)
        values = []
        for seed_id in seed_ids:
            for task_id in task_ids:
                # Shared indices preserve pairing among all four factorial cells.
                state_ids = rng.integers(0, STATES_PER_TASK, STATES_PER_TASK)
                block = cube[:, :, seed_id, task_id][:, :, state_ids]
                delta_a = block[0, 1].mean() - block[0, 0].mean()
                delta_c = block[1, 1].mean() - block[1, 0].mean()
                values.append(delta_c - delta_a)
        estimates[sample] = np.mean(values)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(100 * low), float(100 * high)


def seed_t_interval(values: np.ndarray) -> tuple[float, float]:
    """Classical seed-level mean t interval (n=3, df=2), in percentage points."""
    if values.shape != (len(SEEDS),):
        raise ValueError(f"expected one value for each of {len(SEEDS)} training seeds")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical_t_df2 = 4.302652729696142
    return mean - critical_t_df2 * standard_error, mean + critical_t_df2 * standard_error


def analyze(
    cells: Mapping[str, dict[str, Any]],
    training_manifests: Mapping[str, dict[str, Any]] | None,
    evaluation_manifests: Mapping[str, dict[str, Any]] | None = None,
    runtime_manifests: Mapping[str, dict[str, Any]] | None = None,
    *,
    allow_unverified_training_data: bool = False,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    eval_provenance = validate_evaluation_provenance(cells)
    if evaluation_manifests is None:
        if not allow_unverified_training_data:
            raise ValueError(
                "evaluation manifests are required to bind source and variant configs"
            )
        evaluation_manifest_provenance = {
            "status": "UNVERIFIED_EXPLORATORY_ONLY",
            "warning": "source/variant evaluation manifests were not supplied",
        }
    else:
        evaluation_manifest_provenance = validate_evaluation_manifests(
            cells, evaluation_manifests
        )
    if training_manifests is None:
        if not allow_unverified_training_data:
            raise ValueError(
                "training manifests are required for a claim-bearing analysis; "
                "pass --allow_unverified_training_data only for exploratory diagnosis"
            )
        training_provenance = {
            "status": "UNVERIFIED_EXPLORATORY_ONLY",
            "warning": "training dataset/checkpoint provenance was not supplied",
        }
    else:
        if evaluation_manifests is None:
            if not allow_unverified_training_data:
                raise ValueError("training provenance cannot bind without evaluation manifests")
            training_provenance = {
                "status": "UNVERIFIED_EXPLORATORY_ONLY",
                "warning": "training manifests supplied without source/variant binding manifests",
            }
        else:
            training_provenance = validate_training_provenance(
                cells, training_manifests, evaluation_manifests
            )
    if runtime_manifests is None:
        if not allow_unverified_training_data:
            raise ValueError(
                "runtime manifests are required to rule out hardware/software confounding; "
                "the exploratory escape hatch must not be used for claims"
            )
        runtime_provenance = {
            "status": "UNVERIFIED_EXPLORATORY_ONLY",
            "warning": "evaluation hardware/software provenance was not supplied",
        }
    else:
        runtime_provenance = validate_runtime_provenance(cells, runtime_manifests)

    result: dict[str, Any] = {
        "schema_version": 4, "evaluation_provenance": eval_provenance,
        "evaluation_manifest_provenance": evaluation_manifest_provenance,
        "training_provenance": training_provenance,
        "runtime_provenance": runtime_provenance,
        "within_head": {},
    }
    deltas: dict[str, list[float]] = {"A": [], "C": []}
    c_deployment_deltas: list[float] = []
    all_task_rows: list[dict[str, Any]] = []
    for head in HEADS:
        for seed in SEEDS:
            name = f"{head}_s{seed}"
            suffix = "_nas10" if head == "C" else ""
            item = comparison(
                cells[f"{head}base_s{seed}{suffix}"],
                cells[f"{head}lang_s{seed}{suffix}"],
            )
            item["n_action_steps"] = 10
            result["within_head"][name] = item
            deltas[head].append(float(item["delta_pp"]))
            all_task_rows.extend(item["per_task"])

            if head == "C":
                deployment_item = comparison(
                    cells[f"Cbase_s{seed}"], cells[f"Clang_s{seed}"]
                )
                deployment_item["n_action_steps"] = 5
                result.setdefault("deployment_secondary", {})[f"C_s{seed}"] = deployment_item
                c_deployment_deltas.append(float(deployment_item["delta_pp"]))

    global_adjusted = holm([float(row["p_value"]) for row in all_task_rows])
    for row, corrected in zip(all_task_rows, global_adjusted, strict=True):
        row["holm_global_60_p_value"] = corrected

    interactions = np.array(deltas["C"]) - np.array(deltas["A"])
    bootstrap_low, bootstrap_high = hierarchical_interaction_ci(
        to_cube(cells, c_n_action_steps=10), samples=bootstrap_samples
    )
    t_low, t_high = seed_t_interval(interactions)
    result["training_seed_summary"] = {}
    for head in HEADS:
        head_deltas = np.asarray(deltas[head], dtype=float)
        head_low, head_high = seed_t_interval(head_deltas)
        result["training_seed_summary"][head] = {
            "n_action_steps": 10,
            "language_delta_pp_by_seed": deltas[head],
            "mean_delta_pp": float(np.mean(head_deltas)),
            "sample_sd_pp": float(np.std(head_deltas, ddof=1)),
            "seed_level_t_95_ci_pp": [float(head_low), float(head_high)],
            "inference_note": "Training seed is the replication unit (n=3, df=2).",
        }
    result["head_by_language_interaction"] = {
        "interaction_pp_by_seed": interactions.tolist(),
        "mean_interaction_pp": float(interactions.mean()),
        "sample_sd_pp": float(interactions.std(ddof=1)),
        "n_positive_seeds": int(np.sum(interactions > 0)),
        "seed_level_t_95_ci_pp": [float(t_low), float(t_high)],
        "hierarchical_bootstrap_95_ci_pp": [bootstrap_low, bootstrap_high],
        "inference_note": (
            "The seed-level t interval treats training seed (n=3) as the replication unit. "
            "A and C are both evaluated at 10 actions/replan. The hierarchical bootstrap "
            "additionally resamples paired tasks/states and is secondary."
        ),
    }
    # Added before locked-outcome access, but absent from the original pre-registration.
    result["exploratory_aggregate_regression_guardrail"] = {
        "status": "NOT_PREREGISTERED_DESCRIPTIVE_ONLY",
        "margin_pp": EXPLORATORY_REGRESSION_MARGIN_PP,
        "definition": (
            "Language-minus-standard suite success must be no worse than -margin; "
            "reported both for the seed mean and every individual seed."
        ),
        "by_head": {
            head: {
                "mean_delta_pp": float(np.mean(deltas[head])),
                "worst_seed_delta_pp": float(np.min(deltas[head])),
                "mean_passes": bool(np.mean(deltas[head]) >= -EXPLORATORY_REGRESSION_MARGIN_PP),
                "all_seeds_pass": bool(np.min(deltas[head]) >= -EXPLORATORY_REGRESSION_MARGIN_PP),
            }
            for head in HEADS
        },
        "deployment_secondary_C_nas5": {
            "language_delta_pp_by_seed": c_deployment_deltas,
            "mean_delta_pp": float(np.mean(c_deployment_deltas)),
            "worst_seed_delta_pp": float(np.min(c_deployment_deltas)),
            "mean_passes": bool(
                np.mean(c_deployment_deltas) >= -EXPLORATORY_REGRESSION_MARGIN_PP
            ),
            "all_seeds_pass": bool(
                np.min(c_deployment_deltas) >= -EXPLORATORY_REGRESSION_MARGIN_PP
            ),
        },
    }
    result["cadence_control"] = {}
    for seed in SEEDS:
        result["cadence_control"][f"s{seed}"] = {
            "nas5_language_effect": comparison(
                cells[f"Cbase_s{seed}"], cells[f"Clang_s{seed}"]
            ),
            "nas10_language_effect": comparison(
                cells[f"Cbase_s{seed}_nas10"], cells[f"Clang_s{seed}_nas10"]
            ),
            "baseline_nas5_to_nas10": comparison(
                cells[f"Cbase_s{seed}"], cells[f"Cbase_s{seed}_nas10"]
            ),
            "language_nas5_to_nas10": comparison(
                cells[f"Clang_s{seed}"], cells[f"Clang_s{seed}_nas10"]
            ),
        }
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    markdown = [
        "# Locked language-factorial results", "",
        f"Evaluation provenance: **{result['evaluation_provenance']['status']}**.  ",
        f"Config-variant binding: **{result['evaluation_manifest_provenance']['status']}**.  ",
        f"Training provenance: **{result['training_provenance']['status']}**.  ",
        f"Runtime provenance: **{result['runtime_provenance']['status']}**.", "",
        "| Head | Actions/replan | Seed | Standard | Granular mix | Delta | Paired p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for head in HEADS:
        for seed in SEEDS:
            item = result["within_head"][f"{head}_s{seed}"]
            markdown.append(
                f"| {head} | 10 | {seed} | {100 * item['left_rate']:.1f}% | "
                f"{100 * item['right_rate']:.1f}% | {item['delta_pp']:+.1f} pp | "
                f"{item['overall_mcnemar']['p_value']:.4g} |"
            )
    interaction = result["head_by_language_interaction"]
    a_summary = result["training_seed_summary"]["A"]
    c_summary = result["training_seed_summary"]["C"]
    markdown.extend([
        "",
        f"Primary A@10 language effect: **{a_summary['mean_delta_pp']:+.1f} pp** "
        f"(training-seed t 95% CI [{a_summary['seed_level_t_95_ci_pp'][0]:+.1f}, "
        f"{a_summary['seed_level_t_95_ci_pp'][1]:+.1f}] pp).",
        "",
        f"Primary C@10 language effect: **{c_summary['mean_delta_pp']:+.1f} pp** "
        f"(training-seed t 95% CI [{c_summary['seed_level_t_95_ci_pp'][0]:+.1f}, "
        f"{c_summary['seed_level_t_95_ci_pp'][1]:+.1f}] pp).",
        "",
        f"Primary cadence-matched A@10 vs C@10 head × language interaction: "
        f"**{interaction['mean_interaction_pp']:+.1f} pp** "
        f"(training-seed t 95% CI [{interaction['seed_level_t_95_ci_pp'][0]:+.1f}, "
        f"{interaction['seed_level_t_95_ci_pp'][1]:+.1f}] pp; secondary hierarchical "
        f"bootstrap 95% CI [{interaction['hierarchical_bootstrap_95_ci_pp'][0]:+.1f}, "
        f"{interaction['hierarchical_bootstrap_95_ci_pp'][1]:+.1f}] pp).",
        "",
        "The cadence-matched interaction—not a best-cell or deployment-cadence comparison—is "
        "the primary spline-specific estimand. The training seed is the replication unit (n=3).",
        "",
        "The 5 pp aggregate-regression guardrail is an analysis-plan addition made before "
        "outcome access, not an originally pre-registered noninferiority margin; it is descriptive only.",
    ])
    markdown.extend(
        [
            "",
            "## Deployment-cadence secondary (C@5)",
            "",
            "| Seed | Standard | Granular mix | Delta | Paired p |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        item = result["deployment_secondary"][f"C_s{seed}"]
        markdown.append(
            f"| {seed} | {100 * item['left_rate']:.1f}% | "
            f"{100 * item['right_rate']:.1f}% | {item['delta_pp']:+.1f} pp | "
            f"{item['overall_mcnemar']['p_value']:.4g} |"
        )
    if (
        result["training_provenance"]["status"] != "validated"
        or result["evaluation_manifest_provenance"]["status"] != "validated"
        or result["runtime_provenance"]["status"] != "validated"
    ):
        markdown.extend(
            ["", "> **UNVERIFIED TRAINING DATA — EXPLORATORY OUTPUT ONLY. DO NOT USE FOR CLAIMS.**"]
        )
    return "\n".join(markdown) + "\n"


def _parse_bindings(items: list[str], kind: str) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for item in items:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --{kind} binding: {item}")
        if name in bindings:
            raise ValueError(f"duplicate {kind} name: {name}")
        bindings[name] = Path(raw_path)
    return bindings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--train_manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--evaluation_manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--runtime_manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--allow_unverified_training_data", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--json_out", type=Path)
    parser.add_argument("--markdown_out", type=Path)
    args = parser.parse_args()

    bindings = _parse_bindings(args.cell, "cell")
    manifest_bindings = _parse_bindings(args.train_manifest, "train_manifest")
    evaluation_manifest_bindings = _parse_bindings(
        args.evaluation_manifest, "evaluation_manifest"
    )
    runtime_bindings = _parse_bindings(args.runtime_manifest, "runtime_manifest")
    cells = {name: load_cell(path) for name, path in bindings.items()}
    manifests = (
        {name: load_training_manifest(path) for name, path in manifest_bindings.items()}
        if manifest_bindings else None
    )
    evaluation_manifests = (
        {
            name: load_evaluation_manifest(path)
            for name, path in evaluation_manifest_bindings.items()
        }
        if evaluation_manifest_bindings
        else None
    )
    runtime_manifests = (
        {name: load_training_manifest(path) for name, path in runtime_bindings.items()}
        if runtime_bindings else None
    )
    result = analyze(
        cells,
        manifests,
        evaluation_manifests,
        runtime_manifests,
        allow_unverified_training_data=args.allow_unverified_training_data,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["inputs"] = {name: str(path.resolve()) for name, path in bindings.items()}
    if manifest_bindings:
        result["training_manifest_inputs"] = {
            name: str(path.resolve()) for name, path in manifest_bindings.items()
        }
    if evaluation_manifest_bindings:
        result["evaluation_manifest_inputs"] = {
            name: str(path.resolve())
            for name, path in evaluation_manifest_bindings.items()
        }
    if runtime_bindings:
        result["runtime_manifest_inputs"] = {
            name: str(path.resolve()) for name, path in runtime_bindings.items()
        }
    markdown_text = render_markdown(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.write_text(markdown_text)
    print(markdown_text)


if __name__ == "__main__":
    main()
