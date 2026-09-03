from __future__ import annotations

import copy
import hashlib
import unittest

import numpy as np

from scripts.analysis.language_factorial import (
    SEEDS,
    analyze,
    canonical_names,
    exact_mcnemar,
    hierarchical_interaction_ci,
    holm,
    render_markdown,
    to_cube,
    validate_cell_integrity,
    validate_evaluation_provenance,
    validate_evaluation_manifests,
    validate_runtime_provenance,
    validate_training_provenance,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def cell(name: str, *, threshold: int, nas: int, model_name: str | None = None) -> dict:
    head = name[0]
    tasks = []
    total = 0
    for task_id in range(10):
        episodes = []
        for state_id in range(50):
            success = state_id < threshold
            total += int(success)
            episodes.append(
                {
                    "state_id": state_id,
                    "seed": 100_000 + task_id * 1_000 + state_id,
                    "success": success,
                    "steps": 100,
                    "policy_calls": 10,
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "task_description": f"task {task_id}",
                "success_rate": threshold / 50,
                "episodes": episodes,
            }
        )
    return {
        "schema_version": 1,
        "protocol": "libero_explicit_init_state_v1",
        "checkpoint_config_sha256": digest(f"config-{head}-nas{nas}"),
        "model_sha256": digest(model_name or f"model-{name}"),
        "policy_type": "smolvla" if head == "A" else "smolvla_spline",
        "n_action_steps": nas,
        "suite": "libero_10",
        "control_frequency_hz": 20,
        "state_ids": list(range(50)),
        "seed_base": 100_000,
        "n_tasks": 10,
        "n_episodes": 500,
        "n_successes": total,
        "success_rate": total / 500,
        "policy_source_sha256": digest(f"policy-source-{head}"),
        "env_source_sha256": digest("env-source"),
        "lerobot_commit": "deadbeef",
        "stats_sha256": None if head == "A" else digest("spline-stats"),
        "evaluator_sha256": digest("locked-evaluator"),
        "slurm_job_id": str(900_000 + int(digest(name)[:5], 16)),
        "tasks": tasks,
    }


def factorial_cells(*, cadence: bool = True) -> dict[str, dict]:
    result = {}
    thresholds = {("A", "base"): 20, ("A", "lang"): 22, ("C", "base"): 20, ("C", "lang"): 25}
    for name in canonical_names():
        head = name[0]
        lang = "lang" if "lang" in name else "base"
        result[name] = cell(
            name,
            threshold=thresholds[(head, lang)],
            nas=10 if head == "A" else 5,
        )
    if cadence:
        nas10_thresholds = {"base": 20, "lang": 24}
        for lang in ("base", "lang"):
            for seed in SEEDS:
                canonical = f"C{lang}_s{seed}"
                control = f"{canonical}_nas10"
                result[control] = cell(
                    control,
                    threshold=nas10_thresholds[lang],
                    nas=10,
                    model_name=f"model-{canonical}",
                )
    return result


def training_manifests(cells: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for name in canonical_names():
        head = name[0]
        lang = "lang" if "lang" in name else "base"
        seed = int(name.rsplit("s", 1)[1])
        manifest = {
            "arm": head,
            "seed": seed,
            "dataset_info_sha256": digest("dataset-info"),
            "dataset_tasks_sha256": digest(f"dataset-tasks-{lang}"),
            "policy_type": cells[name]["policy_type"],
            "n_action_steps": 10 if head == "A" else 5,
            "lerobot_commit": "training-commit",
            "training_source_hashes": {
                "configuration.py": digest(f"training-configuration-{head}"),
                "policy_source": cells[name]["policy_source_sha256"],
                "train.py": digest("training-loop"),
            },
            "model_sha256": cells[name]["model_sha256"],
            "final_policy_config_sha256": cells[name]["checkpoint_config_sha256"],
            "train_config_sha256": digest(f"train-config-{name}"),
            "saved_train_seed": seed,
            "dataset_content_hashes": {
                "source_dataset_identity_sha256": digest("source-dataset"),
                "non_language_parquet_sha256": digest("non-language-parquet"),
                "task_index_schedule_sha256": digest(f"task-index-schedule-{lang}"),
                "episode_metadata_sha256": digest("episode-metadata"),
                "video_content_manifest_sha256": digest("video-content-manifest"),
            },
        }
        if head == "C":
            manifest["spline_stats_sha256"] = digest("spline-stats")
        result[name] = manifest
    return result


def evaluation_manifests(cells: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for name, payload in cells.items():
        head = name[0]
        native_nas = 10 if head == "A" else 5
        source_name = name.removesuffix("_nas10")
        source_config = {
            "type": payload["policy_type"],
            "n_action_steps": native_nas,
            "device": "cuda",
        }
        variant_config = dict(source_config)
        variant_config["n_action_steps"] = payload["n_action_steps"]
        artifact_hash = digest(f"evaluation-manifest-{name}")
        variant_checkpoint = f"/immutable/variants/{name}"
        payload["evaluation_manifest_sha256"] = artifact_hash
        payload["checkpoint"] = variant_checkpoint
        result[name] = {
            "schema_version": 2,
            "source_checkpoint": f"/immutable/sources/{source_name}",
            "variant_checkpoint": variant_checkpoint,
            "source_config_sha256": digest(f"config-{head}-nas{native_nas}"),
            "variant_config_sha256": payload["checkpoint_config_sha256"],
            "model_sha256": payload["model_sha256"],
            "original_n_action_steps": native_nas,
            "evaluated_n_action_steps": payload["n_action_steps"],
            "policy_type": payload["policy_type"],
            "suite": "libero_10",
            "state_start": 0,
            "states_per_task": 50,
            "control_frequency_hz": 20,
            "initial_states": list(range(50)),
            "slurm_job_id": payload["slurm_job_id"],
            "source_config": source_config,
            "variant_config": variant_config,
            "_artifact_sha256": artifact_hash,
        }
    return result


def runtime_manifests(cells: dict[str, dict]) -> dict[str, dict]:
    return {
        name: {
            "slurm_job_id": payload["slurm_job_id"],
            "gpu_model": "NVIDIA L40S",
            "cuda_version": "12.4",
            "cudnn_version": "9.1.0",
            "torch_version": "2.7.1",
        }
        for name, payload in cells.items()
    }


class PrimitiveStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_known_cases(self):
        left = np.array([True] * 5 + [False] * 5)
        right = np.array([False] * 5 + [False] * 5)
        result = exact_mcnemar(left, right)
        self.assertEqual((result["left_only"], result["right_only"]), (5, 0))
        self.assertAlmostEqual(result["p_value"], 0.0625)
        self.assertEqual(exact_mcnemar(left, left)["p_value"], 1.0)

    def test_holm_known_values_and_monotonicity(self):
        adjusted = holm([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
        with self.assertRaises(ValueError):
            holm([1.1])

    def test_hierarchical_bootstrap_is_deterministic_and_paired(self):
        cube = to_cube(factorial_cells(cadence=False), c_n_action_steps=5)
        first = hierarchical_interaction_ci(cube, samples=200, seed=9)
        second = hierarchical_interaction_ci(cube, samples=200, seed=9)
        self.assertEqual(first, second)
        self.assertLess(first[0], 6.0)
        self.assertGreater(first[1], 6.0)


class IntegrityAndProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.cells = factorial_cells()
        self.evaluation_manifests = evaluation_manifests(self.cells)
        self.manifests = training_manifests(self.cells)
        self.runtime_manifests = runtime_manifests(self.cells)

    def test_valid_artifacts_and_provenance(self):
        for name, payload in self.cells.items():
            validate_cell_integrity(payload, name)
        eval_audit = validate_evaluation_provenance(self.cells)
        binding_audit = validate_evaluation_manifests(
            self.cells, self.evaluation_manifests
        )
        train_audit = validate_training_provenance(
            self.cells, self.manifests, self.evaluation_manifests
        )
        runtime_audit = validate_runtime_provenance(self.cells, self.runtime_manifests)
        self.assertEqual(eval_audit["status"], "validated")
        self.assertEqual(
            eval_audit["cadence_matched_c_cells"], "validated_all_three_seeds"
        )
        self.assertEqual(train_audit["status"], "validated")
        self.assertEqual(binding_audit["n_bound_manifests"], 18)
        self.assertEqual(runtime_audit["gpu_model"], "NVIDIA L40S")

    def test_rejects_missing_state_and_wrong_rng_seed(self):
        damaged = copy.deepcopy(self.cells["Abase_s1000"])
        damaged["tasks"][0]["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "noncanonical|50 episodes"):
            validate_cell_integrity(damaged, "damaged")
        damaged = copy.deepcopy(self.cells["Abase_s1000"])
        damaged["tasks"][4]["episodes"][3]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_cell_integrity(damaged, "damaged")

    def test_rejects_evaluator_mismatch_and_model_reuse(self):
        damaged = copy.deepcopy(self.cells)
        damaged["Alang_s1001"]["evaluator_sha256"] = digest("other-evaluator")
        with self.assertRaisesRegex(ValueError, "evaluator_sha256"):
            validate_evaluation_provenance(damaged)
        damaged = copy.deepcopy(self.cells)
        damaged["Alang_s1001"]["model_sha256"] = damaged["Abase_s1001"]["model_sha256"]
        with self.assertRaisesRegex(ValueError, "reuse model"):
            validate_evaluation_provenance(damaged)

    def test_rejects_within_head_config_drift(self):
        damaged = copy.deepcopy(self.cells)
        damaged["Alang_s1001"]["checkpoint_config_sha256"] = digest("drifted-config")
        with self.assertRaisesRegex(ValueError, "checkpoint_config_sha256"):
            validate_evaluation_provenance(damaged)

    def test_rejects_broken_or_partial_cadence_control(self):
        damaged = copy.deepcopy(self.cells)
        del damaged["Clang_s1000_nas10"]
        with self.assertRaisesRegex(ValueError, "missing required cadence-matched C cells"):
            validate_evaluation_provenance(damaged)
        damaged = copy.deepcopy(self.cells)
        damaged["Cbase_s1000_nas10"]["model_sha256"] = digest("wrong-model")
        with self.assertRaisesRegex(ValueError, "changed model"):
            validate_evaluation_provenance(damaged)
        damaged = copy.deepcopy(self.cells)
        damaged["Cbase_s1000_nas10"]["checkpoint_config_sha256"] = damaged[
            "Cbase_s1000"
        ]["checkpoint_config_sha256"]
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_evaluation_provenance(damaged)

    def test_rejects_seed_specific_cadence_model_mismatch(self):
        damaged = copy.deepcopy(self.cells)
        damaged["Clang_s1002_nas10"]["model_sha256"] = digest("wrong-seed-model")
        with self.assertRaisesRegex(ValueError, "seed 1002 cadence control changed model"):
            validate_evaluation_provenance(damaged)

    def test_rejects_unbound_eval_manifest_or_variant_config(self):
        damaged_cells = copy.deepcopy(self.cells)
        damaged_cells["Alang_s1001"]["evaluation_manifest_sha256"] = digest(
            "different-evaluation-manifest"
        )
        with self.assertRaisesRegex(ValueError, "does not bind the supplied"):
            validate_evaluation_manifests(damaged_cells, self.evaluation_manifests)

        damaged_manifests = copy.deepcopy(self.evaluation_manifests)
        damaged_manifests["Clang_s1001_nas10"]["variant_config"]["hidden_size"] = 999
        with self.assertRaisesRegex(ValueError, "beyond n_action_steps"):
            validate_evaluation_manifests(self.cells, damaged_manifests)

    def test_rejects_cadence_variants_from_different_source_configs(self):
        damaged = copy.deepcopy(self.evaluation_manifests)
        damaged["Cbase_s1002_nas10"]["source_config_sha256"] = digest(
            "different-source-config"
        )
        with self.assertRaisesRegex(ValueError, "do not share one source"):
            validate_evaluation_manifests(self.cells, damaged)

    def test_rejects_data_leakage_and_training_confound(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Alang_s1001"]["dataset_tasks_sha256"] = digest("different-lang-table")
        with self.assertRaisesRegex(ValueError, "do not share one dataset"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)
        damaged = copy.deepcopy(self.manifests)
        for name in canonical_names():
            if "lang" in name:
                damaged[name]["dataset_tasks_sha256"] = digest("dataset-tasks-base")
        with self.assertRaisesRegex(ValueError, "distinct task-table"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)
        damaged = copy.deepcopy(self.manifests)
        damaged["Clang_s1002"]["lerobot_commit"] = "different-commit"
        with self.assertRaisesRegex(ValueError, "training source commits differ"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_non_language_dataset_difference(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Alang_s1001"]["dataset_content_hashes"][
            "non_language_parquet_sha256"
        ] = digest("different-actions-or-observations")
        with self.assertRaisesRegex(ValueError, "full dataset-content identity"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

        damaged = copy.deepcopy(self.manifests)
        for name in canonical_names():
            if "lang" in name:
                damaged[name]["dataset_content_hashes"][
                    "video_content_manifest_sha256"
                ] = digest("different-videos")
        with self.assertRaisesRegex(ValueError, "differ outside the language"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_training_eval_stats_mismatch(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Clang_s1002"]["spline_stats_sha256"] = digest("other-stats")
        with self.assertRaisesRegex(ValueError, "training/evaluation spline statistics differ"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_same_commit_but_different_training_source_file(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Clang_s1002"]["training_source_hashes"]["train.py"] = digest(
            "silently-modified-untracked-model"
        )
        with self.assertRaisesRegex(ValueError, "source-file hashes differ"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_policy_source_mutation_between_training_and_eval(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Clang_s1002"]["training_source_hashes"]["policy_source"] = digest(
            "policy-source-mutated-after-training"
        )
        with self.assertRaisesRegex(ValueError, "training/evaluation policy source differs"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_model_or_saved_train_seed_binding_mismatch(self):
        damaged = copy.deepcopy(self.manifests)
        damaged["Alang_s1002"]["model_sha256"] = digest("wrong-final-model")
        with self.assertRaisesRegex(ValueError, "bind the evaluated model"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)
        damaged = copy.deepcopy(self.manifests)
        damaged["Alang_s1002"]["saved_train_seed"] = 1000
        with self.assertRaisesRegex(ValueError, "saved train-config seed is wrong"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)
        damaged = copy.deepcopy(self.manifests)
        damaged["Alang_s1002"]["final_policy_config_sha256"] = digest(
            "wrong-training-source-config"
        )
        with self.assertRaisesRegex(ValueError, "bind the evaluation source config"):
            validate_training_provenance(self.cells, damaged, self.evaluation_manifests)

    def test_rejects_task_text_and_hardware_confound(self):
        damaged_cells = copy.deepcopy(self.cells)
        damaged_cells["Alang_s1001"]["tasks"][0]["task_description"] = "different task"
        with self.assertRaisesRegex(ValueError, "task descriptions differ"):
            validate_evaluation_provenance(damaged_cells)
        damaged_runtime = copy.deepcopy(self.runtime_manifests)
        damaged_runtime["Clang_s1002_nas10"]["gpu_model"] = "NVIDIA H100"
        with self.assertRaisesRegex(ValueError, "hardware/software identities differ"):
            validate_runtime_provenance(self.cells, damaged_runtime)


class FactorialAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.cells = factorial_cells()
        self.evaluation_manifests = evaluation_manifests(self.cells)
        self.manifests = training_manifests(self.cells)
        self.runtime_manifests = runtime_manifests(self.cells)

    def test_requires_training_data_provenance_by_default(self):
        with self.assertRaisesRegex(ValueError, "evaluation manifests are required"):
            analyze(self.cells, None, bootstrap_samples=20)
        with self.assertRaisesRegex(ValueError, "training manifests are required"):
            analyze(self.cells, None, self.evaluation_manifests, bootstrap_samples=20)
        with self.assertRaisesRegex(ValueError, "runtime manifests are required"):
            analyze(
                self.cells,
                self.manifests,
                self.evaluation_manifests,
                bootstrap_samples=20,
            )
        exploratory = analyze(
            self.cells,
            None,
            allow_unverified_training_data=True,
            bootstrap_samples=20,
        )
        self.assertEqual(
            exploratory["training_provenance"]["status"], "UNVERIFIED_EXPLORATORY_ONLY"
        )
        self.assertIn("DO NOT USE FOR CLAIMS", render_markdown(exploratory))

    def test_seed_level_interaction_and_corrections(self):
        result = analyze(
            self.cells,
            self.manifests,
            self.evaluation_manifests,
            self.runtime_manifests,
            bootstrap_samples=200,
        )
        interaction = result["head_by_language_interaction"]
        a_summary = result["training_seed_summary"]["A"]
        c_summary = result["training_seed_summary"]["C"]
        np.testing.assert_allclose(a_summary["language_delta_pp_by_seed"], [4.0, 4.0, 4.0])
        np.testing.assert_allclose(a_summary["seed_level_t_95_ci_pp"], [4.0, 4.0])
        np.testing.assert_allclose(c_summary["language_delta_pp_by_seed"], [8.0, 8.0, 8.0])
        np.testing.assert_allclose(c_summary["seed_level_t_95_ci_pp"], [8.0, 8.0])
        # Primary uses C@10: (24-20)*2 - (22-20)*2 = 4 pp.
        np.testing.assert_allclose(interaction["interaction_pp_by_seed"], [4.0, 4.0, 4.0])
        self.assertAlmostEqual(interaction["mean_interaction_pp"], 4.0)
        np.testing.assert_allclose(interaction["seed_level_t_95_ci_pp"], [4.0, 4.0])
        self.assertEqual(interaction["n_positive_seeds"], 3)
        for comparison in result["within_head"].values():
            for row in comparison["per_task"]:
                self.assertIn("holm_within_cell_p_value", row)
                self.assertIn("holm_global_60_p_value", row)
        guardrail = result["exploratory_aggregate_regression_guardrail"]
        self.assertEqual(guardrail["status"], "NOT_PREREGISTERED_DESCRIPTIVE_ONLY")
        self.assertTrue(guardrail["by_head"]["A"]["all_seeds_pass"])
        self.assertIn("cadence_control", result)
        np.testing.assert_allclose(
            guardrail["deployment_secondary_C_nas5"]["language_delta_pp_by_seed"],
            [10.0, 10.0, 10.0],
        )
        markdown = render_markdown(result)
        self.assertIn("Primary A@10 language effect", markdown)
        self.assertIn("Primary C@10 language effect", markdown)
        self.assertIn("Deployment-cadence secondary (C@5)", markdown)


if __name__ == "__main__":
    unittest.main()
