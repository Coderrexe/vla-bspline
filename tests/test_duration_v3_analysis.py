from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.duration_v3_analysis import (
    ValidationError,
    _merge_validated_task_shards,
    analyze,
    canonical_json_sha256,
    exact_mcnemar,
    holm_adjust,
    write_outputs,
)
from scripts.eval.stochastic_duration_v3 import (
    WATERMARK,
    materialize_task_order_manifest,
)


SOURCE_FIELDS = {
    "evaluator_sha256": "1" * 64,
    "controller_sha256": "2" * 64,
    "stochastic_helper_sha256": "3" * 64,
    "v2_hashing_sha256": "4" * 64,
    "hidden_state_helper_sha256": "5" * 64,
    "base_evaluator_sha256": "6" * 64,
    "action_helper_sha256": "7" * 64,
    "policy_source_sha256": "8" * 64,
    "env_source_sha256": "9" * 64,
    "stats_sha256": "a" * 64,
    "lerobot_commit": "b" * 40,
}


def embedded(payload: dict, field: str = "manifest_sha256") -> dict:
    payload[field] = canonical_json_sha256(payload)
    return payload


def episode(
    arm: str,
    spec: dict,
    *,
    success: bool,
    executed: list[int] | None = None,
) -> dict:
    predicted = [20, 24]
    if executed is None:
        executed = list(predicted)
    chunks = [
        {
            "predicted_duration": pred,
            "executed_duration": used,
            "absolute_duration_intervention": abs(pred - used),
        }
        for pred, used in zip(predicted, executed, strict=True)
    ]
    changed = sum(pred != used for pred, used in zip(predicted, executed))
    mode = "predicted" if arm.startswith("predicted_") else arm
    return {
        "arm_label": arm,
        "duration_mode": mode,
        "task_id": 0,
        "state_id": spec["state_id"],
        "repeat_index": spec["repeat_index"],
        "env_seed_u32": spec["env_seed_u32"],
        "policy_seed_u64": spec["policy_seed_u64"],
        "success": success,
        "steps": 2,
        "initial_raw_observation_sha256": "c" * 64,
        "initial_processed_observation_sha256": "d" * 64,
        "initial_mujoco_integration_state_sha256": "e" * 64,
        "compiled_model_xml_sha256": "f" * 64,
        "raw_observation_step_sha256": ["r0", "r1"],
        "processed_observation_step_sha256": ["p0", "p1"],
        "mujoco_integration_state_step_sha256": ["m0", "m1"],
        "executed_action_step_sha256": ["a0", "a1"],
        "executed_action_prefix_sha256": ["x0", "x1"],
        "executed_action_trace_sha256": "f" * 64,
        "predicted_durations": predicted,
        "executed_durations": executed,
        "duration_chunks": chunks,
        "n_duration_calls": 2,
        "n_intervened_duration_calls": changed,
        "intervention_call_fraction": changed / 2,
        "mean_absolute_duration_intervention": sum(
            abs(pred - used) for pred, used in zip(predicted, executed)
        )
        / 2,
        "counterfactual_source_length": 2 if arm == "shuffled" else None,
        "shuffle_source_manifest_sha256": None,
    }


def make_artifact(model_digit: str = "0") -> dict:
    model_hash = model_digit * 64
    checkpoint = f"/checkpoint/{model_digit}"
    config_hash = "c" * 64
    prior_hash = "d" * 64
    manifest = materialize_task_order_manifest(
        suite="libero_10",
        task_id=0,
        state_ids=[0],
        repeats=6,
        seed_base=100_000,
        order_seed=20_260_821,
    )
    resume = {
        "protocol": "duration_v3_atomic_block_resume_identity",
        "checkpoint": checkpoint,
        "model_sha256": model_hash,
        "checkpoint_config_sha256": config_hash,
        "fixed_manifest_sha256": prior_hash,
        "suite": "libero_10",
        "task_ids": [0],
        "state_ids": [0],
        "repeats": 6,
        "seed_base": 100_000,
        "order_seed": 20_260_821,
        "control_frequency_hz": 20,
        "shuffle_seed": 73_921,
        "shuffle_strength_threshold": 0.5,
        "repeatability_equivalence_margin": 0.1,
        "mujoco_gl": "osmesa",
        "require_exact_predicted_replay": True,
        **SOURCE_FIELDS,
    }
    resume_hash = canonical_json_sha256(resume)
    runtime = {
        "protocol": "duration_v3_process_runtime_manifest",
        "mujoco_gl": "osmesa",
        "slurm_job_id": "123",
    }
    runtime_hash = canonical_json_sha256(runtime)
    predicted_success = [True, True, False, True, False, True]
    fixed_success = [False, True, False, False, False, True]
    shuffled_success = [True, False, False, False, False, False]
    blocks = []
    for index, spec in enumerate(manifest["blocks"]):
        episodes = {
            "predicted_donor": episode(
                "predicted_donor", spec, success=predicted_success[index]
            ),
            "predicted_eval": episode(
                "predicted_eval", spec, success=predicted_success[index]
            ),
            "fixed": episode(
                "fixed", spec, success=fixed_success[index], executed=[24, 24]
            ),
            "shuffled": episode(
                "shuffled", spec, success=shuffled_success[index], executed=[24, 20]
            ),
            "predicted_closure": episode(
                "predicted_closure", spec, success=predicted_success[index]
            ),
        }
        source = embedded(
            {
                "protocol": "in_process_predicted_donor_shuffle_source_v3",
                "resume_identity_sha256": resume_hash,
                "checkpoint": checkpoint,
                "model_sha256": model_hash,
                "checkpoint_config_sha256": config_hash,
                "suite": "libero_10",
                "configured_task_ids": [0],
                "configured_state_ids": [0],
                "configured_repeats": 6,
                "task_order_manifest_sha256": manifest["manifest_sha256"],
                "task_id": 0,
                "state_id": 0,
                "repeat_index": spec["repeat_index"],
                "env_seed_u32": spec["env_seed_u32"],
                "policy_seed_u64": spec["policy_seed_u64"],
                "control_frequency_hz": 20,
                "shuffle_seed": 73_921,
                "intervention_protocol": "test",
                "predicted_durations": [20, 24],
                "donor_action_trace_sha256": "f" * 64,
            }
        )
        episodes["shuffled"]["shuffle_source_manifest_sha256"] = source[
            "manifest_sha256"
        ]
        pairings = {
            arm: {
                "arm_label": arm,
                "seed_agreement": True,
                "raw_initial_observation_agreement": True,
                "processed_initial_observation_agreement": True,
                "mujoco_integration_initial_state_agreement": True,
                "compiled_model_xml_agreement": True,
            }
            for arm in spec["arm_order"][1:]
        }
        blocks.append(
            {
                "execution_runtime_sha256": runtime_hash,
                "task_block_ordinal": spec["task_block_ordinal"],
                "task_order_manifest_sha256": manifest["manifest_sha256"],
                "task_id": 0,
                "state_id": 0,
                "repeat_index": spec["repeat_index"],
                "env_seed_u32": spec["env_seed_u32"],
                "policy_seed_u64": spec["policy_seed_u64"],
                "arm_order": spec["arm_order"],
                "efficacy_permutation": spec["efficacy_permutation"],
                "shuffle_source_manifest": source,
                "initial_pairing_by_arm": pairings,
                "episodes": episodes,
            }
        )
    return {
        "schema_version": 3,
        "protocol": "libero_duration_stochastic_paired_block_v3",
        "watermark": WATERMARK,
        "scientific_arms": ["predicted_eval", "fixed", "shuffled"],
        "trace_only_arms": ["predicted_donor", "predicted_closure"],
        "primary_predicted_arm": "predicted_eval",
        "exact_predicted_replay_required": True,
        "exact_predicted_replay_all_pairs": True,
        "fixed_duration": 24,
        "fixed_manifest_sha256": prior_hash,
        "fixed_manifest_protocol": "production_event_duration_training_prior_v2_parquet",
        "fixed_manifest_complete_training_scan": True,
        "shuffle_seed": 73_921,
        "checkpoint": checkpoint,
        "checkpoint_config_sha256": config_hash,
        "model_sha256": model_hash,
        "policy_type": "smolvla_bspline",
        "n_action_steps": 5,
        "suite": "libero_10",
        "task_ids": [0],
        "state_ids": [0],
        "repeats": 6,
        "seed_base": 100_000,
        "order_seed": 20_260_821,
        "control_frequency_hz": 20,
        "n_blocks": 6,
        "n_episodes": 30,
        "efficacy_permutation_counts": manifest["permutation_counts"],
        "order_balance_gate": {"passed": True},
        "resume_identity": resume,
        "resume_identity_sha256": resume_hash,
        "execution_runtime_manifests": {runtime_hash: runtime},
        "initial_pairing_complete": True,
        "initial_pairing_summary": {
            "n_arm_episode_pairs": 24,
            "n_seed_agreements": 24,
            "n_raw_initial_observation_agreements": 24,
            "n_processed_initial_observation_agreements": 24,
            "n_mujoco_integration_initial_state_agreements": 24,
            "n_compiled_model_xml_agreements": 24,
        },
        "shuffle_strength_gate": {
            "target_minimum": 0.5,
            "observed_changed_call_fraction": 1.0,
            "passed": True,
        },
        "tasks": [
            {
                "task_id": 0,
                "task_order_manifest": manifest,
                "n_blocks": 6,
                "blocks": blocks,
            }
        ],
        **SOURCE_FIELDS,
    }


class DurationV3AnalysisTest(unittest.TestCase):
    def write_artifact(self, directory: Path, payload: dict, name: str) -> Path:
        path = directory / name
        path.write_text(json.dumps(payload))
        return path

    def test_exact_mcnemar_and_holm(self) -> None:
        result = exact_mcnemar([True, True, False, False], [False, False, True, False])
        self.assertEqual(result["first_success_second_failure"], 2)
        self.assertEqual(result["first_failure_second_success"], 1)
        self.assertEqual(result["exact_two_sided_p"], 1.0)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adjusted["a"]["holm_adjusted_p"], 0.02)
        self.assertAlmostEqual(adjusted["b"]["holm_adjusted_p"], 0.04)

    def test_disjoint_same_model_task_shards_merge_as_one_checkpoint(self) -> None:
        def normalized(task_id: int, path: str) -> dict:
            artifact = {
                "checkpoint": "/checkpoint/shared",
                "suite": "libero_10",
                "task_ids": [task_id],
                "state_ids": [0],
                "repeats": 1,
                "seed_base": 100_000,
                "order_seed": 20_260_821,
                "control_frequency_hz": 20,
                "shuffle_seed": 73_921,
                "fixed_duration": 24,
                "policy_type": "smolvla_bspline",
                "n_action_steps": 5,
            }
            predicted = {"success": True}
            block = {
                "task_id": task_id,
                "state_id": 0,
                "repeat_index": 0,
                "episodes": {
                    "predicted_donor": dict(predicted),
                    "predicted_eval": dict(predicted),
                    "predicted_closure": dict(predicted),
                    "fixed": {"success": False},
                    "shuffled": {
                        "success": False,
                        "n_intervened_duration_calls": 1,
                        "n_duration_calls": 2,
                    },
                },
            }
            return {
                "path": path,
                "artifact_sha256": str(task_id + 1) * 64,
                "artifact": artifact,
                "model_sha256": "a" * 64,
                "checkpoint_config_sha256": "b" * 64,
                "fixed_manifest_sha256": "c" * 64,
                "source_identity": SOURCE_FIELDS,
                "task_manifest_hashes": {str(task_id): "d" * 64},
                "blocks": [block],
                "exact_predicted_replay": True,
                "exact_predicted_replay_by_pair": {
                    name: True for name in ("donor_vs_eval", "donor_vs_closure", "eval_vs_closure")
                },
                "shuffle_strength": 0.5,
                "shuffle_strength_by_task": {str(task_id): 0.5},
                "mujoco_gl_values": ["osmesa"],
            }

        merged = _merge_validated_task_shards(
            [normalized(0, "left.json"), normalized(1, "right.json")]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["artifact"]["task_ids"], [0, 1])
        self.assertEqual(len(merged[0]["blocks"]), 2)
        self.assertEqual(merged[0]["input_paths"], ["left.json", "right.json"])

        with self.assertRaisesRegex(ValidationError, "overlap"):
            _merge_validated_task_shards(
                [normalized(0, "left.json"), normalized(0, "duplicate.json")]
            )

    def test_valid_single_checkpoint_is_explicitly_exploratory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_artifact(Path(temp), make_artifact("1"), "result.json")
            result = analyze([path], bootstrap_samples=200, bootstrap_seed=9)
        self.assertEqual(result["claim_status"], "exploratory_single_checkpoint_only")
        self.assertTrue(result["all_exact_predicted_replay"])
        self.assertTrue(result["single_checkpoint_hierarchical_bootstrap"]["available"])
        self.assertFalse(result["training_seed_inference"]["fixed"]["available"])
        self.assertEqual(
            result["per_checkpoint"][0]["primary_contrasts"]["fixed"][
                "delta_predicted_minus_control_pp"
            ],
            100 * (4 / 6 - 2 / 6),
        )

    def test_required_exact_replay_is_recomputed_from_episode_traces(self) -> None:
        payload = make_artifact("2")
        payload["tasks"][0]["blocks"][0]["episodes"]["predicted_eval"][
            "executed_action_step_sha256"
        ][1] = "different"
        payload["exact_predicted_replay_all_pairs"] = False
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_artifact(Path(temp), payload, "bad.json")
            with self.assertRaisesRegex(ValidationError, "required exact predicted replay"):
                analyze([path], bootstrap_samples=10)

    def test_legacy_diagnostic_without_exact_summary_is_recomputed(self) -> None:
        payload = make_artifact("9")
        payload.pop("exact_predicted_replay_all_pairs")
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_artifact(Path(temp), payload, "legacy.json")
            result = analyze([path], bootstrap_samples=10)
        self.assertTrue(result["all_exact_predicted_replay"])

    def test_duplicate_block_is_rejected(self) -> None:
        payload = make_artifact("3")
        payload["tasks"][0]["blocks"].append(
            copy.deepcopy(payload["tasks"][0]["blocks"][0])
        )
        payload["tasks"][0]["n_blocks"] = 7
        payload["n_blocks"] = 7
        payload["n_episodes"] = 35
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_artifact(Path(temp), payload, "duplicate.json")
            with self.assertRaisesRegex(ValidationError, "duplicate block coordinate"):
                analyze([path], bootstrap_samples=10)

    def test_weak_shuffle_is_rejected_even_if_reported_consistently(self) -> None:
        payload = make_artifact("8")
        for block in payload["tasks"][0]["blocks"]:
            shuffled = block["episodes"]["shuffled"]
            shuffled["executed_durations"] = [20, 24]
            shuffled["duration_chunks"] = [
                {
                    "predicted_duration": 20,
                    "executed_duration": 20,
                    "absolute_duration_intervention": 0,
                },
                {
                    "predicted_duration": 24,
                    "executed_duration": 24,
                    "absolute_duration_intervention": 0,
                },
            ]
            shuffled["n_intervened_duration_calls"] = 0
            shuffled["intervention_call_fraction"] = 0.0
            shuffled["mean_absolute_duration_intervention"] = 0.0
        payload["shuffle_strength_gate"]["observed_changed_call_fraction"] = 0.0
        payload["shuffle_strength_gate"]["passed"] = False
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_artifact(Path(temp), payload, "weak.json")
            with self.assertRaisesRegex(ValidationError, "shuffle strength gate failed"):
                analyze([path], bootstrap_samples=10)

    def test_cross_checkpoint_config_mismatch_is_rejected(self) -> None:
        first = make_artifact("8")
        second = make_artifact("9")
        second["checkpoint_config_sha256"] = "e" * 64
        second["resume_identity"]["checkpoint_config_sha256"] = "e" * 64
        second["resume_identity_sha256"] = canonical_json_sha256(
            second["resume_identity"]
        )
        # Keep every block-local shuffle manifest internally bound to the new
        # result identity so only the cross-checkpoint mismatch is under test.
        for block in second["tasks"][0]["blocks"]:
            source = block["shuffle_source_manifest"]
            source["checkpoint_config_sha256"] = "e" * 64
            source["resume_identity_sha256"] = second["resume_identity_sha256"]
            source.pop("manifest_sha256")
            embedded(source)
            block["episodes"]["shuffled"]["shuffle_source_manifest_sha256"] = source[
                "manifest_sha256"
            ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = [
                self.write_artifact(directory, first, "first.json"),
                self.write_artifact(directory, second, "second.json"),
            ]
            with self.assertRaisesRegex(ValidationError, "config hash mismatch"):
                analyze(paths, bootstrap_samples=10)

    def test_three_distinct_models_unlock_training_seed_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = [
                self.write_artifact(directory, make_artifact(digit), f"{digit}.json")
                for digit in ("4", "5", "6")
            ]
            result = analyze(paths, bootstrap_samples=10)
        self.assertEqual(result["n_independent_checkpoints"], 3)
        self.assertTrue(result["training_seed_inference"]["fixed"]["available"])
        self.assertEqual(result["claim_status"], "multi_checkpoint_with_training_seed_intervals")

    def test_atomic_pair_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            artifact = self.write_artifact(
                directory, make_artifact("7"), "result.json"
            )
            result = analyze([artifact], bootstrap_samples=10)
            prefix = directory / "analysis"
            json_path, markdown_path = write_outputs(result, prefix)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            with self.assertRaises(FileExistsError):
                write_outputs(result, prefix)


if __name__ == "__main__":
    unittest.main()
