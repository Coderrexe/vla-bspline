from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval" / "libero_duration_ablation_v3.py"
CONTROLLER = ROOT / "scripts" / "eval" / "duration_intervention_v3.py"


class LiberoDurationAblationV3ContractTest(unittest.TestCase):
    def test_uses_production_scalar_path_and_no_v2_cached_duration(self) -> None:
        controller = CONTROLLER.read_text()
        self.assertIn("self._original(tokens_unnorm)", controller)
        self.assertNotIn("last_predicted_T_batch", controller)

    def test_task_local_randomized_manifest_replaces_global_ordinal(self) -> None:
        source = EVALUATOR.read_text()
        self.assertIn("materialize_task_order_manifest(", source)
        self.assertIn("task_order_manifest_sha256", source)
        self.assertNotIn("global_block_ordinal", source)
        self.assertNotIn("efficacy_order(", source)

    def test_five_rollout_atomic_block_and_all_predicted_pairs(self) -> None:
        source = EVALUATOR.read_text()
        for arm in (
            "predicted_donor",
            "predicted_eval",
            "fixed",
            "shuffled",
            "predicted_closure",
        ):
            self.assertIn(arm, source)
        self.assertIn("duration_v3_atomic_complete_block", source)
        self.assertIn("atomic_write_json_new(block_path, block_artifact)", source)
        for pair in ("donor_vs_eval", "donor_vs_closure", "eval_vs_closure"):
            self.assertIn(pair, source)
        self.assertIn("execution_runtime_manifest", source)
        self.assertIn("execution_runtime_sha256", source)

    def test_step_replan_and_coverage_labelled_mujoco_telemetry(self) -> None:
        source = EVALUATOR.read_text()
        for field in (
            "raw_observation_step_sha256",
            "processed_observation_step_sha256",
            "raw_observation_component_step_sha256",
            "processed_observation_component_step_sha256",
            "mujoco_integration_state_step_sha256",
            "canonical_replan_record_sha256",
            "predicted_duration",
            "executed_duration",
        ):
            self.assertIn(field, source)
        self.assertIn("locate_robosuite_sim(_base_env(env))", source)
        self.assertIn("integration_state_sha256(simulator)", source)
        self.assertIn("compiled_model_xml_sha256", source)
        self.assertIn("observation_component_hashes(", source)
        self.assertNotIn("initial_full_simulator_state", source)

    def test_stochastic_watermark_and_gates_are_explicit(self) -> None:
        source = EVALUATOR.read_text()
        for field in (
            "stochastic_result",
            "bitwise_replay_assumed",
            "shuffle_strength_gate",
            "repeatability_equivalence_gate",
            "predicted_repeatability_by_task",
            "order_balance_gate",
            "equal_counterfactual_source_multiset_fraction",
        ):
            self.assertIn(field, source)
        self.assertIn("pilot_all_tasks_within_margin", source)
        self.assertIn("shuffle_all_tasks_passed", source)

    def test_preregistered_defaults(self) -> None:
        source = EVALUATOR.read_text()
        self.assertIn('"--states_per_task", type=int, default=12', source)
        self.assertIn('"--repeats", type=int, default=1', source)
        self.assertIn('"confirmatory_95pct_ci_margin": 0.05', source)

    def test_optional_exact_replay_gate_publishes_then_fails(self) -> None:
        source = EVALUATOR.read_text()
        self.assertIn('"--require_exact_predicted_replay"', source)
        self.assertIn('"exact_predicted_replay_all_pairs"', source)
        self.assertLess(
            source.index("atomic_write_json_new(output, result)"),
            source.index("exact predicted replay was required"),
        )


if __name__ == "__main__":
    unittest.main()
