from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval" / "libero_duration_ablation_v4_frozen.py"
FROZEN = ROOT / "scripts" / "eval" / "libero_frozen_reset.py"


class LiberoDurationAblationV4FrozenContractTest(unittest.TestCase):
    def test_single_capture_then_all_arms_restore_the_same_live_model(self) -> None:
        source = EVALUATOR.read_text()
        self.assertIn("_capture_once(env, state_id, env_seed)", source)
        self.assertEqual(source.count("env.reset(seed=env_seed)"), 1)
        self.assertIn("FrozenLiberoState.capture", source)
        self.assertIn("frozen.restore(base_env, _structured_sha256, wrapper_root=env)", source)
        self.assertIn("no_reset_after_block_capture", source)
        self.assertNotIn("_explicit_reset", source)
        self.assertIn("step_without_terminal_autoreset", source)
        self.assertIn("suppressed_terminal_autoresets", source)
        self.assertIn("terminal_autoreset_protocol", source)

    def test_frozen_restore_uses_full_mujoco_state_and_rejects_xml_drift(self) -> None:
        source = FROZEN.read_text()
        self.assertIn("mujoco.mj_setState", source)
        self.assertIn("mujoco.mj_forward", source)
        self.assertIn("compiled model XML changed", source)
        self.assertIn("integration state did not round-trip exactly", source)
        self.assertIn("_restore_wrapper_fields", source)
        self.assertIn("_autoreset_envs", source)

    def test_five_arm_pairing_proves_raw_processed_and_xml_identity(self) -> None:
        source = EVALUATOR.read_text()
        for arm in (
            "predicted_donor", "predicted_eval", "fixed", "shuffled",
            "predicted_closure",
        ):
            self.assertIn(arm, source)
        for field in (
            "initial_raw_observation_sha256",
            "initial_processed_observation_sha256",
            "compiled_model_xml_sha256",
            "n_all_frozen_initial_field_agreements",
            "frozen_initial_pairing_complete",
        ):
            self.assertIn(field, source)
        self.assertIn("--require_frozen_initial_pairing", source)

    def test_v3_counterbalance_and_atomic_provenance_are_retained(self) -> None:
        source = EVALUATOR.read_text()
        for field in (
            "materialize_task_order_manifest", "EFFICACY_PERMUTATIONS",
            "COUNTERBALANCE_PROTOCOL", "BLOCK_SEED_PROTOCOL",
            "atomic_write_json_new(block_path, artifact)",
            "predicted_repeatability", "shuffle_strength_gate",
            "repeatability_equivalence_gate",
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
