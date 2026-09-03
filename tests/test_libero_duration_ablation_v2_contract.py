from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval" / "libero_duration_ablation_v2.py"
CONTROLLER = ROOT / "scripts" / "eval" / "duration_intervention_v2.py"


class LiberoDurationAblationV2ContractTest(unittest.TestCase):
    def test_canonical_determinism_is_imported_before_torch(self) -> None:
        source = EVALUATOR.read_text()
        self.assertLess(
            source.index("from libero_locked_eval_v2 import"),
            source.index("import torch"),
        )

    def test_evaluator_runs_repeat_gate_before_interventions(self) -> None:
        source = EVALUATOR.read_text()
        predicted = source.index('arm_label="predicted"')
        repeat = source.index('arm_label="predicted_repeat"')
        gate = source.index("_predicted_repeat_gate(", repeat)
        fixed = source.index('arm_label="fixed"')
        shuffled = source.index('arm_label="shuffled"')
        self.assertTrue(predicted < repeat < gate < fixed < shuffled)
        self.assertIn(
            '"one process, one loaded policy, one loaded environment set; "',
            source,
        )

    def test_only_three_scientific_duration_modes_exist(self) -> None:
        module = ast.parse(CONTROLLER.read_text())
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "VALID_MODES"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.literal_eval(assignment.value),
            ("predicted", "fixed", "shuffled"),
        )

    def test_localization_and_immutable_output_contracts_are_present(self) -> None:
        source = EVALUATOR.read_text()
        for field in (
            "initial_raw_observation_sha256",
            "initial_processed_observation_sha256",
            "executed_action_step_sha256",
            "executed_action_prefix_sha256",
            "executed_action_trace_sha256",
        ):
            self.assertIn(field, source)
        self.assertIn("_initial_state_gate(", source)
        self.assertIn("intervention_initial_state_gates", source)
        self.assertIn("atomic_write_json_new(output, result)", source)
        self.assertIn("if output.exists():", source)


if __name__ == "__main__":
    unittest.main()
