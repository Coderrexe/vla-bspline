from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.data.build_two_subgoal_clause_dataset import (
    BOUNDARY_VERSION,
    TARGETS,
    first_valid_release,
    load_targets,
    _numeric_stats,
)


class TwoSubgoalClauseDatasetTest(unittest.TestCase):
    def test_skips_short_failed_grasp_and_switches_after_release(self):
        gripper = np.asarray(
            [-1] * 4 + [1] * 7 + [-1] * 5 + [1] * 48 + [-1] * 10,
            dtype=np.float32,
        )
        event = first_valid_release(gripper, min_closed_run=48)
        self.assertEqual(event["release_action_index"], 64)
        self.assertEqual(event["switch_frame"], 65)
        self.assertEqual(event["closed_run_actions"], 48)
        self.assertEqual([row["valid"] for row in event["release_candidates"]], [False, True])

    def test_requires_a_sustained_release(self):
        with self.assertRaisesRegex(ValueError, "no close->open release"):
            first_valid_release(np.asarray([-1] * 3 + [1] * 5 + [-1] * 3), 8)

    def test_target_map_uses_true_suite_text(self):
        self.assertEqual(
            TARGETS[0]["evaluator_task"],
            "put both the alphabet soup and the tomato sauce in the basket",
        )
        self.assertIn("white mug", TARGETS[4]["evaluator_task"])
        self.assertEqual(BOUNDARY_VERSION, "causal_first_sustained_release_v1")

    def test_external_target_spec_is_validated(self):
        payload = {
            "targets": {
                "0": {
                    "evaluator_task": "task zero",
                    "clauses": ["do zero a", "do zero b"],
                },
                "1": {
                    "evaluator_task": "task one",
                    "clauses": ["do one a", "do one b"],
                },
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(payload))
            targets = load_targets(str(path))
        self.assertEqual(sorted(targets), [0, 1])
        self.assertEqual(targets[1]["clauses"][1], "do one b")

    def test_external_target_spec_rejects_duplicate_clauses(self):
        payload = {
            "targets": {
                "0": {"evaluator_task": "task zero", "clauses": ["same", "a"]},
                "1": {"evaluator_task": "task one", "clauses": ["same", "b"]},
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_targets(str(path))

    def test_numeric_stats_are_population_stats(self):
        values = np.asarray([[0.0, 1.0], [2.0, 5.0]])
        stats = _numeric_stats(values)
        self.assertEqual(stats["count"], [2])
        self.assertEqual(stats["mean"], [1.0, 3.0])
        self.assertEqual(stats["std"], [1.0, 2.0])
        self.assertEqual(stats["q50"], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
