from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.eval.duration_intervention import (
    DurationIntervention,
    load_predicted_traces,
    load_task_duration_map,
)


class DummyPolicy:
    def __init__(self, predictions=(7, 11, 19, 23)):
        self.config = SimpleNamespace(predict_duration=True, min_seg=6, horizon_max=24)
        self._values = iter(predictions)

    def _predicted_T(self, _tokens):
        return next(self._values)


class DurationInterventionTest(unittest.TestCase):
    def test_predicted_is_identity_and_traced(self):
        policy = DummyPolicy()
        control = DurationIntervention("predicted")
        control.attach(policy)
        control.begin_episode(2, 4)
        self.assertEqual([policy._predicted_T(None) for _ in range(3)], [7, 11, 19])
        record = control.end_episode()
        self.assertEqual(record["predicted_durations"], [7, 11, 19])
        self.assertEqual(record["executed_durations"], [7, 11, 19])
        self.assertEqual(record["n_intervened_duration_calls"], 0)
        self.assertEqual(record["mean_absolute_duration_intervention"], 0.0)

    def test_fixed_changes_only_executed_duration(self):
        policy = DummyPolicy()
        control = DurationIntervention("fixed", fixed_duration=24)
        control.attach(policy)
        control.begin_episode(0, 0)
        self.assertEqual([policy._predicted_T(None) for _ in range(3)], [24, 24, 24])
        record = control.end_episode()
        self.assertEqual(record["predicted_durations"], [7, 11, 19])
        self.assertEqual(record["executed_durations"], [24, 24, 24])
        self.assertEqual(record["n_intervened_duration_calls"], 3)
        self.assertEqual(record["intervention_call_fraction"], 1.0)

    def test_fixed_task_requires_complete_mapping(self):
        policy = DummyPolicy()
        control = DurationIntervention("fixed_task", task_durations={"task zero": 13})
        control.attach(policy)
        with self.assertRaises(KeyError):
            control.begin_episode(1, 0, "task one")
        control.begin_episode(0, 0, "task zero")
        self.assertEqual(policy._predicted_T(None), 13)

    def test_shuffled_preserves_same_episode_multiset_and_cycles(self):
        source = {(3, 9): [6, 8, 12, 24]}
        policy = DummyPolicy(predictions=(7, 11, 19, 23, 7))
        control = DurationIntervention("shuffled", source_traces=source, shuffle_seed=5)
        control.attach(policy)
        control.begin_episode(3, 9)
        executed = [policy._predicted_T(None) for _ in range(5)]
        record = control.end_episode()
        self.assertCountEqual(executed[:4], source[(3, 9)])
        self.assertTrue(all(left != right for left, right in zip(executed[:4], source[(3, 9)])))
        self.assertEqual(executed[4], executed[0])
        self.assertEqual(record["reused_counterfactual_values"], 1)
        self.assertEqual(record["counterfactual_source_length"], 4)
        self.assertEqual(record["counterfactual_unique_durations"], 4)

    def test_rejects_out_of_range_intervention(self):
        policy = DummyPolicy()
        control = DurationIntervention("fixed", fixed_duration=25)
        control.attach(policy)
        control.begin_episode(0, 0)
        with self.assertRaises(ValueError):
            policy._predicted_T(None)

    def test_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({"durations_by_task": {"0": 12, "1": 24}}))
            self.assertEqual(load_task_duration_map(mapping), {"0": 12, "1": 24})

            result = root / "predicted.json"
            result.write_text(
                json.dumps(
                    {
                        "duration_mode": "predicted",
                        "tasks": [
                            {
                                "task_id": 0,
                                "episodes": [
                                    {"state_id": 2, "predicted_durations": [6, 9, 24]}
                                ],
                            }
                        ],
                    }
                )
            )
            traces, payload = load_predicted_traces(result)
            self.assertEqual(traces, {(0, 2): [6, 9, 24]})
            self.assertEqual(payload["duration_mode"], "predicted")


if __name__ == "__main__":
    unittest.main()
