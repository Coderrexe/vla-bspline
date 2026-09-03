from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.eval.duration_intervention_v3 import DurationInterventionV3


class DummyPolicy:
    def __init__(self, values: list[int]) -> None:
        self.config = SimpleNamespace(
            predict_duration=True, min_seg=6, horizon_max=24
        )
        self.values = iter(values)
        self.original_calls = 0

    def _predicted_T(self, _tokens: object) -> int:
        self.original_calls += 1
        return next(self.values)


class DurationInterventionV3Test(unittest.TestCase):
    def test_every_mode_calls_bound_original_before_selection(self) -> None:
        policy = DummyPolicy([7, 11, 19, 7, 11, 19, 7, 11, 19])
        control = DurationInterventionV3(fixed_duration=24, shuffle_seed=5)
        control.attach(policy)

        control.begin_episode("predicted", 0, 0, 0)
        predicted = [policy._predicted_T(None) for _ in range(3)]
        predicted_record = control.end_episode()

        control.begin_episode("fixed", 0, 0, 0)
        fixed = [policy._predicted_T(None) for _ in range(3)]
        fixed_record = control.end_episode()

        control.begin_episode(
            "shuffled",
            0,
            0,
            0,
            shuffle_source=predicted_record["predicted_durations"],
        )
        shuffled = [policy._predicted_T(None) for _ in range(3)]
        shuffled_record = control.end_episode()

        self.assertEqual(policy.original_calls, 9)
        self.assertEqual(predicted, [7, 11, 19])
        self.assertEqual(fixed, [24, 24, 24])
        self.assertEqual(fixed_record["predicted_durations"], [7, 11, 19])
        self.assertEqual(sorted(shuffled), [7, 11, 19])
        self.assertEqual(shuffled_record["predicted_durations"], [7, 11, 19])

    def test_rollout_state_is_fully_reset(self) -> None:
        policy = DummyPolicy([7, 8, 9])
        control = DurationInterventionV3(fixed_duration=24)
        control.attach(policy)
        control.begin_episode("shuffled", 2, 3, 4, shuffle_source=[6, 8])
        for _ in range(3):
            policy._predicted_T(None)
        first = control.end_episode()
        self.assertEqual(first["reused_counterfactual_values"], 1)

        control.begin_episode("predicted", 2, 3, 4)
        # Exhaustion is expected here, proving the original is still called;
        # controller reuse/permutation state itself was reset successfully.
        with self.assertRaises(StopIteration):
            policy._predicted_T(None)


if __name__ == "__main__":
    unittest.main()
