import unittest

from scripts.analysis.two_subgoal_clause_stats import matched_clock_head_interaction


class MatchedClockInteractionTest(unittest.TestCase):
    def test_recovers_seed_level_difference_in_differences(self) -> None:
        # Across every seed A improves on one of four states while C improves
        # on two, so the matched-clock interaction is exactly +25 points.
        seeds = (1000, 1001, 1002)
        keys = [(0, index) for index in range(4)]
        a_original = {seed: dict.fromkeys(keys, False) for seed in seeds}
        a_clause = {seed: {key: index == 0 for index, key in enumerate(keys)} for seed in seeds}
        c_original = {seed: dict.fromkeys(keys, False) for seed in seeds}
        c_clause = {seed: {key: index < 2 for index, key in enumerate(keys)} for seed in seeds}
        result = matched_clock_head_interaction(a_original, a_clause, c_original, c_clause)
        self.assertEqual(result["n_training_seeds"], 3)
        self.assertAlmostEqual(result["mean_interaction"], 0.25)
        self.assertEqual(result["training_seed_t_95_ci_df2"], [0.25, 0.25])
        self.assertTrue(all(row["c_minus_a_language_interaction"] == 0.25 for row in result["per_seed"]))

    def test_rejects_mismatched_state_grid(self) -> None:
        seeds = (1000, 1001, 1002)
        complete = {seed: {(0, 0): False} for seed in seeds}
        incomplete = {seed: {(0, 0): False} for seed in seeds}
        incomplete[1002] = {(0, 1): False}
        with self.assertRaisesRegex(ValueError, "state grid differs"):
            matched_clock_head_interaction(complete, complete, complete, incomplete)


if __name__ == "__main__":
    unittest.main()
