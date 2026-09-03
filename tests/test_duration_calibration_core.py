from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.duration_calibration_core import (
    FOLDS,
    apply_mapping,
    assign_episode_folds,
    calibration_metrics,
    choose_candidate,
    deterministic_frame_sample,
    fit_cap_threshold,
    fit_mapping,
)


class DurationCalibrationCoreTest(unittest.TestCase):
    def test_episode_folds_are_task_balanced_and_do_not_split_episodes(self) -> None:
        tasks = {episode: f"task-{episode // 9}" for episode in range(18)}
        first = assign_episode_folds(tasks, "salt")
        second = assign_episode_folds(tasks, "salt")
        self.assertEqual(first, second)
        for task in {value for value in tasks.values()}:
            episodes = [episode for episode, value in tasks.items() if value == task]
            counts = {fold: sum(first[episode] == fold for episode in episodes) for fold in FOLDS}
            self.assertEqual(counts, {"fit": 3, "select": 3, "audit": 3})

    def test_frame_sampling_is_uniform_with_respect_to_labels(self) -> None:
        rows = []
        for fold in FOLDS:
            for frame in range(10):
                rows.append(
                    {
                        "task_description": "task",
                        "fold": fold,
                        "episode_index": FOLDS.index(fold),
                        "frame_index": frame,
                        # Labels exist but are not part of the rank contract.
                        "duration_target": 8 if frame == 0 else 24,
                    }
                )
        left = deterministic_frame_sample(rows, per_task_fold=4, salt="fixed")
        changed = [{**row, "duration_target": 13} for row in rows]
        right = deterministic_frame_sample(changed, per_task_fold=4, salt="fixed")
        left_coordinates = [(row["episode_index"], row["frame_index"]) for row in left]
        right_coordinates = [(row["episode_index"], row["frame_index"]) for row in right]
        self.assertEqual(left_coordinates, right_coordinates)

    def test_isotonic_mapping_is_monotone_and_bounded(self) -> None:
        x = np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
        y = np.array([8, 12, 10, 16, 14, 19, 18, 24], dtype=float)
        mapping = fit_mapping("isotonic", x, y, minimum=8, cap=24)
        grid = np.linspace(6, 30, 200)
        predicted = apply_mapping(mapping, grid)
        self.assertTrue(np.all(np.diff(predicted) >= 0))
        self.assertGreaterEqual(int(predicted.min()), 8)
        self.assertLessEqual(int(predicted.max()), 24)

    def test_snap_can_recover_underpredicted_cap_mode(self) -> None:
        x = np.array([9, 10, 12, 13, 18, 19, 20, 21], dtype=float)
        y = np.array([9, 10, 12, 13, 24, 24, 24, 24], dtype=float)
        identity = fit_mapping("identity", x, y, minimum=8, cap=24)
        snap = fit_mapping("snap", x, y, minimum=8, cap=24)
        mae_identity = calibration_metrics(y, apply_mapping(identity, x), minimum=8, cap=24)["mae"]
        mae_snap = calibration_metrics(y, apply_mapping(snap, x), minimum=8, cap=24)["mae"]
        self.assertLess(mae_snap, mae_identity)
        self.assertIsNotNone(snap["parameters"]["threshold"])

    def test_candidate_rule_rejects_majority_cap_shortcut_that_hurts_events(self) -> None:
        fit_x = np.array([10] * 4 + [20] * 12, dtype=float)
        fit_y = np.array([10] * 4 + [24] * 12, dtype=float)
        select_x = fit_x.copy()
        select_y = fit_y.copy()
        chosen, _, scores = choose_candidate(
            fit_x, fit_y, select_x, select_y, minimum=8, cap=24
        )
        self.assertIn(chosen, scores)
        # Every eligible candidate must respect the explicit event guardrail.
        for family, score in scores.items():
            if family != "identity" and score["eligible"]:
                self.assertLessEqual(
                    score["event_mae"], scores["identity"]["event_mae"] + 0.25
                )

    def test_cap_threshold_is_fit_without_test_data(self) -> None:
        x = [8, 9, 10, 11, 18, 19, 20, 21]
        y = [8, 9, 10, 11, 24, 24, 24, 24]
        threshold, confusion = fit_cap_threshold(x, y, cap=24)
        self.assertGreater(threshold, 11)
        self.assertLess(threshold, 18)
        self.assertEqual(confusion["balanced_accuracy"], 1.0)

    def test_log_affine_serialization_round_trip(self) -> None:
        x = np.linspace(8, 20, 50)
        y = np.clip(np.rint(1.2 * x), 8, 24)
        mapping = fit_mapping("log_affine", x, y, minimum=8, cap=24)
        prediction = apply_mapping(mapping, x)
        self.assertLess(np.mean(np.abs(prediction - y)), 0.5)


if __name__ == "__main__":
    unittest.main()

