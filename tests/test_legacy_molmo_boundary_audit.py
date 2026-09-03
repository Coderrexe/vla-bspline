"""Tests for the legacy-to-production Molmo boundary audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "legacy_molmo_boundary_audit.py"
)
SPEC = importlib.util.spec_from_file_location("legacy_molmo_boundary_audit_test", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def actions_with_speed(length: int, speed: float = 1.0) -> np.ndarray:
    actions = np.zeros((length, 7), dtype=np.float32)
    actions[:, 0] = speed
    actions[:, 6] = -1.0
    return actions


class LegacyMolmoBoundaryAuditTest(unittest.TestCase):
    def test_toggle_exposes_historical_k_plus_one_membership(self):
        actions = actions_with_speed(24)
        actions[9:, 6] = 1.0

        legacy = audit.legacy_segments(actions, 24, 8, 0.15)
        corrected = audit.corrected_segments(actions, 24, 8, 0.15)

        self.assertEqual(legacy[0], audit.Segment(0, 10, "toggle"))
        self.assertEqual(corrected[0], audit.Segment(0, 9, "toggle"))
        metrics = audit.compare_partitions(legacy, corrected)
        self.assertFalse(metrics["exact_partition"])
        self.assertEqual(metrics["legacy_to_corrected_nearest"][0], 1)
        self.assertGreater(metrics["legacy_intervals_requiring_recaption"], 0)

    def test_raw_same_sign_gripper_change_is_corrected_only_toggle(self):
        actions = actions_with_speed(24)
        actions[9:, 6] = -0.5

        legacy = audit.legacy_segments(actions, 24, 8, 0.15)
        corrected = audit.corrected_segments(actions, 24, 8, 0.15)

        self.assertNotEqual(legacy[0].event_type, "toggle")
        self.assertEqual(corrected[0], audit.Segment(0, 9, "toggle"))

    def test_window_quantile_differs_from_episode_global_nonzero_median(self):
        actions = actions_with_speed(48)
        actions[9:11, 0] = 0.1
        actions[24:, 0] = 0.01

        legacy = audit.legacy_segments(actions, 24, 8, 0.15)
        corrected = audit.corrected_segments(actions, 24, 8, 0.15)

        self.assertNotEqual(legacy[0].event_type, "pause")
        self.assertEqual(corrected[0], audit.Segment(0, 10, "pause"))

    def test_exact_partition_has_zero_recaption_and_ambiguity(self):
        partition = [
            audit.Segment(0, 8, "cap"),
            audit.Segment(8, 15, "episode_end"),
        ]
        metrics = audit.compare_partitions(partition, partition)

        self.assertTrue(metrics["exact_partition"])
        self.assertEqual(metrics["exact_reusable_legacy_intervals"], 2)
        self.assertEqual(metrics["legacy_intervals_requiring_recaption"], 0)
        self.assertEqual(
            metrics["legacy_label_frames_outside_dominant_corrected_interval"], 0
        )

    def test_overlap_metrics_capture_crossed_boundaries_and_frame_ambiguity(self):
        legacy = [
            audit.Segment(0, 10, "toggle"),
            audit.Segment(10, 20, "episode_end"),
        ]
        corrected = [
            audit.Segment(0, 8, "toggle"),
            audit.Segment(8, 20, "episode_end"),
        ]
        metrics = audit.compare_partitions(legacy, corrected)

        self.assertEqual(metrics["legacy_intervals_crossing_corrected_boundary"], 1)
        self.assertEqual(metrics["corrected_intervals_crossing_legacy_boundary"], 1)
        # The first legacy interval has two frames outside corrected [0, 8);
        # the second has none outside its dominant corrected [8, 20).
        self.assertEqual(
            metrics["legacy_label_frames_outside_dominant_corrected_interval"], 2
        )

    def test_aggregate_uses_boundary_and_frame_denominators(self):
        exact = audit.compare_partitions(
            [audit.Segment(0, 10, "cap")],
            [audit.Segment(0, 10, "cap")],
        )
        shifted = audit.compare_partitions(
            [audit.Segment(0, 6, "toggle"), audit.Segment(6, 10, "episode_end")],
            [audit.Segment(0, 5, "toggle"), audit.Segment(5, 10, "episode_end")],
        )
        result = audit.aggregate_episode_metrics([exact, shifted])

        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["episodes_with_exact_partition"], 1)
        self.assertEqual(result["legacy_boundaries"], 1)
        self.assertEqual(result["corrected_boundaries"], 1)
        self.assertEqual(result["exact_boundaries"], 0)
        self.assertEqual(
            result["legacy_to_corrected_nearest_boundary_distance"]["median"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
