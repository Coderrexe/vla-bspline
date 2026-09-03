"""Deterministic tests for Molmo language-segment boundary parity.

Run from the repository root with:

    python -m unittest tests.test_molmo_segment_labels
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "molmo_segment_labels.py"
)
SPEC = importlib.util.spec_from_file_location("molmo_segment_labels_test", MODULE_PATH)
molmo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(molmo)


def actions_with_speed(length: int, speed: float = 1.0) -> np.ndarray:
    actions = np.zeros((length, 7), dtype=np.float32)
    actions[:, 0] = speed
    actions[:, 6] = -1.0
    return actions


class MolmoSegmentBoundaryTest(unittest.TestCase):
    def test_toggle_boundary_is_k_exclusive_and_event_starts_next_label(self):
        actions = actions_with_speed(16)
        actions[9:, 6] = 1.0

        segments = molmo.language_segments(actions, 16, 8, 0.15)

        self.assertEqual(segments[0], molmo.LanguageSegment(0, 9, "toggle"))
        self.assertEqual(segments[1].start, 9)
        self.assertEqual(segments[-1].end, len(actions))
        self.assertEqual(
            molmo.v2_segments(actions, 16, 8, 0.15),
            [(0, 9), (9, 16)],
        )

    def test_raw_gripper_value_change_is_a_toggle_even_when_sign_is_unchanged(self):
        actions = actions_with_speed(16)
        actions[9:, 6] = -0.5

        segments = molmo.language_segments(actions, 16, 8, 0.15)

        self.assertEqual(segments[0], molmo.LanguageSegment(0, 9, "toggle"))

    def test_pause_threshold_is_recomputed_from_each_production_window(self):
        actions = actions_with_speed(32, speed=1.0)
        actions[9:11, 0] = 0.1
        actions[16:, 0] = 0.01

        # The first fixed window has median speed 1.0, so 0.1 is below its
        # threshold and the second low action (k=10) is the production event.
        # An episode-global nonzero median is much lower and misses this pause.
        segments = molmo.language_segments(actions, 16, 8, 0.15)

        self.assertEqual(segments[0], molmo.LanguageSegment(0, 10, "pause"))

    def test_cap_tiling_and_short_tail_cover_every_action_once(self):
        actions = actions_with_speed(40)

        segments = molmo.language_segments(actions, 16, 8, 0.15)

        self.assertEqual(
            [(s.start, s.end) for s in segments],
            [(0, 16), (16, 32), (32, 40)],
        )
        self.assertEqual([s.event_type for s in segments], ["cap", "cap", "episode_end"])
        covered = [
            index
            for segment in segments
            for index in range(segment.start, segment.end)
        ]
        self.assertEqual(covered, list(range(len(actions))))

    def test_remainder_shorter_than_min_seg_is_clipped_to_real_episode(self):
        actions = actions_with_speed(19)

        segments = molmo.language_segments(actions, 16, 8, 0.15)

        self.assertEqual(segments[-1], molmo.LanguageSegment(16, 19, "episode_end"))

    def test_input_validation_and_empty_episode(self):
        self.assertEqual(molmo.language_segments(np.zeros((0, 7)), 16, 8, 0.15), [])
        with self.assertRaisesRegex(ValueError, "shape"):
            molmo.language_segments(np.zeros(7), 16, 8, 0.15)
        with self.assertRaisesRegex(ValueError, "at least 7"):
            molmo.language_segments(np.zeros((3, 6)), 16, 8, 0.15)
        with self.assertRaisesRegex(ValueError, "min_seg"):
            molmo.language_segments(np.zeros((3, 7)), 4, 8, 0.15)

    def test_legacy_manifest_is_rejected_before_expensive_labeling(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "segments.jsonl"
            manifest.write_text(json.dumps({"episode_index": 0, "seg_start": 0}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Re-extract into a new workdir"):
                molmo._validate_boundary_records(str(manifest))

    def test_multiview_camera_arguments_are_ordered_and_deduplicated(self):
        args = SimpleNamespace(
            camera=["observation.images.image,observation.images.image2",
                    "observation.images.image"],
        )
        self.assertEqual(
            molmo._selected_cameras(args, "unused"),
            ["observation.images.image", "observation.images.image2"],
        )

    def test_phase_bank_parser_accepts_only_one_in_range_identifier(self):
        self.assertEqual(molmo._parse_phase_id("P03", 12), 3)
        self.assertEqual(molmo._parse_phase_id(" p3. ", 12), 3)
        self.assertIsNone(molmo._parse_phase_id("P03: grasp", 12))
        self.assertIsNone(molmo._parse_phase_id("P12", 12))
        prompt, candidates = molmo._phase_prompt(
            "put both the alphabet soup and the cream cheese box in the basket"
        )
        self.assertIn("P00: move toward the alphabet soup can", prompt)
        self.assertEqual(len(candidates), 12)


if __name__ == "__main__":
    unittest.main()
