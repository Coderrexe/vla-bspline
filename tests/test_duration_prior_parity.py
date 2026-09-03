from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path

import torch

from scripts.analysis.duration_prior_parity import (
    compare_window_and_target,
    parse_sample,
)

EVENT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "policy"
    / "smolvla_spline"
    / "event_targets.py"
)
EVENT_SPEC = importlib.util.spec_from_file_location(
    "duration_parity_event_targets", EVENT_SOURCE
)
event_targets = importlib.util.module_from_spec(EVENT_SPEC)
assert EVENT_SPEC.loader is not None
EVENT_SPEC.loader.exec_module(event_targets)


EVENT_KWARGS = {
    "pose_lo": 0,
    "grip_idx": 6,
    "min_seg": 2,
    "horizon_max": 4,
    "pause_frac": 0.15,
}


class DurationPriorParityTest(unittest.TestCase):
    def setUp(self):
        self.action = torch.zeros(4, 7, dtype=torch.float32)
        self.action[:, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.action[:, 6] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        self.pad = torch.tensor([False, False, False, True])

    def test_exact_window_and_event_target_match(self):
        duration = compare_window_and_target(
            direct_action=self.action,
            direct_pad=self.pad,
            lerobot_action=self.action.clone(),
            lerobot_pad=self.pad.clone(),
            first_event_indices=event_targets.first_event_indices,
            event_kwargs=EVENT_KWARGS,
        )
        self.assertEqual(duration, 2)

    def test_action_mismatch_fails_before_event_comparison(self):
        changed = self.action.clone()
        changed[1, 3] = 1.0
        with self.assertRaisesRegex(AssertionError, "action window mismatch"):
            compare_window_and_target(
                direct_action=self.action,
                direct_pad=self.pad,
                lerobot_action=changed,
                lerobot_pad=self.pad,
                first_event_indices=event_targets.first_event_indices,
                event_kwargs=EVENT_KWARGS,
            )

    def test_padding_mismatch_is_fatal(self):
        changed = self.pad.clone()
        changed[-1] = False
        with self.assertRaisesRegex(AssertionError, "padding mismatch"):
            compare_window_and_target(
                direct_action=self.action,
                direct_pad=self.pad,
                lerobot_action=self.action,
                lerobot_pad=changed,
                first_event_indices=event_targets.first_event_indices,
                event_kwargs=EVENT_KWARGS,
            )

    def test_sample_coordinates_are_explicit_and_nonnegative(self):
        self.assertEqual(parse_sample("12:34"), (12, 34))
        for invalid in ("12", "1:2:3", "a:2", "-1:2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_sample(invalid)


if __name__ == "__main__":
    unittest.main()
