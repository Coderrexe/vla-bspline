"""Focused regression tests for production/offline v2 target parity.

Run with a PyTorch environment from the repository root:

    python -m unittest tests.test_event_targets
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "policy" / "smolvla_spline" / "event_targets.py"
)
SPEC = importlib.util.spec_from_file_location("spline_event_targets", MODULE_PATH)
event_targets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(event_targets)

ROBOCASA_PREP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "robocasa_prep.py"
ROBOCASA_SPEC = importlib.util.spec_from_file_location("robocasa_prep_test", ROBOCASA_PREP_PATH)
robocasa_prep = importlib.util.module_from_spec(ROBOCASA_SPEC)
assert ROBOCASA_SPEC.loader is not None
ROBOCASA_SPEC.loader.exec_module(robocasa_prep)


def legacy_production_targets(actions, pad, pm_bank, bl_bank, pg_bank):
    """The pre-refactor production implementation, frozen as a parity oracle."""

    min_seg, horizon_max, pause_frac = 8, 16, 0.15
    pose_lo, grip_idx = 5, 11
    batch_size = actions.shape[0]
    events = torch.zeros(batch_size, horizon_max, dtype=torch.bool)
    grip = actions[..., grip_idx]
    events[:, 1:] |= grip[:, 1:] != grip[:, :-1]
    speed = actions[..., pose_lo : pose_lo + 6].norm(dim=-1)
    median = torch.quantile(speed, 0.5, dim=1, keepdim=True) + 1e-9
    low = speed < pause_frac * median
    events[:, 1:] |= low[:, 1:] & low[:, :-1]
    events |= pad
    events[:, :min_seg] = False
    has_event = events.any(dim=1)
    first = torch.argmax(events.int(), dim=1)
    duration = torch.where(has_event, first, torch.full_like(first, horizon_max))
    duration = duration.clamp(min_seg, horizon_max)

    pose = actions[..., pose_lo : pose_lo + 6] * (~pad).unsqueeze(-1)
    segment_mask = (
        torch.arange(horizon_max).unsqueeze(0) < duration.unsqueeze(1)
    ).to(pose.dtype)
    pose = pose * segment_mask.unsqueeze(-1)
    path = torch.cat([torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1)
    bank_index = duration - min_seg
    endpoint = path.gather(1, duration.view(-1, 1, 1).expand(-1, 1, 6))
    middle = torch.einsum(
        "bmh,bhd->bmd", pm_bank[bank_index], path - bl_bank[bank_index] * endpoint
    )
    pose_ctrl = torch.cat([torch.zeros_like(endpoint), middle, endpoint], dim=1)
    grip_ctrl = torch.einsum("bnh,bh->bn", pg_bank[bank_index], grip)
    pass_ctrl = torch.einsum("bnh,bhk->bnk", pg_bank[bank_index], actions[..., :5])
    return duration, pose_ctrl, grip_ctrl, pass_ctrl


class EventTargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.banks = event_targets.make_event_operator_banks(
            n_ctrl=8,
            degree=3,
            min_seg=8,
            horizon_max=16,
        )

    def test_refactor_is_bit_exact_to_previous_production_path(self):
        actions = torch.randn(7, 16, 12, dtype=torch.float32)
        actions[..., 11] = -1
        actions[0, 9:, 11] = 1
        actions[1, 11:, 11] = -0.5  # raw value change is a production toggle
        actions[2, 10:12, 5:11] = 0
        pad = torch.zeros(7, 16, dtype=torch.bool)
        pad[3, 13:] = True
        pad[4, 4:] = True  # min_seg masks early end; first eligible pad is index 8

        pm_bank, bl_bank, pg_bank, _ = self.banks
        expected = legacy_production_targets(actions, pad, pm_bank, bl_bank, pg_bank)
        actual = event_targets.build_event_spline_targets(
            actions,
            pad,
            pm_bank=pm_bank,
            bl_bank=bl_bank,
            pg_bank=pg_bank,
            pose_lo=5,
            grip_idx=11,
            pass_dims=(0, 1, 2, 3, 4),
            min_seg=8,
            horizon_max=16,
            pause_frac=0.15,
        )

        self.assertTrue(torch.equal(actual.duration, expected[0]))
        self.assertTrue(torch.equal(actual.pose_ctrl, expected[1]))
        self.assertTrue(torch.equal(actual.grip_ctrl, expected[2]))
        self.assertTrue(torch.equal(actual.pass_ctrl, expected[3]))

    def test_toggle_boundary_is_k_and_event_action_is_excluded(self):
        actions = torch.zeros(1, 16, 12)
        actions[..., 5] = 1
        actions[..., 11] = -1
        actions[:, 9:, 11] = 1
        pad = torch.zeros(1, 16, dtype=torch.bool)
        pm_bank, bl_bank, pg_bank, _ = self.banks
        target = event_targets.build_event_spline_targets(
            actions,
            pad,
            pm_bank=pm_bank,
            bl_bank=bl_bank,
            pg_bank=pg_bank,
            pose_lo=5,
            grip_idx=11,
            min_seg=8,
            horizon_max=16,
            pause_frac=0.15,
        )

        self.assertEqual(target.duration.item(), 9)
        self.assertEqual(target.pose_ctrl[0, -1, 0].item(), 9.0)
        torch.testing.assert_close(target.grip_ctrl, -torch.ones_like(target.grip_ctrl))

    def test_pause_uses_full_window_interpolated_median(self):
        actions = torch.zeros(1, 16, 12)
        actions[..., 5] = 1.0
        actions[:, 10:12, 5] = 0.01
        actions[..., 11] = -1
        duration = event_targets.first_event_indices(
            actions,
            None,
            pose_lo=5,
            grip_idx=11,
            min_seg=8,
            horizon_max=16,
            pause_frac=0.15,
        )
        self.assertEqual(duration.item(), 11)

    def test_no_event_uses_horizon_cap(self):
        actions = torch.zeros(1, 16, 12)
        actions[..., 5] = 1.0
        actions[..., 11] = -1
        self.assertEqual(
            event_targets.first_event_indices(
                actions,
                None,
                pose_lo=5,
                grip_idx=11,
                min_seg=8,
                horizon_max=16,
                pause_frac=0.15,
            ).item(),
            16,
        )
        self.assertEqual(
            event_targets.first_event_types(
                actions,
                None,
                pose_lo=5,
                grip_idx=11,
                min_seg=8,
                horizon_max=16,
                pause_frac=0.15,
            ),
            ["cap"],
        )

    def test_padding_is_zeroed_for_fit_and_ends_at_first_eligible_index(self):
        actions = torch.zeros(1, 16, 12)
        actions[..., 5] = 2.0  # represents edge-repeated padded actions too
        actions[..., 11] = -1
        pad = torch.zeros(1, 16, dtype=torch.bool)
        pad[:, 5:] = True
        pm_bank, bl_bank, pg_bank, _ = self.banks
        target = event_targets.build_event_spline_targets(
            actions,
            pad,
            pm_bank=pm_bank,
            bl_bank=bl_bank,
            pg_bank=pg_bank,
            pose_lo=5,
            grip_idx=11,
            min_seg=8,
            horizon_max=16,
            pause_frac=0.15,
        )

        self.assertEqual(target.duration.item(), 8)
        self.assertEqual(target.pose_ctrl[0, -1, 0].item(), 10.0)
        self.assertEqual(
            event_targets.first_event_types(
                actions,
                pad,
                pose_lo=5,
                grip_idx=11,
                min_seg=8,
                horizon_max=16,
                pause_frac=0.15,
            ),
            ["episode_end"],
        )

    def test_robocasa_stats_integration_obeys_k_exclusive_boundary(self):
        episode = np.zeros((16, 12), dtype=np.float32)
        episode[:, 5] = 1
        episode[:, 11] = -1
        episode[9:, 11] = 1
        result = robocasa_prep.fit_stats(
            [episode],
            n_ctrl=8,
            cap=16,
            min_seg=8,
            collect=True,
            stride=100,
        )

        self.assertEqual(result["n_seg"], 1)
        self.assertEqual(result["toggle_frac"], 1.0)
        self.assertEqual(result["T_mean"], 9.0)
        # Nine pre-toggle x deltas are fitted; the toggling action at k=9 is
        # excluded exactly as it is during policy training.
        self.assertAlmostEqual(result["stats_json"]["pose_ctrl_mean"][-1][0], 9.0)


if __name__ == "__main__":
    unittest.main()
