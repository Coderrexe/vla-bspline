from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from scripts.analysis.duration_training_prior import (
    discrete_median,
    future_windows,
    validate_episode_frames,
)


class DurationTrainingPriorTest(unittest.TestCase):
    def test_discrete_median_uses_lower_observed_integer_crossing_half(self):
        self.assertEqual(discrete_median(Counter({8: 2, 12: 1, 24: 1})), 8)
        self.assertEqual(discrete_median(Counter({8: 1, 12: 1, 24: 2})), 12)

    def test_future_windows_repeat_final_action_and_mark_padding(self):
        actions = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
        windows, pad = future_windows(actions, horizon=3)
        np.testing.assert_array_equal(
            windows.numpy(),
            np.asarray(
                [
                    [[0.0], [1.0], [2.0]],
                    [[1.0], [2.0], [2.0]],
                    [[2.0], [2.0], [2.0]],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            pad.numpy(),
            np.asarray(
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, True],
                ]
            ),
        )

    def test_episode_frames_must_be_zero_based_and_contiguous(self):
        validate_episode_frames(17, [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "must start at 0"):
            validate_episode_frames(17, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "expected 2, got 3"):
            validate_episode_frames(17, [0, 1, 3])
        with self.assertRaisesRegex(ValueError, "has no frames"):
            validate_episode_frames(17, [])


if __name__ == "__main__":
    unittest.main()
