import unittest

import numpy as np

from scripts.eval.robocasa_video import video_frame


class RoboCasaProgressVideoTest(unittest.TestCase):
    def test_prefers_external_camera_and_converts_chw(self) -> None:
        external = np.zeros((1, 3, 8, 10), dtype=np.float32)
        external[:, 0] = 1.0
        wrist = np.full((1, 8, 10, 3), 17, dtype=np.uint8)
        frame = video_frame({"pixels": {"image": external, "wrist": wrist}})
        self.assertEqual(frame.shape, (8, 10, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(np.all(frame[..., 0] == 255))
        self.assertTrue(np.all(frame[..., 1:] == 0))

    def test_rejects_observation_without_camera(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no camera-valued"):
            video_frame({"robot_state": np.zeros((1, 12), dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
