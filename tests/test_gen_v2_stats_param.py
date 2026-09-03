from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

lerobot_io = types.ModuleType("lerobot_io")
lerobot_io.episode_arrays = lambda _episode: None
lerobot_io.episode_indices = lambda: np.asarray([], dtype=np.int64)
sys.modules.setdefault("lerobot_io", lerobot_io)
math_module = types.ModuleType("validate_spline_head_math")
math_module.basis_matrix = lambda _u, _n: None
sys.modules.setdefault("validate_spline_head_math", math_module)

from scripts.data.gen_v2_stats_param import selected_episode_indices


class StatsEpisodeSelectionTest(unittest.TestCase):
    def test_selects_exact_provenance_episode_set(self):
        payload = {"annotations": [{"episode_index": 9}, {"episode_index": 2}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(json.dumps(payload))
            selected, record = selected_episode_indices(str(path), np.asarray([2, 4, 9]))
        np.testing.assert_array_equal(selected, np.asarray([2, 9]))
        self.assertEqual(record["count"], 2)
        self.assertEqual(record["episode_indices"], [2, 9])

    def test_rejects_held_out_or_unknown_episode(self):
        payload = {"annotations": [{"episode_index": 7}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "absent"):
                selected_episode_indices(str(path), np.asarray([1, 2]))


if __name__ == "__main__":
    unittest.main()
