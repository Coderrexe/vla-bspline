from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.build_event_language_dataset import (
    EXPECTED_BOUNDARY_VERSION,
    choose_granular_episodes,
    read_labels,
    validate_episode_partition,
)


def row(episode: int, start: int, end: int, task: int = 4) -> dict:
    return {
        "episode_index": episode,
        "seg_start": start,
        "seg_end": end,
        "task_index": task,
        "task": "task",
        "label": "move toward object",
        "boundary_version": EXPECTED_BOUNDARY_VERSION,
        "boundary_config": {"horizon_max": 24},
        "label_mode": "phase_bank",
        "prompt_version": "test",
        "model_id": "test/model",
    }


class EventLanguageDatasetTest(unittest.TestCase):
    def test_episode_mix_is_task_stratified_and_reproducible(self):
        rows = []
        for task in (0, 4):
            for episode in range(task * 10, task * 10 + 10):
                rows.append(row(episode, 0, 4, task))
        first = choose_granular_episodes(rows, 0.5, 17)
        second = choose_granular_episodes(rows, 0.5, 17)
        self.assertEqual(first, second)
        self.assertEqual(sum(episode < 10 for episode in first), 5)
        self.assertEqual(sum(episode >= 40 for episode in first), 5)

    def test_partition_requires_exact_coverage(self):
        validate_episode_partition([row(1, 0, 4), row(1, 4, 9)], 9, 1)
        with self.assertRaisesRegex(ValueError, "gap/overlap"):
            validate_episode_partition([row(1, 0, 4), row(1, 5, 9)], 9, 1)
        with self.assertRaisesRegex(ValueError, "labels end"):
            validate_episode_partition([row(1, 0, 4)], 9, 1)

    def test_reader_rejects_duplicates_and_legacy_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            duplicate = row(1, 0, 4)
            path.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_labels(path)
            legacy = row(1, 0, 4)
            legacy["boundary_version"] = "legacy"
            path.write_text(json.dumps(legacy) + "\n")
            with self.assertRaisesRegex(ValueError, "boundary_version"):
                read_labels(path)


if __name__ == "__main__":
    unittest.main()
