"""Tests for the zero-copy semantic-caption shuffle control."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "build_language_shuffle_control.py"
)
SPEC = importlib.util.spec_from_file_location("language_shuffle_control_test", MODULE_PATH)
control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(control)


def make_dataset(root: Path) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    strings = ["original zero", "original one", "a", "b", "c", "d", "e", "unused"]
    tasks = pd.DataFrame({"task_index": range(len(strings))}, index=strings)
    tasks.to_parquet(root / "meta" / "tasks.parquet")
    episodes = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "tasks": [["original zero"], ["original one"]],
        }
    )
    episodes.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    # IDs 2/3 support only task 0; IDs 4/5 support both tasks; ID 6 is a
    # task-1 singleton; ID 7 is unused.  Only 2/3 and 4/5 are derangeable.
    data = pd.DataFrame(
        {
            "episode_index": [0] * 8 + [1] * 6,
            "frame_index": list(range(8)) + list(range(6)),
            "task_index": [0, 2, 2, 3, 4, 4, 5, 0, 1, 4, 4, 5, 6, 1],
            "action": [[float(index)] for index in range(14)],
        }
    )
    data.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


class LanguageShuffleControlTest(unittest.TestCase):
    def test_same_support_derangement_preserves_singletons(self):
        supports = {2: (0,), 3: (0,), 4: (0, 1), 5: (0, 1), 6: (1,)}
        mapping, _ = control.make_same_support_derangement(
            supports, list(range(2, 8)), seed=1701
        )
        self.assertEqual(mapping[2], 3)
        self.assertEqual(mapping[3], 2)
        self.assertEqual(mapping[4], 5)
        self.assertEqual(mapping[5], 4)
        self.assertEqual(mapping[6], 6)
        self.assertEqual(mapping[7], 7)

    def test_analysis_reports_exact_coverage_and_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            make_dataset(root)
            result = control.analyze(root, original_task_count=2, seed=1701)

        self.assertEqual(result["appended_frame_count"], 10)
        self.assertEqual(result["deranged_frame_count"], 9)
        self.assertEqual(result["appended_run_count"], 7)
        self.assertEqual(result["deranged_run_count"], 6)
        self.assertEqual(result["fixed_used_label_count"], 1)
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(
            {tuple(pair["original_task_support"]) for pair in result["mapping"]},
            {(0,), (0, 1)},
        )

    def test_overlay_changes_only_task_lookup_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            output = parent / "control"
            make_dataset(source)
            source_tasks_hash = control.sha256_file(source / "meta" / "tasks.parquet")
            source_data_hash = control.sha256_file(
                source / "data" / "chunk-000" / "file-000.parquet"
            )

            control.build_overlay(source, output, original_task_count=2, seed=1701)
            source_table = control.read_task_table(source / "meta" / "tasks.parquet")
            output_table = control.read_task_table(output / "meta" / "tasks.parquet")
            output_data_hash = control.sha256_file(
                output / "data" / "chunk-000" / "file-000.parquet"
            )
            manifest = json.loads(
                (output / "_semantic_shuffle_control" / "manifest.json").read_text()
            )
            overlay_scan = control.analyze(output, 2, 1701)

            self.assertTrue((output / "data").is_symlink())
            self.assertEqual(source_data_hash, output_data_hash)
            self.assertEqual(source_tasks_hash, control.sha256_file(source / "meta" / "tasks.parquet"))
            self.assertNotEqual(source_table.strings, output_table.strings)
            self.assertEqual(output_table.strings[2], source_table.strings[3])
            self.assertEqual(output_table.strings[6], source_table.strings[6])
            source_schedule = control.analyze(source, 2, 1701)["task_index_schedule_sha256"]
            self.assertEqual(manifest["task_index_schedule_sha256"], source_schedule)
            self.assertEqual(overlay_scan["task_index_schedule_sha256"], source_schedule)
            with self.assertRaises(FileExistsError):
                control.build_overlay(source, output, original_task_count=2, seed=1701)


if __name__ == "__main__":
    unittest.main()
