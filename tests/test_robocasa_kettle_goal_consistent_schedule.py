from __future__ import annotations

import unittest

import pyarrow as pa

from scripts.data.build_robocasa_kettle_goal_consistent_schedule import transform


class GoalConsistentScheduleTest(unittest.TestCase):
    def test_copies_terminal_burner_into_only_place_rows(self) -> None:
        schema = pa.schema(
            [
                ("episode_index", pa.int64()),
                ("frame_index", pa.int64()),
                ("instruction", pa.string()),
                ("atomic_skill", pa.string()),
                ("stage", pa.string()),
                ("subtask_idx", pa.int64()),
            ]
        )
        rows = [
            {"episode_index": 7, "frame_index": 0, "instruction": "pick up the kettle from the counter", "atomic_skill": "pick", "stage": "a", "subtask_idx": 0},
            {"episode_index": 7, "frame_index": 1, "instruction": "place the kettle on the stove burner", "atomic_skill": "place", "stage": "b", "subtask_idx": 1},
            {"episode_index": 7, "frame_index": 2, "instruction": "turn on the front-left burner where the kettle is placed", "atomic_skill": "turn", "stage": "c", "subtask_idx": 2},
        ]
        source = pa.Table.from_pylist(rows, schema=schema)
        result, goals, changed = transform(source)
        self.assertEqual(goals, {7: "front-left"})
        self.assertEqual(changed, 1)
        self.assertEqual(
            result["instruction"].to_pylist()[1],
            "place the kettle on the front-left stove burner",
        )
        for column in set(source.column_names) - {"instruction"}:
            self.assertTrue(source[column].equals(result[column]))

    def test_requires_one_burner_goal_per_episode(self) -> None:
        table = pa.table(
            {
                "episode_index": [0],
                "frame_index": [0],
                "instruction": ["place the kettle on the stove burner"],
                "atomic_skill": ["place"],
                "stage": ["x"],
                "subtask_idx": [0],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "without a burner-specific"):
            transform(table)


if __name__ == "__main__":
    unittest.main()
