from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts/eval/libero_compositional_holdout.py"
SPEC = importlib.util.spec_from_file_location("libero_compositional_holdout", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holdout_replaces_only_second_goal():
    MODULE.validate_source_and_holdout_goals(MODULE.SOURCE_GOALS, MODULE.HELD_OUT_GOALS)


def test_prompt_control_changes_only_piecewise_language():
    assert MODULE._prompt("whole", 0) == MODULE._prompt("whole", 1)
    assert MODULE._prompt("clause", 0) != MODULE._prompt("clause", 1)


def test_first_completed_handles_ties_and_missing():
    assert MODULE._first_completed([None, None]) == []
    assert MODULE._first_completed([4, None]) == [0]
    assert MODULE._first_completed([8, 3]) == [1]
    assert MODULE._first_completed([5, 5]) == [0, 1]


def test_seen_controls_cover_both_training_compositions():
    t0 = MODULE.TASK_VARIANTS["seen_t0"]
    t1 = MODULE.TASK_VARIANTS["seen_t1"]
    assert t0["source_task_id"] == 0
    assert t1["source_task_id"] == 1
    assert t0["source_goals"] == t0["requested_goals"]
    assert t1["source_goals"] == t1["requested_goals"]
    assert MODULE.TASK_VARIANTS["heldout"]["requested_goals"] != t0["requested_goals"]


def test_prompt_uses_selected_variant_without_changing_whole_phase_text():
    for variant in MODULE.TASK_VARIANTS.values():
        whole = [
            MODULE._prompt("whole", index, variant["requested_description"], variant["clauses"])
            for index in (0, 1)
        ]
        clause = [
            MODULE._prompt("clause", index, variant["requested_description"], variant["clauses"])
            for index in (0, 1)
        ]
        assert whole == [variant["requested_description"]] * 2
        assert clause == variant["clauses"]
