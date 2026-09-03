from pathlib import Path
import importlib.util


PATH = Path(__file__).parents[1] / "scripts/eval/libero_clause_order_steering.py"
SPEC = importlib.util.spec_from_file_location("libero_clause_order_steering", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_completed_handles_none_and_ties():
    assert MODULE._first_completed([None, None]) == []
    assert MODULE._first_completed([7, None]) == [0]
    assert MODULE._first_completed([9, 4]) == [1]
    assert MODULE._first_completed([4, 4]) == [0, 1]


def test_expected_official_goals_are_locked():
    MODULE._validate_task_goals(
        0,
        [
            ["in", "alphabet_soup_1", "basket_1_contain_region"],
            ["in", "tomato_sauce_1", "basket_1_contain_region"],
        ],
    )
    MODULE._validate_task_goals(
        4,
        [
            ["on", "porcelain_mug_1", "plate_1"],
            ["on", "white_yellow_mug_1", "plate_2"],
        ],
    )
