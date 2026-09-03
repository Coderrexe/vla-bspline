from pathlib import Path
import importlib.util
import json


PATH = Path(__file__).parents[1] / "scripts/analysis/libero_clause_order_stats.py"
SPEC = importlib.util.spec_from_file_location("libero_clause_order_stats", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(state, condition, first, success):
    return {
        "task_id": 0,
        "state_id": state,
        "condition": condition,
        "first_completed_goal_indices": first,
        "success": success,
    }


def test_exact_binomial_mcnemar():
    assert MODULE.exact_binomial_two_sided(0, 0) == 1.0
    assert MODULE.exact_binomial_two_sided(4, 4) == 0.125
    assert MODULE.exact_binomial_two_sided(2, 4) == 1.0


def test_summary_counts_direction_and_success_separately():
    rows = [
        _row(0, "normal", [0], True),
        _row(0, "reversed", [1], True),
        _row(1, "normal", [1], True),
        _row(1, "reversed", [0], False),
        _row(2, "normal", [], False),
        _row(2, "reversed", [1], True),
    ]
    result = MODULE._summarize(rows)
    assert result["requested_first_flip"]["count"] == 1
    assert result["paired_first_goal_mcnemar"]["directional_normal0_reversed1"] == 1
    assert result["paired_first_goal_mcnemar"]["antidirectional_normal1_reversed0"] == 1
    assert result["final_success_descriptive_noninferiority"]["reversed_minus_normal"] == 0


def test_failed_gate_is_rejected(tmp_path):
    gate = tmp_path / "failed.json"
    gate.write_text(json.dumps({"protocol": MODULE.GATE_PROTOCOL, "passed": False}))
    try:
        MODULE._load_passed_gate(gate, 50)
    except ValueError as error:
        assert "not a passed" in str(error)
    else:
        raise AssertionError("failed replay gate was accepted")
