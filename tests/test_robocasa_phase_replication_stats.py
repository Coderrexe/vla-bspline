from __future__ import annotations

from scripts.analysis.robocasa_phase_replication_stats import (
    exact_mcnemar,
    progress_summary,
    summarize_pair,
    wilson,
)


def episode(succeeded: bool, initial: str) -> dict:
    return {
        "extrema": {"success": succeeded},
        "initial_observation_sha256": initial,
    }


def test_pair_summary_uses_only_exact_initials_for_mcnemar():
    original = {
        1: episode(False, "a"),
        2: episode(True, "b"),
        3: episode(True, "different-left"),
    }
    official = {
        1: episode(True, "a"),
        2: episode(False, "b"),
        3: episode(True, "different-right"),
    }
    result = summarize_pair(original, official)
    assert result["original"]["successes"] == 2
    assert result["official"]["successes"] == 2
    assert result["identical_initial_observation_pairs"] == 2
    assert result["identical_initial_subset_contingency"] == {
        "both_fail": 0,
        "original_fail_official_success": 1,
        "original_success_official_fail": 1,
        "both_success": 0,
    }
    assert result["identical_initial_subset_exact_mcnemar_pvalue"] == 1.0


def test_exact_mcnemar_and_wilson_known_values():
    assert exact_mcnemar(16, 4) == 0.01181793212890625
    low, high = wilson(0, 10)
    assert low == 0
    assert 0.27 < high < 0.29


def test_progress_summary_uses_episode_extrema():
    episodes = {
        0: {"extrema": {"water_on": True, "washed_count": 2.0}},
        1: {"extrema": {"water_on": False, "washed_count": 1.0}},
    }
    result = progress_summary("RinseSinkBasin", episodes)
    assert result["ever_water_on"]["episodes"] == 1
    assert result["washed_at_least_one_region"]["rate"] == 1.0
    assert result["washed_at_least_two_regions"]["rate"] == 0.5
