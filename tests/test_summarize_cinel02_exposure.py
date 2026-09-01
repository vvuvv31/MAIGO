import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import summarize_cinel02_exposure as summarize


def _ledger():
    sums = [0.0] * (18 * 3 * 40 * 13)
    counts = [0] * (18 * 3 * 40 * 4)
    cell = (8 * 3 + 1) * 40 + 12  # 7Be, generation 1, 120--130 MeV/u
    sums[cell * 13 + 0] = 10.0
    sums[cell * 13 + 1] = 7.0
    sums[cell * 13 + 2] = 3.0
    sums[cell * 13 + 3] = 6.0
    sums[cell * 13 + 4] = 1.0
    sums[cell * 13 + 5] = 1.0
    sums[cell * 13 + 6] = 1.0
    sums[cell * 13 + 7] = 2.0
    sums[cell * 13 + 8] = 3.0
    sums[cell * 13 + 9] = 5.0
    sums[cell * 13 + 10] = 0.5
    sums[cell * 13 + 11] = 1.0
    sums[cell * 13 + 12] = 1.5
    counts[cell * 4 + 0] = 4
    counts[cell * 4 + 1] = 3
    counts[cell * 4 + 2] = 1
    counts[cell * 4 + 3] = 2
    return {
        "cinel02_secondary_exposure_layout": {"shape": [18, 3, 40]},
        "cinel02_secondary_exposure_sums": sums,
        "cinel02_secondary_exposure_counts": counts,
    }


def test_summarize_preserves_eligibility_and_coverage_categories():
    row = summarize.summarize(_ledger())["species"][8]
    assert row["path_mm_total"] == 10.0
    assert row["path_mm_generation_eligible"] == 7.0
    assert row["path_mm_generation_blocked"] == 3.0
    assert row["path_mm_rate_covered"] == 6.0
    assert row["path_mm_rate_uncovered"] == 1.0
    assert row["collision_candidates"] == 4
    assert row["replay_valid"] == 3
    assert row["parent_killed"] == 1
    assert row["parent_continued"] == 2
    assert row["generation_blocked_fraction"] == 0.3
    assert row["hazard_blocked_h"] == 0.5
    assert row["hazard_blocked_o"] == 1.0
    assert row["hazard_blocked_total"] == 1.5
    assert row["blocked_hazard_fraction"] == 1.5 / 6.5
    assert row["candidate_minus_tau_runtime"] == -1.0
    assert row["candidate_to_tau_runtime"] == 0.8
    assert row["rate_coverage_fraction_of_eligible"] == 6.0 / 7.0
    by_generation = summarize.summarize(_ledger())["species_by_generation"]
    generation_row = next(item for item in by_generation if item["species"] == "7Be" and item["transport_generation"] == 1)
    assert generation_row["tau_blocked_counterfactual"] == 1.5
    assert generation_row["collision_candidates"] == 4


def test_summarize_rejects_wrong_shape():
    ledger = _ledger()
    ledger["cinel02_secondary_exposure_layout"]["shape"] = [18, 2, 40]
    try:
        summarize.summarize(ledger)
    except ValueError as exc:
        assert "04A exposure shape" in str(exc)
    else:
        raise AssertionError("wrong exposure shape was accepted")
