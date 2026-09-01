import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "startup" / "package_tools"))
from enrich_cinel02_coverage import enrich


def test_rate_join_and_conservative_runtime_demand():
    cell = {
        "projectile_z": 1, "projectile_a": 1, "target_z": 1, "target_a": 1,
        "energy_bin_id": 0, "qualified": True,
    }
    report = {"cells": [cell]}
    rates = {(1, 1, 1, 1, 0): 0.01}
    diagnostics = [0] * 1024
    diagnostics[64] = 7
    enriched = enrich(report, rates, {"cinel02_diagnostics": diagnostics})
    assert enriched["rate_join"]["matched_occupied_cells"] == 1
    assert enriched["runtime_demand"]["total_hazards"] == 7
    # A 50-MeV coarse bucket cannot be certified by one qualified 1-MeV cell.
    # The conservative rule nevertheless evaluates every represented positive-rate
    # cell and explicitly records that the reconciliation is not exact.
    assert enriched["runtime_demand"]["exact_1MeV_lookup_reconciliation"] is False
