import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import audit_cinel02_replay_support_occupancy as audit


def test_classification_prefers_source_gap_over_boundary():
    assert audit._classify([[10.0, 30.0]], [[10.0, 20.0]], 18.0, 25.0) == (
        "raw_or_compiler_support_gap"
    )
    assert audit._classify([[10.0, 30.0]], [[10.0, 20.0]], 18.0, 8.0) == (
        "post_em_support_boundary"
    )
    assert audit._classify([[10.0, 30.0]], [[10.0, 30.0]], 18.0, 18.0) == (
        "index_or_lookup_anomaly"
    )


def test_audit_excludes_be6_and_reports_energy_weighted_miss():
    counts = [0] * (18 * 2 * 3 * 40 * 5)
    rate_energy = [0.0] * len(counts)
    replay_energy = [0.0] * len(counts)
    loss_energy = [0.0] * len(counts)
    base = audit._cell_base(1, 0, 1, 2)  # 2H, H, G1, 20--30 MeV/u
    counts[base] = 10
    counts[base + 1] = 8
    counts[base + 2] = 2
    rate_energy[base] = 240.0
    replay_energy[base] = 220.0
    rate_energy[base + 2] = 45.0
    replay_energy[base + 2] = 40.0
    loss_energy[base + 2] = 5.0
    census = {"rows": [{
        "projectile_z": 1, "projectile_a": 2,
        "target_z": 1, "target_a": 1,
        "rate_nonzero_intervals_MeV_per_u": [[10.0, 30.0]],
        "package_support_intervals_MeV_per_u": [[10.0, 30.0]],
    }]}
    ledger = {
        "cinel02_replay_status_layout": {"shape": [18, 2, 3, 40, 5]},
        "cinel02_replay_status_counts": counts,
        "cinel02_replay_status_rate_query_energy_MeV": rate_energy,
        "cinel02_replay_status_replay_query_energy_MeV": replay_energy,
        "cinel02_replay_status_continuous_loss_to_collision_MeV": loss_energy,
    }
    report = audit.audit(census, ledger)
    assert report["summary"]["lookup_miss_count"] == 2
    assert report["summary"]["by_isotope"]["2H"]["miss_fraction_count"] == 0.2
    assert report["summary"]["by_isotope"]["2H"]["miss_fraction_replay_energy"] == 40.0 / 220.0
    assert all(row["isotope"] != "6Be" for row in report["rows"])
