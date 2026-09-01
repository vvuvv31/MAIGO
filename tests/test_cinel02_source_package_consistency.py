import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import audit_cinel02_source_package_consistency as audit


def test_unmatched_sorted_values_uses_replay_tolerance():
    assert audit._unmatched([1.0, 2.0, 4.0], [0.6, 1.5, 4.49], 0.51) == 0
    assert audit._unmatched([1.0, 3.0], [1.0], 0.51) == 1


def test_species_and_target_key_tables_are_explicit():
    assert (4, 6, "6Be") in audit.SPECIES
    assert audit.TARGET_LABELS[(8, 16)] == "O"
    assert len(audit.KEYS) == 36


def test_exact_query_accepts_node_and_window_boundaries_without_nearest_fallback():
    key = (1, 2, 8, 16)
    package = {
        "nodes_by_key": {key: [(100.0, 0, 1), (200.0, 1, 2)]},
        "node_energies_by_key": {key: [100.0, 200.0]},
        "indices": [0, 1],
        "interactions": [
            (1, 2, 8, 16, 100.0),
            (1, 2, 8, 16, 200.0),
        ],
    }
    assert audit._exact_query(package, key, 100.0, 0.0, 0.0) == 0
    assert audit._exact_query(package, key, 100.0, 0.0, 1.0) == 0
    assert audit._exact_query(package, key, 99.49, 0.51, 0.0) == 0
    assert audit._exact_query(package, key, 200.51, 0.51, 1.0) == 1
    assert audit._exact_query(package, key, 150.0, 0.0, 0.0) is None


def test_exact_query_rejects_key_mismatch_even_when_energy_is_supported():
    key = (1, 2, 1, 1)
    package = {
        "nodes_by_key": {key: [(50.0, 0, 1)]},
        "node_energies_by_key": {key: [50.0]},
        "indices": [0],
        "interactions": [(1, 2, 8, 16, 50.0)],
    }
    assert audit._exact_query(package, key, 50.0, 0.0, 0.0) is None
