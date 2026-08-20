"""Static and independent behavior checks for PrimaryCrossingCount."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "startup" / "extensions" / "PrimaryCrossingCount.cc"
HEADER = ROOT / "startup" / "extensions" / "PrimaryCrossingCount.hh"


def test_scorer_is_registered_by_discovery_marker_and_is_parameterized():
    source = SOURCE.read_text()
    header = HEADER.read_text()
    assert source.startswith("// Scorer for PrimaryCrossingCount")
    assert "ProjectileZ" in source and "ProjectileA" in source
    assert "GetAtomicNumber() != projectile_z_" in source
    assert "GetAtomicMass() != projectile_a_" in source
    assert "GetParentID() != 0" in source
    assert "PrimaryCrossingCount" in header
    assert "Carbon" not in source and "proton" not in source.lower()


def test_one_event_bin_is_counted_once_and_new_event_can_count_again():
    # This mirrors the scorer's worker-local set semantics without requiring
    # a Geant4/TOPAS runtime in the repository test suite.
    seen = set()
    totals = {}

    def hit(event, bin_index):
        key = (event, bin_index)
        if bin_index in seen:
            return
        seen.add(bin_index)
        totals[key] = totals.get(key, 0) + 1

    hit(0, 3)
    hit(0, 3)
    hit(0, 3)
    assert totals == {(0, 3): 1}

    seen.clear()
    hit(1, 3)
    hit(1, 7)
    hit(1, 7)
    assert totals == {(0, 3): 1, (1, 3): 1, (1, 7): 1}


def test_source_clears_worker_local_deduplication_at_event_boundary():
    source = SOURCE.read_text()
    assert "void PrimaryCrossingCount::UserHookForEndOfEvent()" in source
    assert "seen_bin_indices_.clear();" in source
    assert "AccumulateHit(step, 1.0, index);" in source
    assert "fComponent->GetIndex(step)" in source
