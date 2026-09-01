#!/usr/bin/env python3
"""Classify occupied CINEL02 replay misses without changing runtime physics.

The runtime ledger stores occupancy in isotope x target x reaction-generation x
10-MeV/u cells, not individual collision records.  This audit therefore uses
the cell means and labels its classifications as provisional.  It combines the
existing rate/package census with the per-status replay energies to distinguish
rate occupancy gaps, post-EM support-boundary crossings, source/compiler gaps,
and lookup/index anomalies.  Be-6 is intentionally excluded because the
TOPAS-compatible policy makes it non-transportable.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SPECIES = [
    (1, 1, "1H"), (1, 2, "2H"), (1, 3, "3H"), (2, 3, "3He"),
    (2, 4, "4He"), (2, 6, "6He"), (3, 6, "6Li"), (3, 7, "7Li"),
    (4, 7, "7Be"), (4, 9, "9Be"), (4, 10, "10Be"), (5, 8, "8B"),
    (5, 10, "10B"), (5, 11, "11B"), (6, 10, "10C"), (6, 11, "11C"),
    (6, 12, "12C"), (4, 6, "6Be"),
]
TARGETS = [(1, 1, "H"), (8, 16, "O")]
STATUS_NAMES = ("collision_candidate", "replay_valid", "replay_lookup_miss",
                "replay_invalid_event", "post_em_below_cutoff")
STATUS_COUNT = len(STATUS_NAMES)
SPECIES_INDEX = {(z, a): i for i, (z, a, _) in enumerate(SPECIES)}
TOLERANCE_MEVU = 0.51


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _inside(energy: float, intervals: list[list[float]]) -> bool:
    return any(left - 1.0e-9 <= energy <= right + 1.0e-9
               for left, right in intervals)


def _classify(rate_intervals: list[list[float]], package_intervals: list[list[float]],
              rate_energy: float, replay_energy: float) -> str:
    """Classify a miss using one occupied cell's mean energies."""
    rate_at_start = _inside(rate_energy, rate_intervals)
    package_at_replay = _inside(replay_energy, package_intervals)
    rate_at_replay = _inside(replay_energy, rate_intervals)
    package_at_start = _inside(rate_energy, package_intervals)
    if not rate_at_start:
        return "rate_occupancy_or_interpolation_gap"
    if package_at_replay and rate_at_replay:
        return "index_or_lookup_anomaly"
    if not package_at_replay and rate_at_replay:
        return "raw_or_compiler_support_gap"
    if package_at_start and not package_at_replay:
        return "post_em_support_boundary"
    return "replay_support_gap"


def _cell_base(species: int, target: int, generation: int, energy_bin: int) -> int:
    return ((((species * 2) + target) * 3 + generation) * 40 + energy_bin) * STATUS_COUNT


def _finite_or_zero(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def audit(census: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    layout = ledger.get("cinel02_replay_status_layout")
    if not isinstance(layout, dict) or layout.get("shape") != [18, 2, 3, 40, 5]:
        raise ValueError("ledger replay-status shape must be [18,2,3,40,5]")
    counts = ledger.get("cinel02_replay_status_counts")
    rate_energy = ledger.get("cinel02_replay_status_rate_query_energy_MeV")
    replay_energy = ledger.get("cinel02_replay_status_replay_query_energy_MeV")
    loss_energy = ledger.get("cinel02_replay_status_continuous_loss_to_collision_MeV")
    expected = 18 * 2 * 3 * 40 * 5
    if not all(isinstance(values, list) and len(values) == expected
               for values in (counts, rate_energy, replay_energy, loss_energy)):
        raise ValueError("ledger replay-status arrays have an unexpected length")
    census_rows = census.get("rows")
    if not isinstance(census_rows, list):
        raise ValueError("census is missing rows")
    coverage = {}
    for row in census_rows:
        key = (int(row["projectile_z"]), int(row["projectile_a"]),
               int(row["target_z"]), int(row["target_a"]))
        coverage[key] = row

    rows: list[dict[str, Any]] = []
    by_class = defaultdict(lambda: {"miss_count": 0, "miss_replay_energy_MeV": 0.0,
                                    "candidate_count": 0})
    by_species = defaultdict(lambda: {"candidate_count": 0, "miss_count": 0,
                                      "miss_replay_energy_MeV": 0.0,
                                      "candidate_replay_energy_MeV": 0.0,
                                      "classifications": defaultdict(int)})
    for species_index, (z, a, species_name) in enumerate(SPECIES):
        if species_name == "6Be":
            continue
        for target_index, (tz, ta, target_name) in enumerate(TARGETS):
            census_row = coverage.get((z, a, tz, ta), {})
            rate_intervals = census_row.get("rate_nonzero_intervals_MeV_per_u", [])
            package_intervals = census_row.get("package_support_intervals_MeV_per_u", [])
            for generation in range(3):
                for energy_bin in range(40):
                    base = _cell_base(species_index, target_index, generation, energy_bin)
                    candidate = int(counts[base])
                    miss = int(counts[base + 2])
                    if candidate == 0 and miss == 0:
                        continue
                    a_float = float(a)
                    e_rate = _finite_or_zero(rate_energy[base]) / max(1.0, candidate) / a_float
                    e_replay = _finite_or_zero(replay_energy[base]) / max(1.0, candidate) / a_float
                    miss_replay = _finite_or_zero(replay_energy[base + 2])
                    miss_rate = _finite_or_zero(rate_energy[base + 2])
                    miss_loss = _finite_or_zero(loss_energy[base + 2])
                    classification = (_classify(rate_intervals, package_intervals,
                                                 e_rate, e_replay) if miss else "none")
                    miss_fraction = miss / candidate if candidate else 0.0
                    energy_total = _finite_or_zero(replay_energy[base])
                    energy_fraction = miss_replay / energy_total if energy_total > 0.0 else 0.0
                    optical_proxy = (-math.log1p(-miss_fraction)
                                     if 0.0 <= miss_fraction < 1.0 else float("inf"))
                    energy_optical_proxy = (-math.log1p(-energy_fraction)
                                            if 0.0 <= energy_fraction < 1.0 else float("inf"))
                    row = {
                        "isotope": species_name, "projectile_z": z, "projectile_a": a,
                        "target": target_name, "target_z": tz, "target_a": ta,
                        "reaction_generation": generation, "energy_bin": energy_bin,
                        "candidate_count": candidate,
                        "valid_count": int(counts[base + 1]),
                        "lookup_miss_count": miss,
                        "invalid_count": int(counts[base + 3]),
                        "post_em_below_cutoff_count": int(counts[base + 4]),
                        "mean_rate_query_energy_MeV_per_u": e_rate,
                        "mean_replay_query_energy_MeV_per_u": e_replay,
                        "candidate_rate_query_energy_MeV": _finite_or_zero(rate_energy[base]),
                        "candidate_replay_query_energy_MeV": energy_total,
                        "miss_rate_query_energy_MeV": miss_rate,
                        "miss_replay_query_energy_MeV": miss_replay,
                        "miss_continuous_loss_to_collision_MeV": miss_loss,
                        "miss_fraction_count": miss_fraction,
                        "miss_fraction_replay_energy": energy_fraction,
                        "missed_hazard_optical_depth_proxy": optical_proxy,
                        "missed_energy_optical_depth_proxy": energy_optical_proxy,
                        "classification": classification,
                        "classification_basis": "occupied-cell mean; provisional without per-collision energies",
                    }
                    rows.append(row)
                    species_total = by_species[species_name]
                    species_total["candidate_count"] += candidate
                    species_total["miss_count"] += miss
                    species_total["miss_replay_energy_MeV"] += miss_replay
                    species_total["candidate_replay_energy_MeV"] += energy_total
                    if miss:
                        species_total["classifications"][classification] += 1
                        bucket = by_class[classification]
                        bucket["candidate_count"] += candidate
                        bucket["miss_count"] += miss
                        bucket["miss_replay_energy_MeV"] += miss_replay

    # Convert defaultdicts to JSON-safe objects and report the skipped policy.
    species_summary = {}
    for name, value in by_species.items():
        species_summary[name] = {
            **{key: val for key, val in value.items() if key != "classifications"},
            "miss_fraction_count": (value["miss_count"] / value["candidate_count"]
                                     if value["candidate_count"] else 0.0),
            "miss_fraction_replay_energy": (
                value["miss_replay_energy_MeV"] / value["candidate_replay_energy_MeV"]
                if value["candidate_replay_energy_MeV"] > 0.0 else 0.0),
            "classifications": dict(value["classifications"]),
        }
    return {
        "schema": "CINEL02_REPLAY_SUPPORT_OCCUPANCY_AUDIT_V1",
        "tolerance_MeV_per_u": TOLERANCE_MEVU,
        "classification_basis": "cell means from aggregate runtime ledger; provisional",
        "transport_policy": {"6Be": "non_transportable_prompt_decay"},
        "rows": rows,
        "summary": {
            "occupied_cell_count": len(rows),
            "miss_cell_count": sum(1 for row in rows if row["lookup_miss_count"]),
            "candidate_count": sum(row["candidate_count"] for row in rows),
            "lookup_miss_count": sum(row["lookup_miss_count"] for row in rows),
            "miss_fraction_count": (
                sum(row["lookup_miss_count"] for row in rows) /
                sum(row["candidate_count"] for row in rows)
                if rows else 0.0),
            "by_classification": {key: dict(value) for key, value in by_class.items()},
            "by_isotope": species_summary,
            "excluded_isotopes": [{"isotope": "6Be", "policy": "non_transportable_prompt_decay"}],
            "optical_depth_note": "No path-length field exists in the compact ledger; optical-depth values are -log(1-miss fraction) hazard proxies, not physical integrated path optical depths.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    report = audit(_read_json(args.census), _read_json(args.ledger))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(report["rows"][0]) if report["rows"] else []
        with args.csv_output.open("w", encoding="utf-8") as stream:
            if fields:
                stream.write(",".join(fields) + "\n")
                for row in report["rows"]:
                    stream.write(",".join(str(row[field]) for field in fields) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
