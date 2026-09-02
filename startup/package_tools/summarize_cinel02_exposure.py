#!/usr/bin/env python3
"""Summarize the secondary CINEL02 exposure ledger emitted by carbon_mc.

This tool is intentionally descriptive: it never modifies rates, package data,
or a runtime ledger.  It aggregates the 04C isotope x transport-generation x
10-MeV/u-bin arrays into isotope and isotope×generation rows, preserving generation eligibility
and H/O coverage as separate path categories.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SPECIES = [
    "1H", "2H", "3H", "3He", "4He", "6He", "6Li", "7Li", "7Be",
    "9Be", "10Be", "8B", "10B", "11B", "10C", "11C", "12C", "6Be",
]
SUM_METRICS = [
    "path_mm_total", "path_mm_generation_eligible", "path_mm_generation_blocked",
    "path_mm_rate_covered", "path_mm_rate_uncovered", "path_mm_h_uncovered",
    "path_mm_o_uncovered", "hazard_h", "hazard_o", "hazard_total",
    "hazard_blocked_h", "hazard_blocked_o", "hazard_blocked_total",
    "path_mm_continuous_rate_covered", "path_mm_continuous_rate_uncovered",
    "hazard_continuous", "hazard_blocked_continuous", "stopping_loss_MeV",
]
COUNT_METRICS = [
    "collision_candidates", "replay_valid", "parent_killed", "parent_continued",
]


def summarize(ledger: dict[str, Any]) -> dict[str, Any]:
    layout = ledger.get("cinel02_secondary_exposure_layout")
    if not isinstance(layout, dict) or layout.get("shape") != [18, 3, 40]:
        raise ValueError("ledger is missing 04C exposure shape [18,3,40]")
    sums = ledger.get("cinel02_secondary_exposure_sums")
    counts = ledger.get("cinel02_secondary_exposure_counts")
    if not isinstance(sums, list) or len(sums) != 18 * 3 * 40 * len(SUM_METRICS):
        raise ValueError("unexpected 04C exposure sum array length")
    if not isinstance(counts, list) or len(counts) != 18 * 3 * 40 * len(COUNT_METRICS):
        raise ValueError("unexpected 04A exposure count array length")

    rows: list[dict[str, Any]] = []
    for species_index, species in enumerate(SPECIES):
        row: dict[str, Any] = {"species": species}
        for metric_index, metric in enumerate(SUM_METRICS):
            value = 0.0
            for generation in range(3):
                for energy_bin in range(40):
                    cell = (species_index * 3 + generation) * 40 + energy_bin
                    value += float(sums[cell * len(SUM_METRICS) + metric_index])
            row[metric] = value
        for metric_index, metric in enumerate(COUNT_METRICS):
            value = 0
            for generation in range(3):
                for energy_bin in range(40):
                    cell = (species_index * 3 + generation) * 40 + energy_bin
                    value += int(counts[cell * len(COUNT_METRICS) + metric_index])
            row[metric] = value
        eligible = row["path_mm_generation_eligible"]
        total = row["path_mm_total"]
        row["generation_blocked_fraction"] = (
            row["path_mm_generation_blocked"] / total if total > 0.0 else 0.0
        )
        row["rate_coverage_fraction_of_eligible"] = (
            row["path_mm_rate_covered"] / eligible if eligible > 0.0 else 0.0
        )
        tau_runtime = row["hazard_total"]
        tau_blocked = row["hazard_blocked_total"]
        row["tau_runtime"] = tau_runtime
        row["tau_blocked_counterfactual"] = tau_blocked
        row["blocked_hazard_fraction"] = (
            tau_blocked / (tau_runtime + tau_blocked)
            if tau_runtime + tau_blocked > 0.0 else 0.0
        )
        row["candidate_minus_tau_runtime"] = row["collision_candidates"] - tau_runtime
        row["candidate_to_tau_runtime"] = (
            row["collision_candidates"] / tau_runtime if tau_runtime > 0.0 else None
        )
        row["candidate_z_score"] = (
            row["candidate_minus_tau_runtime"] / math.sqrt(tau_runtime)
            if tau_runtime > 0.0 else None
        )
        tau_continuous = row["hazard_continuous"]
        row["tau_continuous"] = tau_continuous
        row["tau_blocked_continuous"] = row["hazard_blocked_continuous"]
        row["tau_runtime_minus_continuous"] = tau_runtime - tau_continuous
        row["tau_continuous_to_runtime"] = (
            tau_continuous / tau_runtime if tau_runtime > 0.0 else None
        )
        row["continuous_rate_coverage_fraction_of_total"] = (
            row["path_mm_continuous_rate_covered"] / total if total > 0.0 else 0.0
        )
        row["stopping_residence_mm_per_MeV"] = (
            total / row["stopping_loss_MeV"]
            if row["stopping_loss_MeV"] > 0.0 else None
        )
        rows.append(row)
    generation_rows: list[dict[str, Any]] = []
    for species_index, species in enumerate(SPECIES):
        for generation in range(3):
            row: dict[str, Any] = {"species": species, "transport_generation": generation}
            for metric_index, metric in enumerate(SUM_METRICS):
                value = 0.0
                for energy_bin in range(40):
                    cell = (species_index * 3 + generation) * 40 + energy_bin
                    value += float(sums[cell * len(SUM_METRICS) + metric_index])
                row[metric] = value
            for metric_index, metric in enumerate(COUNT_METRICS):
                value = 0
                for energy_bin in range(40):
                    cell = (species_index * 3 + generation) * 40 + energy_bin
                    value += int(counts[cell * len(COUNT_METRICS) + metric_index])
                row[metric] = value
            total = row["path_mm_total"]
            eligible = row["path_mm_generation_eligible"]
            tau_runtime = row["hazard_total"]
            tau_blocked = row["hazard_blocked_total"]
            row["generation_blocked_fraction"] = (
                row["path_mm_generation_blocked"] / total if total > 0.0 else 0.0
            )
            row["rate_coverage_fraction_of_eligible"] = (
                row["path_mm_rate_covered"] / eligible if eligible > 0.0 else 0.0
            )
            row["tau_runtime"] = tau_runtime
            row["tau_blocked_counterfactual"] = tau_blocked
            row["blocked_hazard_fraction"] = (
                tau_blocked / (tau_runtime + tau_blocked)
                if tau_runtime + tau_blocked > 0.0 else 0.0
            )
            row["candidate_minus_tau_runtime"] = row["collision_candidates"] - tau_runtime
            row["candidate_to_tau_runtime"] = (
                row["collision_candidates"] / tau_runtime if tau_runtime > 0.0 else None
            )
            row["candidate_z_score"] = (
                row["candidate_minus_tau_runtime"] / math.sqrt(tau_runtime)
                if tau_runtime > 0.0 else None
            )
            tau_continuous = row["hazard_continuous"]
            row["tau_continuous"] = tau_continuous
            row["tau_blocked_continuous"] = row["hazard_blocked_continuous"]
            row["tau_runtime_minus_continuous"] = tau_runtime - tau_continuous
            row["tau_continuous_to_runtime"] = (
                tau_continuous / tau_runtime if tau_runtime > 0.0 else None
            )
            row["continuous_rate_coverage_fraction_of_total"] = (
                row["path_mm_continuous_rate_covered"] / total if total > 0.0 else 0.0
            )
            row["stopping_residence_mm_per_MeV"] = (
                total / row["stopping_loss_MeV"]
                if row["stopping_loss_MeV"] > 0.0 else None
            )
            generation_rows.append(row)
    return {
        "schema": "cinel02_secondary_exposure_v3",
        "layout": layout,
        "species": rows,
        "species_by_generation": generation_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="energy_ledger.json from carbon_mc")
    parser.add_argument("-o", "--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    with args.ledger.open() as stream:
        report = summarize(json.load(stream))
    payload = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
