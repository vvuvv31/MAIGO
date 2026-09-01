#!/usr/bin/env python3
"""Join CINEL02 coverage cells with rates and coarse GPU demand diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SPECIES = ((1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6),
           (3, 6), (3, 7), (4, 7), (4, 9), (4, 10), (5, 8),
           (5, 10), (5, 11), (6, 10), (6, 11), (6, 12))


def load_rates(path: Path) -> dict[tuple[int, int, int, int, int], float]:
    rates: dict[tuple[int, int, int, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["projectile_z"]), int(row["projectile_a"]),
                   int(row["target_z"]), int(row["target_a"]),
                   math.floor(float(row["energy_MeV_per_u"])))
            value = float(row["macroscopic_cross_section_per_mm"])
            if key in rates:
                raise ValueError(f"duplicate rate cell {key}")
            rates[key] = value
    return rates


def enrich(report: dict, rates: dict, runtime: dict | None = None) -> dict:
    keyed = {}
    for cell in report["cells"]:
        key = tuple(cell[name] for name in
                    ("projectile_z", "projectile_a", "target_z", "target_a", "energy_bin_id"))
        cell["rate_per_mm"] = rates.get(key)
        keyed[key] = cell
    report["rate_join"] = {
        "matched_occupied_cells": sum(c["rate_per_mm"] is not None for c in report["cells"]),
        "missing_occupied_cells": sum(c["rate_per_mm"] is None for c in report["cells"]),
        "positive_rate_occupied_cells": sum((c["rate_per_mm"] or 0.0) > 0.0 for c in report["cells"]),
    }
    if runtime is None:
        return report
    diagnostics = runtime["cinel02_diagnostics"]
    rows = []
    covered_demand = total_demand = 0
    for species_index, projectile in enumerate(SPECIES):
        for target_index, target in enumerate(((1, 1), (8, 16))):
            for generation in range(3):
                for coarse_bin in range(8):
                    slot = 64 + (((species_index * 2 + target_index) * 3 + generation) * 8 + coarse_bin)
                    demand = int(diagnostics[slot])
                    if not demand:
                        continue
                    low = coarse_bin * 50
                    high = 400 if coarse_bin == 7 else low + 50
                    demanded_cells = [keyed.get((*projectile, *target, ebin)) for ebin in range(low, high)]
                    # Conservative: the coarse bucket passes only if every positive-rate
                    # 1-MeV cell represented by the package is statistically qualified.
                    relevant = [c for c in demanded_cells if c is not None and (c["rate_per_mm"] or 0.0) > 0.0]
                    qualified = bool(relevant) and all(c["qualified"] for c in relevant)
                    total_demand += demand
                    if qualified:
                        covered_demand += demand
                    rows.append({"projectile_z": projectile[0], "projectile_a": projectile[1],
                                 "target_z": target[0], "target_a": target[1],
                                 "generation": generation, "energy_low_MeV_per_u": low,
                                 "energy_high_MeV_per_u": high, "hazard_count": demand,
                                 "represented_positive_rate_cells": len(relevant),
                                 "conservatively_qualified": qualified})
    report["runtime_demand"] = {
        "resolution": "50 MeV/u aggregate hazard histogram",
        "exact_1MeV_lookup_reconciliation": False,
        "rows": rows, "total_hazards": total_demand,
        "conservatively_qualified_hazards": covered_demand,
        "conservative_coverage_fraction": covered_demand / total_demand if total_demand else None,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--rates", required=True, type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.coverage.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime.read_text(encoding="utf-8")) if args.runtime else None
    enrich(report, load_rates(args.rates), runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rate_join": report["rate_join"],
                      "runtime_demand": report.get("runtime_demand")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
