#!/usr/bin/env python3
"""Rebin an audited 1-MeV CINEL02 coverage report onto the hybrid-log grid."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cinel02_coverage_grid import bin_index, hybrid_edges


SPECIAL_MINIMUM = {(6, 12): 1000, (1, 1): 1000, (2, 4): 1000}


def rebin(source: dict) -> dict:
    maximum = max(float(c["energy_high_MeV_per_u"]) for c in source["cells"])
    edges = hybrid_edges(maximum)
    groups = defaultdict(list)
    for cell in source["cells"]:
        centre = 0.5 * (float(cell["energy_low_MeV_per_u"]) +
                        float(cell["energy_high_MeV_per_u"]))
        key = (int(cell["projectile_z"]), int(cell["projectile_a"]),
               int(cell["target_z"]), int(cell["target_a"]), bin_index(centre, edges))
        groups[key].append(cell)
    cells = []
    for key, members in sorted(groups.items()):
        zp, ap, zt, at, index = key
        events = sum(int(c["raw_event_count"]) for c in members)
        products = sum(int(c["product_count"]) for c in members)
        minimum = SPECIAL_MINIMUM.get((zp, ap), 500)
        # Captured events have unit weight; preserve the conservative minimum of
        # reported ESS and raw count if a future weighted campaign is supplied.
        effective = min(events, sum(float(c["effective_event_count"]) for c in members))
        rates = [c.get("rate_per_mm") for c in members if c.get("rate_per_mm") is not None]
        rate_variation = None
        if rates and max(rates) > 0.0:
            rate_variation = (max(rates) - min(rates)) / max(rates)
        gates = {"minimum_event_count": events >= minimum,
                 "minimum_effective_event_count": effective >= minimum,
                 "internal_rate_variation_le_10pct": rate_variation is None or rate_variation <= 0.10}
        cells.append({
            "projectile_z": zp, "projectile_a": ap, "target_z": zt, "target_a": at,
            "energy_bin_id": index, "energy_low_MeV_per_u": edges[index],
            "energy_high_MeV_per_u": edges[index + 1], "raw_event_count": events,
            "effective_event_count": effective, "product_count": products,
            "source_1MeV_cell_count": len(members),
            "rate_min_per_mm": min(rates) if rates else None,
            "rate_max_per_mm": max(rates) if rates else None,
            "relative_rate_variation": rate_variation,
            "gates": gates, "qualified": all(gates.values()),
        })
    occupied = set(groups)
    gaps = []
    identities = {(k[0], k[1]) for k in groups}
    for projectile in sorted(identities):
        targets = {(k[2], k[3]) for k in groups if k[:2] == projectile}
        for target in ((1, 1), (8, 16)):
            if target not in targets:
                gaps.append({"kind": "missing_target", "projectile_z": projectile[0],
                             "projectile_a": projectile[1], "target_z": target[0], "target_a": target[1]})
        for target in targets:
            indices = sorted(k[4] for k in groups if k[:4] == (*projectile, *target))
            for index in range(indices[0], indices[-1] + 1):
                if (*projectile, *target, index) not in occupied:
                    gaps.append({"kind": "energy_gap", "projectile_z": projectile[0],
                                 "projectile_a": projectile[1], "target_z": target[0],
                                 "target_a": target[1], "energy_bin_id": index})
    qualified = bool(cells) and all(c["qualified"] for c in cells) and not gaps
    return {
        "format": "CINEL02_HYBRID_COVERAGE_V1", "campaign_uuid": source["campaign_uuid"],
        "structurally_valid": source["structurally_valid"], "qualified": qualified,
        "energy_grid": {"kind": "hybrid-log", "edges_MeV_per_u": edges,
                        "segments": ["0-10:1 MeV/u", "10-50:2 MeV/u", ">=50:5% geometric"]},
        "source_report": {"format": source["format"], "resolution": "1 MeV/u"},
        "thresholds": source["thresholds"], "raw_files": source["raw_files"],
        "cells": cells, "energy_gaps": gaps,
        "summary": {"cell_count": len(cells),
                    "qualified_cell_count": sum(c["qualified"] for c in cells),
                    "unqualified_cell_count": sum(not c["qualified"] for c in cells),
                    "gap_count": len(gaps),
                    "interaction_count": sum(c["raw_event_count"] for c in cells)},
        "limitations": [
            "event count, ESS, product count and rate extrema are exactly aggregated",
            "birth-energy and yield bootstrap moments require direct raw rescan and are not emitted",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = rebin(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
