#!/usr/bin/env python3
"""Restrict a CINEL02 coverage report to the 17 GPU runtime ion species."""

import argparse
import json
from pathlib import Path


SPECIES = {(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6),
           (3, 6), (3, 7), (4, 7), (4, 9), (4, 10), (5, 8),
           (5, 10), (5, 11), (6, 10), (6, 11), (6, 12)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    report["cells"] = [c for c in report["cells"]
                       if (int(c["projectile_z"]), int(c["projectile_a"])) in SPECIES]
    report["energy_gaps"] = [g for g in report.get("energy_gaps", [])
                             if (int(g["projectile_z"]), int(g["projectile_a"])) in SPECIES]
    report["scope"] = {"projectile_species": [list(v) for v in sorted(SPECIES)],
                       "excluded_nonruntime_projectiles": True}
    report["summary"] = {
        "cell_count": len(report["cells"]),
        "qualified_cell_count": sum(c["qualified"] for c in report["cells"]),
        "unqualified_cell_count": sum(not c["qualified"] for c in report["cells"]),
        "gap_count": len(report["energy_gaps"]),
        "interaction_count": sum(c["raw_event_count"] for c in report["cells"]),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
