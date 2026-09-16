#!/usr/bin/env python3
"""Collect exploratory candidate screens into one machine-readable table."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.campaign.glob("screen_*")):
        result = directory / "results.json"
        if not result.exists():
            continue
        measured = [row for row in json.loads(result.read_text()) if not row["warmup"]]
        base = next(row for row in measured if row["kind"] == "base")
        candidate = next(row for row in measured if row["kind"] == "cand")
        gain = {key: 100.0 * (base[key] / candidate[key] - 1.0)
                for key in ("wall_s", "elapsed_s", "primary_s", "secondary_s")}
        rows.append({
            "candidate": directory.name.removeprefix("screen_"),
            "screening_only": True,
            "measured_pairs": 1,
            "gain_percent": gain,
            "audit_identical": candidate["audit_identical"],
            "max_dose_difference_percent_peak": candidate["max_diff_pct_peak"],
            "quality_accepted": candidate["quality"]["accepted"],
            "overflow": candidate["quality"]["queue_overflow_count"],
            "promotable": False,
        })
    output = {"schema": "transport_candidate_screen_v1", "candidates": rows,
              "warning": "Exploratory rejection evidence only; never sufficient for promotion."}
    (args.campaign / "candidate_screen_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
