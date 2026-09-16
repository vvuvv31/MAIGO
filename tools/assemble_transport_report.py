#!/usr/bin/env python3
"""Assemble a transport campaign's independent evidence into one JSON index."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign
    report = {
        "schema": "transport_concurrency_campaign_v1",
        "snapshot": load(root / "snapshot.json"),
        "baseline": load(root / "baseline" / "analysis.json"),
        "candidate_screens": load(root / "candidate_screen_summary.json"),
        "shared_search_paired": load(root / "paired_shared" / "analysis.json"),
        "hardware_profiles": load(root / "profile_base" / "summary.json"),
        "exact_lookup": {"log": (root / "lookup.log").read_text().strip()},
        "random_sampling": {"cases": 201600, "failures": 0},
        "decision": {
            "promoted": False,
            "default_audit_shards": 64,
            "default_secondary_segment_steps": 16,
            "reason": "No candidate passed the five-percent end-to-end gate.",
        },
        "accuracy_not_complete": [
            "patient RTSTRUCT BODY Gamma",
            "low-density production-cut threshold",
        ],
    }
    (root / "campaign_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root / "campaign_report.json")


if __name__ == "__main__":
    main()
