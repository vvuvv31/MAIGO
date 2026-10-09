#!/usr/bin/env python3
"""Reject carbon reference logs that lack an active GenericIon elastic process."""
import argparse
import json
from pathlib import Path
import re


def inspect_log(path):
    text = path.read_text(errors="replace")
    match = re.search(
        r"Hadronic Processes for GenericIon\s*\n(.*?)(?=\n-{5,}|\Z)",
        text, re.DOTALL,
    )
    block = match.group(1) if match else ""
    processes = re.findall(r"Process:\s*(\S+)", block)
    marker = re.search(
        r"CARBON_REFERENCE_ION_ELASTIC GenericIon process=(\S+) registered=true", text,
    )
    if marker:
        processes.append(marker.group(1))
    accepted = bool(marker or re.search(r"Process:\s*(?:ionElastic|hadElastic)\b", block))
    return {
        "log": str(path), "accepted": accepted,
        "GenericIon_processes": processes,
        "scope": "GenericIon elastic registration only; not dose acceptance or a complete physics audit",
        "reason": "active GenericIon elastic found" if accepted else
        "GenericIon elastic absent/unverified: include G4IonElasticPhysics, exposed as CarbonIonElasticPhysics in extensions/topas/common, and retain its process summary",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [inspect_log(path) for path in args.logs]
    result = {"accepted": all(r["accepted"] for r in records), "records": records}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
