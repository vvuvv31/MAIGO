#!/usr/bin/env python3
"""Validate worker-local CINEL02 raw files without compiling a package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cinel02 import validate_raw_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--single-target-z", type=int, default=None)
    parser.add_argument("--single-target-a", type=int, default=None)
    args = parser.parse_args()
    if (args.single_target_z is None) != (args.single_target_a is None):
        parser.error("--single-target-z and --single-target-a must be provided together")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    target = None if args.single_target_z is None else (args.single_target_z, args.single_target_a)
    records = validate_raw_files(args.raw, metadata, single_target=target)
    print(f"CINEL02 validation passed: {len(records)} interactions")
    print(f"direct products: {sum(len(products) for _, products in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
