#!/usr/bin/env python3
"""Estimate legacy and streaming CINEL02 compiler memory/scratch requirements."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.sidecar.read_text(encoding="utf-8"))
    records = data["records"]
    interactions = int(records["interactions"])
    products = int(records["products"])
    binary = interactions * 476 + products * 72
    # Conservative CPython dict/list amplification calibrated by object layout,
    # not a promise of exact RSS. It is intended for admission control.
    legacy_low = binary * 5.0
    legacy_high = binary * 10.0
    streaming_rss = min(4 * 1024**3, max(512 * 1024**2, binary * 0.03))
    scratch = binary * 2.2
    result = {
        "interactions": interactions, "products": products,
        "binary_record_bytes": int(binary),
        "legacy_peak_RSS_estimate_bytes": [int(legacy_low), int(legacy_high)],
        "streaming_target_RSS_bytes": int(streaming_rss),
        "streaming_scratch_estimate_bytes": int(scratch),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
