#!/usr/bin/env python3
"""Compile validated CINEL02 worker files into a target-conditioned package."""

from __future__ import annotations

import argparse
from pathlib import Path

from cinel02 import compile_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help="optional CINEL02_QUALIFICATION_V1 report to bind into the package sidecar",
    )
    parser.add_argument("--energy-min-mevu", type=float, default=0.0)
    parser.add_argument("--energy-bin-width-mevu", type=float, default=1.0)
    parser.add_argument("--minimum-events-per-bin", type=int, default=32)
    parser.add_argument(
        "--maximum-unsupported-product-energy-fraction",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--single-target-z", type=int, default=None)
    parser.add_argument("--single-target-a", type=int, default=None)
    args = parser.parse_args()
    if (args.single_target_z is None) != (args.single_target_a is None):
        parser.error("--single-target-z and --single-target-a must be provided together")
    target = None if args.single_target_z is None else (args.single_target_z, args.single_target_a)
    result = compile_package(
        args.raw,
        args.metadata,
        args.output,
        args.output_metadata,
        energy_min_mevu=args.energy_min_mevu,
        energy_bin_width_mevu=args.energy_bin_width_mevu,
        minimum_events_per_bin=args.minimum_events_per_bin,
        maximum_unsupported_product_energy_fraction=(
            args.maximum_unsupported_product_energy_fraction
        ),
        qualification_report_path=args.qualification_report,
        single_target=target,
    )
    print(f"CINPKG03 written: {result['output']['path']}")
    print(f"interactions: {result['records']['interactions']}; products: {result['records']['products']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
