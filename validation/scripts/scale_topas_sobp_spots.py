#!/usr/bin/env python3
"""Scale the integer histories in a TOPAS SOBP time-feature file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HISTORY_LINE = re.compile(
    r"^(iv:Tf/Scatterer1/L4/Values\s*=\s*\d+\s+)(.*?)(\s*)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/topas/spots_sobp_water_5_10cm.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/topas/spots_sobp_water_5_10cm_100k.txt"),
    )
    parser.add_argument("--divisor", type=int, default=100)
    args = parser.parse_args()
    if args.divisor <= 0:
        raise SystemExit("--divisor must be positive")

    output_lines: list[str] = []
    found = False
    original_total = 0
    scaled_total = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        match = HISTORY_LINE.match(line)
        if match is None:
            output_lines.append(line.rstrip())
            continue
        values = [int(value) for value in match.group(2).split()]
        if any(value % args.divisor != 0 for value in values):
            raise SystemExit(
                "History weights are not exactly divisible by the requested divisor"
            )
        scaled = [value // args.divisor for value in values]
        original_total = sum(values)
        scaled_total = sum(scaled)
        output_lines.append(
            match.group(1) + " ".join(str(value) for value in scaled)
        )
        found = True

    if not found:
        raise SystemExit("Could not find Tf/Scatterer1/L4/Values")
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {original_total} -> {scaled_total} histories "
        f"(divisor {args.divisor})"
    )


if __name__ == "__main__":
    main()
