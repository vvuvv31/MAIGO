#!/usr/bin/env python3
"""Convert a voxelized TOPAS CSV scorer to the project's depth-dose schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--histories", type=float, required=True)
    parser.add_argument("--bin-width-mm", type=float, default=0.5)
    parser.add_argument("--z-column", type=int, default=2)
    parser.add_argument("--value-column", type=int, required=True)
    parser.add_argument(
        "--already-per-primary",
        action="store_true",
        help="Do not divide the selected TOPAS value by the number of histories.",
    )
    args = parser.parse_args()

    rows: list[list[float]] = []
    with args.input.open(encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(value.strip()) for value in stripped.split(",")])
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"No numeric scorer rows found in {args.input}")

    data = np.asarray(rows, dtype=float)
    required_column = max(args.z_column, args.value_column)
    if data.ndim != 2 or data.shape[1] <= required_column:
        raise SystemExit(
            f"Requested column {required_column}, but numeric data has shape {data.shape}"
        )

    z_indices = data[:, args.z_column].astype(int)
    values = data[:, args.value_column]
    order = np.argsort(z_indices)
    z_indices = z_indices[order]
    values = values[order]
    if not args.already_per_primary:
        values = values / args.histories
    maximum = float(np.max(values))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["depth_mm", "energy_deposition_MeV_per_primary", "relative_dose"]
        )
        for z_index, value in zip(z_indices, values, strict=True):
            writer.writerow(
                [
                    f"{(z_index + 0.5) * args.bin_width_mm:.12g}",
                    f"{value:.12g}",
                    f"{value / maximum if maximum > 0.0 else 0.0:.12g}",
                ]
            )


if __name__ == "__main__":
    main()

