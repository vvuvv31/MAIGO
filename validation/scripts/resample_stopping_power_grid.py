#!/usr/bin/env python3
"""Resample one two-column stopping-power table onto another table's grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#") or line.lower().startswith("energy"):
                continue
            fields = next(csv.reader([line]))
            rows.append((float(fields[0]), float(fields[1])))
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two numeric rows")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path}: energy grid is not strictly increasing")
    return values[:, 0], values[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-reference",
        type=Path,
        help=(
            "Optional same-grid reference material. When supplied, interpolate "
            "the source/reference ratio and multiply it by --reference-grid."
        ),
    )
    parser.add_argument("--reference-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_energy, source_stopping = load(args.source)
    target_energy, target_reference = load(args.reference_grid)
    if args.source_reference is None:
        if (
            target_energy[0] < source_energy[0]
            or target_energy[-1] > source_energy[-1]
        ):
            raise ValueError(
                "reference grid extends outside source table; extrapolation is forbidden"
            )
        result = np.interp(target_energy, source_energy, source_stopping)
        method = "linear interpolation, no extrapolation"
    else:
        reference_energy, source_reference = load(args.source_reference)
        if not np.array_equal(source_energy, reference_energy):
            raise ValueError("source and source-reference grids must match exactly")
        ratio = source_stopping / source_reference
        interpolated_ratio = np.interp(
            target_energy,
            source_energy,
            ratio,
            left=ratio[0],
            right=ratio[-1],
        )
        result = target_reference * interpolated_ratio
        method = (
            f"source/reference ratio interpolation using "
            f"{args.source_reference.as_posix()}, endpoint-clamped, multiplied "
            "by the target reference"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            f"# Resampled from {args.source.as_posix()} onto "
            f"{args.reference_grid.as_posix()}; {method}.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["energy_MeVu", "stopping_power_MeV_per_mm"])
        writer.writerows(
            (f"{energy:.12g}", f"{stopping:.12g}")
            for energy, stopping in zip(target_energy, result, strict=True)
        )
    print(
        f"Wrote {args.output} ({target_energy.size} rows; "
        f"{target_energy[0]:g}--{target_energy[-1]:g} MeV/u)"
    )


if __name__ == "__main__":
    main()
