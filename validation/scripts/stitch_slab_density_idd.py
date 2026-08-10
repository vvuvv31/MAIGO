#!/usr/bin/env python3
"""Stitch three-layer TOPAS slab scorers into a single 0.5 mm depth IDD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_layer(path: Path, value_column: int, histories: float) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(x.strip()) for x in stripped.split(",")])
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"No data in {path}")
    data = np.asarray(rows, dtype=float)
    z_idx = data[:, 2].astype(int)
    values = data[:, value_column] / histories
    order = np.argsort(z_idx)
    return values[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--layer2", type=Path, required=True)
    parser.add_argument("--histories", type=float, required=True)
    parser.add_argument("--value-column", type=int, default=3)
    parser.add_argument("--bin-width-mm", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    l1 = load_layer(args.layer1, args.value_column, args.histories)
    dense = load_layer(args.dense, args.value_column, args.histories)
    l2 = load_layer(args.layer2, args.value_column, args.histories)
    if len(l1) != 100 or len(dense) != 40 or len(l2) != 660:
        raise SystemExit(
            f"Unexpected bin counts: layer1={len(l1)} dense={len(dense)} layer2={len(l2)} "
            "(expect 100/40/660)"
        )
    dose = np.concatenate([l1, dense, l2])
    depth = (np.arange(len(dose)) + 0.5) * args.bin_width_mm
    maximum = float(np.max(dose)) if np.max(dose) > 0 else 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["depth_mm", "energy_deposition_MeV_per_primary", "relative_dose"]
        )
        for z, value in zip(depth, dose, strict=True):
            writer.writerow(
                [f"{z:.12g}", f"{value:.12g}", f"{value / maximum:.12g}"]
            )
    print(f"Wrote {args.output} bins={len(dose)} integral={float(np.trapz(dose, depth)):.6g}")


if __name__ == "__main__":
    main()
