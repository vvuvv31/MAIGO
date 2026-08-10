#!/usr/bin/env python3
"""Split full-plan spot weights into one CSV per carbon energy layer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prepare_ct_prelim_topas import load_combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spots",
        nargs="+",
        type=Path,
        default=[Path("ct/topas/spots_c_01.txt"), Path("ct/topas/spots_c_02.txt")],
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("ct/code/full_plan_weights_exact.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/ct/tuning/energy_layer_weights")
    )
    args = parser.parse_args()

    channels = load_combined(args.spots)
    energies = [value / 12.0 for value in channels[2]]
    weights: list[float] = []
    with args.weights.open(encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if row and row[0].strip():
                weights.append(float(row[0]))
    if len(weights) != len(energies):
        raise SystemExit(f"weights {len(weights)} != spots {len(energies)}")

    layers = sorted(set(energies))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for energy in layers:
        selected = [w if abs(e - energy) < 1.0e-6 else 0.0 for e, w in zip(energies, weights)]
        path = args.output_dir / f"weights_{energy:g}MeVu.csv"
        path.write_text("".join(f"{value:.17g}\n" for value in selected), encoding="ascii")
        summary.append(
            {
                "energy_MeVu": energy,
                "file": str(path),
                "spots": sum(1 for value in selected if value > 0.0),
                "weight_sum": sum(selected),
            }
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
