#!/usr/bin/env python3
"""Convert a GPU water-entrance phase-space CSV into a TPS replay spot file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-source-z-mm", type=float, default=1.0e-4)
    parser.add_argument("--incident-histories", type=int, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.input.open() as stream, args.output.open("w", newline="") as out:
        reader = csv.DictReader(stream)
        writer = csv.writer(out)
        writer.writerow([
            "spot_id", "energy_MeV", "x_mm", "y_mm", "weight",
            "source_x_mm", "source_y_mm", "source_z_mm",
            "direction_x", "direction_y", "direction_z",
            "energy_spread_percent", "sigma_x_mm", "sigma_y_mm",
            "sigma_x_prime", "sigma_y_prime", "correlation_x",
            "correlation_y"])
        for row in reader:
            dz = float(row["direction_z"])
            if dz <= 0.0:
                continue
            n_written += 1
            path = args.gpu_source_z_mm / dz
            writer.writerow([
                n_written, row["kinetic_energy_MeV"], 0, 0, 1,
                float(row["x_mm"]) + path * float(row["direction_x"]),
                float(row["y_mm"]) + path * float(row["direction_y"]),
                args.gpu_source_z_mm,
                row["direction_x"], row["direction_y"], row["direction_z"],
                0, 0, 0, 0, 0, 0, 0])
    print(f"{args.output} particles={n_written} incident={args.incident_histories}")


if __name__ == "__main__":
    main()
