#!/usr/bin/env python3
"""Convert deterministic TOPAS/G4 ion dE/dx rows into the MAIGO GPU table."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path


def read_carbon_delta(path: Path) -> dict[float, float]:
    result: dict[float, float] = {}
    with path.open(newline="") as stream:
        rows = csv.reader(line for line in stream if not line.lstrip().startswith("#"))
        next(rows)
        for energy, fraction in rows:
            result[float(energy)] = float(fraction)
    return result


def interpolate(values: dict[float, float], energy: float) -> float:
    grid = sorted(values)
    if energy <= grid[0]:
        return values[grid[0]]
    if energy >= grid[-1]:
        return values[grid[-1]]
    upper = bisect.bisect_right(grid, energy)
    lo, hi = grid[upper - 1], grid[upper]
    fraction = (energy - lo) / (hi - lo)
    return values[lo] + fraction * (values[hi] - values[lo])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topas_phsp", type=Path)
    parser.add_argument("carbon_delta", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--carbon-stopping-output", type=Path)
    parser.add_argument("--carbon-delta-output", type=Path)
    args = parser.parse_args()

    rows: list[tuple[int, int, float, float, float, float, float]] = []
    with args.topas_phsp.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"{args.topas_phsp}:{line_number}: expected 7 fields")
            z, a = int(fields[0]), int(fields[1])
            energy, unrestricted, restricted, nuclear, raw_delta = map(
                float, fields[2:]
            )
            if not (z > 0 and a >= z and energy > 0 and unrestricted > 0):
                raise ValueError(f"{args.topas_phsp}:{line_number}: invalid ion table row")
            rows.append(
                (z, a, energy, unrestricted, restricted, nuclear, raw_delta)
            )

    carbon_raw = {
        energy: raw_delta
        for z, a, energy, _, _, _, raw_delta in rows
        if (z, a) == (6, 12)
    }
    carbon_measured = read_carbon_delta(args.carbon_delta)
    species = {(z, a) for z, a, *_ in rows}
    grid_size = len(carbon_raw)
    if len(rows) != len(species) * grid_size:
        raise ValueError("Ion tables do not share one complete rectangular energy grid")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        stream.write(
            "# TOPAS 4.2.p3 / Geant4 11.3.2 Water_75eV ion electronic stopping powers.\n"
        )
        stream.write(
            "# delta_electron_fraction preserves the measured C-12 correction and "
            "uses each ion's G4 restricted/unrestricted ratio relative to C-12.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "atomic_number",
                "mass_number",
                "energy_MeVu",
                "stopping_power_MeV_per_mm",
                "restricted_stopping_power_MeV_per_mm",
                "nuclear_stopping_power_MeV_per_mm",
                "raw_delta_electron_fraction",
                "delta_electron_fraction",
            ]
        )
        for z, a, energy, unrestricted, restricted, nuclear, raw_delta in rows:
            c_raw = carbon_raw[energy]
            measured_delta = interpolate(carbon_measured, energy)
            if c_raw > 1.0e-12:
                corrected_delta = measured_delta * raw_delta / c_raw
            else:
                corrected_delta = measured_delta
            corrected_delta = min(1.0 - 1.0e-12, max(0.0, corrected_delta))
            values = (
                z,
                a,
                f"{energy:.12g}",
                f"{unrestricted:.12g}",
                f"{restricted:.12g}",
                f"{nuclear:.12g}",
                f"{raw_delta:.12g}",
                f"{corrected_delta:.12g}",
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Non-finite ion table output")
            writer.writerow(values)

    if args.carbon_stopping_output is not None:
        args.carbon_stopping_output.parent.mkdir(parents=True, exist_ok=True)
        with args.carbon_stopping_output.open("w", newline="") as stream:
            stream.write(
                "# TOPAS/Geant4 C-12 electronic stopping power in Water_75eV; "
                "0.01+0.1*n MeV/u grid.\n"
            )
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["energy_MeVu", "stopping_power_MeV_per_mm"])
            for z, a, energy, unrestricted, *_ in rows:
                if (z, a) == (6, 12):
                    writer.writerow(
                        [f"{energy:.12g}", f"{unrestricted:.12g}"]
                    )

    if args.carbon_delta_output is not None:
        args.carbon_delta_output.parent.mkdir(parents=True, exist_ok=True)
        with args.carbon_delta_output.open("w", newline="") as stream:
            stream.write(
                "# Interpolated TOPAS C-12 delta-electron fraction on the "
                "0.01+0.1*n MeV/u stopping-power grid.\n"
            )
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["energy_MeVu", "delta_electron_fraction"])
            for energy in sorted(carbon_raw):
                writer.writerow(
                    [
                        f"{energy:.12g}",
                        f"{interpolate(carbon_measured, energy):.12g}",
                    ]
                )

    print(
        f"Wrote {len(rows)} rows for {len(species)} isotopes "
        f"on {grid_size} energy points to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
