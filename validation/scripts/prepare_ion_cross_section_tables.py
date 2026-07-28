#!/usr/bin/env python3
"""Convert deterministic TOPAS/G4 isotope cross sections to a GPU CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topas_phsp", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--material-label", required=True)
    args = parser.parse_args()

    rows: list[tuple[int, int, float, float, float]] = []
    with args.topas_phsp.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"{args.topas_phsp}:{line_number}: expected 5 fields"
                )
            z, a = int(fields[0]), int(fields[1])
            energy, cross_section, mean_free_path = map(float, fields[2:])
            if (
                z <= 0
                or a < z
                or energy <= 0
                or cross_section < 0
                or mean_free_path < 0
                or not all(
                    math.isfinite(value)
                    for value in (energy, cross_section, mean_free_path)
                )
            ):
                raise ValueError(
                    f"{args.topas_phsp}:{line_number}: invalid row"
                )
            rows.append((z, a, energy, cross_section, mean_free_path))

    species = {(z, a) for z, a, *_ in rows}
    if not rows or len(rows) % len(species) != 0:
        raise ValueError("Ion cross sections are not a rectangular table")
    grid_size = len(rows) // len(species)
    for species_id in species:
        grid = [
            energy
            for z, a, energy, *_ in rows
            if (z, a) == species_id
        ]
        if len(grid) != grid_size or any(
            right <= left for left, right in zip(grid, grid[1:])
        ):
            raise ValueError(f"Incomplete grid for Z/A={species_id}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        stream.write(
            f"# TOPAS 4.2.p3 / Geant4 11.3.2 {args.material_label} "
            "ion macroscopic inelastic cross sections.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "atomic_number",
                "mass_number",
                "energy_MeVu",
                "macroscopic_inelastic_cross_section_per_mm",
                "mean_free_path_mm",
            )
        )
        for row in rows:
            writer.writerow(
                (row[0], row[1], *(f"{value:.12g}" for value in row[2:]))
            )
    print(
        f"Wrote {len(rows)} rows for {len(species)} isotopes on "
        f"{grid_size} energy points to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
