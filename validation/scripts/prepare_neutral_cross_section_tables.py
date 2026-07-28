#!/usr/bin/env python3
"""Convert deterministic TOPAS neutron/gamma cross sections to GPU CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topas_phsp", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--material-label", required=True)
    args = parser.parse_args()

    rows: list[tuple[int, float, float, float]] = []
    with args.topas_phsp.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(
                    f"{args.topas_phsp}:{line_number}: expected 4 fields"
                )
            pdg_id = int(fields[0])
            energy, cross_section, mean_free_path = map(float, fields[1:])
            if (
                pdg_id not in (22, 2112)
                or energy <= 0.0
                or cross_section < 0.0
                or mean_free_path < 0.0
                or not all(
                    math.isfinite(value)
                    for value in (energy, cross_section, mean_free_path)
                )
            ):
                raise ValueError(
                    f"{args.topas_phsp}:{line_number}: invalid row"
                )
            rows.append((pdg_id, energy, cross_section, mean_free_path))

    species = {pdg_id for pdg_id, *_ in rows}
    if species != {22, 2112} or len(rows) % 2 != 0:
        raise ValueError("Expected rectangular gamma/neutron tables")
    grid_size = len(rows) // 2
    grids: dict[int, list[float]] = {}
    for pdg_id in species:
        grid = [energy for code, energy, *_ in rows if code == pdg_id]
        if len(grid) != grid_size or any(
            right <= left for left, right in zip(grid, grid[1:])
        ):
            raise ValueError(f"Incomplete energy grid for PDG {pdg_id}")
        grids[pdg_id] = grid
    if any(
        not math.isclose(left, right, rel_tol=2.0e-6, abs_tol=1.0e-10)
        for left, right in zip(grids[22], grids[2112], strict=True)
    ):
        raise ValueError("Gamma and neutron energy grids differ")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        stream.write(
            f"# TOPAS 4.2.p3 / Geant4 11.3.2 {args.material_label} "
            "neutral macroscopic total cross sections.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "pdg_id",
                "energy_MeV",
                "macroscopic_total_cross_section_per_mm",
                "mean_free_path_mm",
            )
        )
        for row in rows:
            writer.writerow((row[0], *(f"{value:.12g}" for value in row[1:])))
    print(
        f"Wrote {len(rows)} rows for gamma/neutron on {grid_size} "
        f"energy points to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
