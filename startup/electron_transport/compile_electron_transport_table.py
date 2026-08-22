#!/usr/bin/env python3
"""Convert ElectronTransportNtuple ASCII output to the strict GPU CSV schema."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


HEADER = (
    "kinetic_energy_MeV",
    "electron_collisional_stopping_power_MeV_per_mm",
    "electron_radiative_stopping_power_MeV_per_mm",
    "electron_total_stopping_power_MeV_per_mm",
    "positron_collisional_stopping_power_MeV_per_mm",
    "positron_radiative_stopping_power_MeV_per_mm",
    "positron_total_stopping_power_MeV_per_mm",
)


def compile_table(input_path: Path, output_path: Path) -> None:
    rows: list[tuple[float, ...]] = []
    for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f"{input_path}:{line_number}: expected 7 fields")
        row = tuple(float(field) for field in fields)
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{input_path}:{line_number}: non-finite value")
        energy, ec, er, et, pc, pr, pt = row
        if energy <= 0 or ec <= 0 or er < 0 or pc <= 0 or pr < 0:
            raise ValueError(f"{input_path}:{line_number}: invalid stopping-power row")
        if not math.isclose(et, ec + er, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{input_path}:{line_number}: inconsistent electron total")
        if not math.isclose(pt, pc + pr, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{input_path}:{line_number}: inconsistent positron total")
        if rows and energy <= rows[-1][0]:
            raise ValueError(f"{input_path}:{line_number}: energy grid is not increasing")
        rows.append(row)
    if len(rows) < 2 or rows[0][0] > 0.001 or rows[-1][0] < 500.0:
        raise ValueError("electron table does not cover 0.001 to 500 MeV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows((f"{value:.12g}" for value in row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compile_table(args.input, args.output)


if __name__ == "__main__":
    main()
