#!/usr/bin/env python3
"""Standardize TOPAS/Geant4 C-12 stopping-power n-tuple into GPU CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

COLUMNS = (
    "energy_MeVu",
    "total_kinetic_energy_MeV",
    "electronic_stopping_power_MeV_per_mm",
    "total_stopping_power_MeV_per_mm",
    "csda_range_mm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/topas/output/carbon_c12_stopping_power_water.phsp"),
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("validation/topas/output/carbon_c12_stopping_power_water.header"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("validation/topas/output/stopping-power_topas.log"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/stopping_power_water_geant4_11_3_2.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/stopping_power_water_geant4_11_3_2.metadata.json"),
    )
    parser.add_argument(
        "--gpu-column",
        choices=("electronic", "total"),
        default="electronic",
        help="Which dE/dx column becomes stopping_power_MeV_per_mm for GPU",
    )
    parser.add_argument(
        "--material-label",
        default="Water_75eV",
        help="Material name recorded in comments and metadata",
    )
    parser.add_argument(
        "--material-properties",
        type=Path,
        help=(
            "Optional CarbonMaterialPropertiesNtuple .phsp file whose density "
            "and radiation length are added to metadata"
        ),
    )
    parser.add_argument(
        "--material-properties-header",
        type=Path,
        help="Optional header paired with --material-properties",
    )
    args = parser.parse_args()

    for path in (args.input, args.header, args.log):
        if not path.exists():
            raise SystemExit(f"Required TOPAS output not found: {path}")

    rows: list[list[float]] = []
    with args.input.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            values = [float(v) for v in stripped.split()]
            if len(values) != 5:
                raise SystemExit(
                    f"Expected 5 columns at {args.input}:{line_number}, got {len(values)}"
                )
            if not all(math.isfinite(v) for v in values):
                raise SystemExit(f"Non-finite value at {args.input}:{line_number}")
            rows.append(values)

    if not rows:
        raise SystemExit("No stopping-power rows")
    energies = [row[0] for row in rows]
    if any(right <= left for left, right in zip(energies, energies[1:])):
        raise SystemExit("Energy grid must be strictly increasing")

    # GPU loader expects: energy_MeVu,stopping_power_MeV_per_mm
    dedx_index = 2 if args.gpu_column == "electronic" else 3
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        stream.write(
            f"# TOPAS/Geant4 C-12 stopping power in {args.material_label} "
            f"({args.gpu_column} dE/dx)\n"
        )
        stream.write(
            "# Extracted via CarbonStoppingPowerNtuple / G4EmCalculator; "
            "not Bethe-Bloch development table.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["energy_MeVu", "stopping_power_MeV_per_mm"])
        for row in rows:
            writer.writerow([f"{row[0]:.12g}", f"{row[dedx_index]:.12g}"])

    # Full diagnostic table
    full_csv = args.output_csv.with_name(
        args.output_csv.stem + "_full.csv"
    )
    with full_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(f"{value:.12g}" for value in row)

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    topas = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", log_text)
    geant4 = re.search(r"Geant4 version Name:\s+(\S+)", log_text)
    ref = min(rows, key=lambda r: abs(r[0] - 200.0))
    metadata = {
        "dataset": f"C-12 stopping power in {args.material_label}",
        "material": args.material_label,
        "extraction_method": "G4EmCalculator via CarbonStoppingPowerNtuple",
        "gpu_column": args.gpu_column,
        "topas_version": topas.group(1).strip() if topas else None,
        "geant4_version": geant4.group(1) if geant4 else None,
        "energy_grid_MeV_per_u": {
            "minimum": energies[0],
            "maximum": energies[-1],
            "count": len(energies),
        },
        "reference_200MeVu": {
            "electronic_MeV_per_mm": ref[2],
            "total_MeV_per_mm": ref[3],
            "csda_range_mm": ref[4],
        },
        "input_files": {
            "ntuple": {"path": args.input.as_posix(), "sha256": sha256(args.input)},
            "header": {"path": args.header.as_posix(), "sha256": sha256(args.header)},
            "log": {"path": args.log.as_posix(), "sha256": sha256(args.log)},
        },
        "output_csv": args.output_csv.as_posix(),
        "full_csv": full_csv.as_posix(),
        "global_scale_applied": False,
    }
    if args.material_properties is not None:
        if not args.material_properties.exists():
            raise SystemExit(
                f"Material-properties output not found: "
                f"{args.material_properties}"
            )
        properties_lines = [
            line.strip()
            for line in args.material_properties.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(properties_lines) != 1:
            raise SystemExit(
                "Expected exactly one material-properties row, found "
                f"{len(properties_lines)}"
            )
        tokens = properties_lines[0].split()
        if len(tokens) != 4:
            raise SystemExit(
                "Expected material, density, radiation length, and mass "
                "radiation length in material-properties row"
            )
        metadata["material_properties"] = {
            "geant4_name": tokens[0],
            "density_g_per_cm3": float(tokens[1]),
            "radiation_length_mm": float(tokens[2]),
            "mass_radiation_length_g_per_cm2": float(tokens[3]),
            "source": {
                "path": args.material_properties.as_posix(),
                "sha256": sha256(args.material_properties),
            },
        }
        if args.material_properties_header is not None:
            if not args.material_properties_header.exists():
                raise SystemExit(
                    "Material-properties header not found: "
                    f"{args.material_properties_header}"
                )
            metadata["material_properties"]["header"] = {
                "path": args.material_properties_header.as_posix(),
                "sha256": sha256(args.material_properties_header),
            }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {full_csv}")
    print(f"Wrote {args.metadata}")
    print(
        f"At 200 MeV/u: electronic={ref[2]:.6g} MeV/mm, total={ref[3]:.6g} MeV/mm, "
        f"CSDA range={ref[4]:.4g} mm"
    )


if __name__ == "__main__":
    main()
