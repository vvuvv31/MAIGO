#!/usr/bin/env python3
"""Standardize a TOPAS/Geant4 C-12 inelastic cross-section n-tuple."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


COLUMNS = (
    "energy_MeV_per_u",
    "total_kinetic_energy_MeV",
    "c12_h_inelastic_cross_section_barn",
    "c12_o_inelastic_cross_section_barn",
    "hydrogen_macroscopic_cross_section_per_mm",
    "oxygen_macroscopic_cross_section_per_mm",
    "water_macroscopic_cross_section_per_mm",
    "water_mean_free_path_mm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            values = [float(value) for value in stripped.split()]
            if len(values) != len(COLUMNS):
                raise ValueError(
                    f"Expected {len(COLUMNS)} columns at {path}:{line_number}, got {len(values)}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite value at {path}:{line_number}")
            rows.append(values)
    if not rows:
        raise ValueError(f"No cross-section rows found in {path}")
    return rows


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, object] = {"path": path.as_posix(), "sha256": sha256(path)}
    topas_match = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", text)
    geant4_match = re.search(r"Geant4 version Name:\s+(\S+)", text)
    elapsed_match = re.search(
        r"^\s*Total:\s+User=[^\n]*?Real=([0-9.]+)s", text, flags=re.MULTILINE
    )
    if topas_match:
        result["topas_version"] = topas_match.group(1).strip()
    if geant4_match:
        result["geant4_version"] = geant4_match.group(1)
    if elapsed_match:
        result["elapsed_real_s"] = float(elapsed_match.group(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/topas/output/carbon_c12_h_o_inelastic_cross_sections.phsp"),
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("validation/topas/output/carbon_c12_h_o_inelastic_cross_sections.header"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("validation/topas/output/cross-sections_topas.log"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/c12_inelastic_cross_sections_water_geant4_11_3_2.metadata.json"),
    )
    parser.add_argument("--reference-energy-MeVu", type=float, default=200.0)
    args = parser.parse_args()

    for path in (args.input, args.header, args.log):
        if not path.exists():
            raise SystemExit(f"Required TOPAS output not found: {path}")

    rows = read_rows(args.input)
    energies = [row[0] for row in rows]
    if any(right <= left for left, right in zip(energies, energies[1:])):
        raise SystemExit("Energy grid must be strictly increasing")

    closure = [abs((row[4] + row[5]) - row[6]) for row in rows]
    max_closure = max(closure)
    max_macro = max(row[6] for row in rows)
    tolerance = max(1.0e-10, max_macro * 2.0e-5)
    if max_closure > tolerance:
        raise SystemExit(
            f"H/O macroscopic components do not close to water: {max_closure:.6g} > {tolerance:.6g} 1/mm"
        )

    reference_index = min(
        range(len(rows)), key=lambda index: abs(rows[index][0] - args.reference_energy_MeVu)
    )
    reference = rows[reference_index]
    if abs(reference[0] - args.reference_energy_MeVu) > 1.0e-9:
        raise SystemExit(f"Reference energy {args.reference_energy_MeVu:g} MeV/u is absent")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(f"{value:.12g}" for value in row)

    log_metadata = parse_log(args.log)
    metadata = {
        "dataset": "C-12 inelastic cross sections in Water_75eV",
        "extraction_method": (
            "Direct G4HadronicProcessStore query after TOPAS physics initialization; "
            "not fitted from an attenuation curve"
        ),
        "physics_context": "TOPAS default ion-capable physics from carbon_200MeVu_water.txt",
        "energy_grid_MeV_per_u": {
            "minimum": energies[0],
            "maximum": energies[-1],
            "count": len(energies),
        },
        "reference_point": {
            "energy_MeV_per_u": reference[0],
            "c12_h_inelastic_cross_section_barn": reference[2],
            "c12_o_inelastic_cross_section_barn": reference[3],
            "water_macroscopic_cross_section_per_mm": reference[6],
            "water_mean_free_path_mm": reference[7],
        },
        "maximum_h_o_closure_error_per_mm": max_closure,
        "input_files": {
            "ntuple": {"path": args.input.as_posix(), "sha256": sha256(args.input)},
            "header": {"path": args.header.as_posix(), "sha256": sha256(args.header)},
            "topas_log": log_metadata,
        },
        "output_csv": args.output_csv.as_posix(),
    }
    with args.metadata.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.metadata}")
    print(
        f"At {reference[0]:g} MeV/u: Sigma_water={reference[6]:.8g} 1/mm, "
        f"mean free path={reference[7]:.8g} mm"
    )
    print(f"Maximum H/O closure error: {max_closure:.3e} 1/mm")


if __name__ == "__main__":
    main()
