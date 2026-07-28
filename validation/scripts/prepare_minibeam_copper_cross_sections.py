#!/usr/bin/env python3
"""Standardize the TOPAS/Geant4 C-12 macroscopic inelastic XS in Copper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-energy-MeVu", type=float, default=200.0)
    args = parser.parse_args()

    rows: list[tuple[float, float, float]] = []
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 8 or not all(math.isfinite(value) for value in values):
                raise SystemExit(f"Invalid row at {args.input}:{line_number}")
            # The extension's historical names say "water", but columns 7/8
            # are queried from the scorer component's actual G4Material.
            rows.append((values[0], values[6], values[7]))
    if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise SystemExit("Copper cross-section energy grid is empty or non-monotonic")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "energy_MeV_per_u",
                "c12_copper_macroscopic_inelastic_cross_section_per_mm",
                "c12_copper_inelastic_mean_free_path_mm",
            )
        )
        writer.writerows(tuple(f"{value:.12g}" for value in row) for row in rows)

    log = args.log.read_text(encoding="utf-8", errors="replace")
    topas = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", log)
    geant4 = re.search(r"Geant4 version Name:\s+(\S+)", log)
    reference = min(rows, key=lambda row: abs(row[0] - args.reference_energy_MeVu))
    metadata = {
        "dataset": "C-12 macroscopic inelastic cross section in Copper",
        "extraction_method": (
            "Direct G4HadronicProcessStore::GetInelasticCrossSectionPerVolume "
            "query after TOPAS physics initialization; no attenuation fit"
        ),
        "physics": {
            "topas_version": topas.group(1).strip() if topas else None,
            "geant4_version": geant4.group(1) if geant4 else None,
            "modules": [
                "g4em-standard_opt4",
                "g4h-phy_QGSP_BERT_HP",
                "g4decay",
                "g4ion-binarycascade",
                "g4h-elastic_HP",
                "g4stopping",
            ],
        },
        "energy_grid_MeV_per_u": {
            "minimum": rows[0][0],
            "maximum": rows[-1][0],
            "count": len(rows),
        },
        "reference_point": {
            "energy_MeV_per_u": reference[0],
            "macroscopic_inelastic_cross_section_per_mm": reference[1],
            "mean_free_path_mm": reference[2],
        },
        "inputs": {
            "ntuple": {"path": str(args.input), "sha256": sha256(args.input)},
            "header": {"path": str(args.header), "sha256": sha256(args.header)},
            "log": {"path": str(args.log), "sha256": sha256(args.log)},
        },
        "output_csv": str(args.output_csv),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {args.output_csv}; at {reference[0]:g} MeV/u: "
        f"Sigma={reference[1]:.8g} 1/mm, MFP={reference[2]:.8g} mm"
    )


if __name__ == "__main__":
    main()
