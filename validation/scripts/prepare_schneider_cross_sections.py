#!/usr/bin/env python3
"""Build Geant4-version-normalized Schneider C-12 XS tables.

The material/water ratios come from one Geant4 run. They are multiplied by
the project's authoritative water table, allowing the composition correction
to be used without changing the absolute Geant4 physics version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(rows)
    data = list(reader)
    return (
        np.asarray([float(row["energy_MeV_per_u"]) for row in data]),
        np.asarray([float(row["water_macroscopic_cross_section_per_mm"]) for row in data]),
    )


def load_topas_macro(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"unexpected TOPAS XS table layout: {path}")
    return data[:, 0], data[:, 6]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", default="/tmp/maigo_schneider_xs")
    parser.add_argument("--water-phsp", default="/tmp/maigo_topas_xs_water75.phsp")
    parser.add_argument(
        "--reference-water",
        default="data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv",
    )
    parser.add_argument(
        "--output",
        default="data/c12_inelastic_cross_sections_schneider_geant4_11_3_2.csv",
    )
    parser.add_argument("--sections", type=int, default=25)
    parser.add_argument("--runtime-log", default="")
    args = parser.parse_args()

    water_phsp = Path(args.water_phsp)
    reference_path = Path(args.reference_water)
    energy, reference_water = load_reference_csv(reference_path)
    local_energy, local_water = load_topas_macro(water_phsp)
    if not np.array_equal(energy, local_energy):
        raise SystemExit("TOPAS and reference water energy grids differ")

    values: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    for section in range(args.sections):
        path = Path(f"{args.input_prefix}_{section:02d}.phsp")
        section_energy, section_macro = load_topas_macro(path)
        if not np.array_equal(energy, section_energy):
            raise SystemExit(f"energy grid mismatch in {path}")
        ratio = np.divide(section_macro, local_water, out=np.ones_like(section_macro),
                          where=local_water > 0.0)
        values.append(reference_water * ratio)
        sources.append({"section": section, "path": str(path), "sha256": sha256(path)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["energy_MeV_per_u"] +
                        [f"section_{section:02d}_mass_xs_per_mm_at_1g_cm3"
                         for section in range(args.sections)])
        for row in range(energy.size):
            writer.writerow([f"{energy[row]:.9g}"] +
                            [f"{section_values[row]:.9g}" for section_values in values])

    metadata = {
        "dataset": "C-12 inelastic mass cross sections for 25 Schneider materials",
        "method": (
            "Geant4 11.3.2 Schneider/Water_75eV material ratios multiplied by "
            "the authoritative Geant4 11.3.2 Water_75eV table"
        ),
        "reference_water": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
        },
        "ratio_water": {"path": str(water_phsp), "sha256": sha256(water_phsp)},
        "section_sources": sources,
        "density_g_per_cm3": 1.0,
        "sections": args.sections,
        "energy_samples": int(energy.size),
    }
    if args.runtime_log:
        runtime_log = Path(args.runtime_log)
        text = runtime_log.read_text(encoding="utf-8", errors="replace")
        topas = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", text)
        geant4 = re.search(r"Geant4 version Name:\s*([^\s]+)", text)
        metadata["runtime"] = {
            "log": str(runtime_log),
            "log_sha256": sha256(runtime_log),
            "topas_version": topas.group(1).strip() if topas else None,
            "geant4_version": geant4.group(1) if geant4 else None,
        }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output} ({energy.size} energies x {args.sections} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
