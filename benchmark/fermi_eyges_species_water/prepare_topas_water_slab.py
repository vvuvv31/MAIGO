#!/usr/bin/env python3
"""Prepare the one-shot p/d/t/He-4 pure-water FE calibration matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path("/mnt/sda/wuwei/fe_species_water_em10gev_20260919")
HISTORIES = 250_000
SPECIES = {
    "p": ("proton", 1, 1, 2212),
    "d": ("deuteron", 1, 2, 1000010020),
    "t": ("triton", 1, 3, 1000010030),
    "he4": ("alpha", 2, 4, 1000020040),
}
ENERGIES_MEVU = (50, 150, 300)
THICKNESSES_MM = (1, 10)


def render(template: str, replacements: dict[str, str]) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(f"@@{key}@@", value)
    if "@@" in result:
        raise RuntimeError("Unresolved TOPAS template token")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    template_path = HERE / "topas_water_slab.txt.in"
    template = template_path.read_text()
    cases = []
    for species_index, (tag, (particle, z, a, pdg)) in enumerate(SPECIES.items()):
        for energy_index, energy_mevu in enumerate(ENERGIES_MEVU):
            for thickness_index, thickness_mm in enumerate(THICKNESSES_MM):
                name = f"{tag}_e{energy_mevu}_l{thickness_mm}"
                case = run_root / name
                (case / "output").mkdir(parents=True, exist_ok=True)
                half = 0.5 * thickness_mm
                seed = 26091900 + species_index * 100 + energy_index * 10 + thickness_index
                text = render(template, {
                    "SEED": str(seed),
                    "HALF_THICKNESS_MM": f"{half:.6f}",
                    "SOURCE_Z_MM": f"{half + 0.015:.6f}",
                    "EXIT_Z_MM": f"{-half - 0.015:.6f}",
                    "PARTICLE": particle,
                    "TOTAL_ENERGY_MEV": f"{energy_mevu * a:.6f}",
                })
                config = case / "run.txt"
                config.write_text(text)
                cases.append({
                    "name": name,
                    "particle": particle,
                    "atomic_number": z,
                    "mass_number": a,
                    "pdg": pdg,
                    "energy_MeVu": energy_mevu,
                    "thickness_mm": thickness_mm,
                    "histories": HISTORIES,
                    "seed": seed,
                    "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                })
    manifest = {
        "purpose": "species-specific FE calibration in G4_WATER",
        "topas_physics": ["g4em-standard_opt4", "g4decay"],
        "production_cut_mm": 0.05,
        "water_max_step_mm": 0.05,
        "cases": cases,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(cases)} cases in {run_root}")


if __name__ == "__main__":
    main()
