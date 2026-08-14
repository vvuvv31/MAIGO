#!/usr/bin/env python3
"""Compare the legacy density-SPR curve with the TOPAS section HU LUT."""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


CASES = {
    "20022516": "benchmark/ct/20022516/grid/patient_ct.bin",
    "RT06423": "benchmark/ct/RT06423/grid/patient_ct.bin",
    "RT07575": "benchmark/ct/grids/patient_ct_rt07575_edge_corrected.bin",
}


def load_stopping_power(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(encoding="ascii") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("energy_"):
                continue
            energy, stopping = line.split(",")[:2]
            rows.append((float(energy), float(stopping)))
    data = np.asarray(rows)
    return data[:, 0], data[:, 1]


def load_ct(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        _, version, nx, ny, nz = struct.unpack("<5I", stream.read(20))
        stream.read(24)
        count = nx * ny * nz
        density = np.fromfile(stream, dtype="<f4", count=count)
        section = np.fromfile(stream, dtype="u1", count=count)
        section_count = struct.unpack("<I", stream.read(4))[0]
    if version < 2 or section_count != 25:
        raise ValueError(f"expected 25-section CCTG v2/v3: {path}")
    return density, section


def density_spr(
    density: np.ndarray,
    energy: float,
    water: tuple[np.ndarray, np.ndarray],
    air: tuple[np.ndarray, np.ndarray],
    lung: tuple[np.ndarray, np.ndarray],
    bone: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    def factor(table: tuple[np.ndarray, np.ndarray], reference_density: float) -> float:
        return float(np.interp(energy, *table) /
                     (reference_density * np.interp(energy, *water)))

    air_factor = factor(air, 0.00120479)
    lung_factor = factor(lung, 1.04)
    bone_factor = factor(bone, 1.85)
    result = np.ones_like(density, dtype=np.float64)
    low = density <= 0.26
    result[low] = air_factor + ((density[low] - 0.00120479) /
                                (0.26 - 0.00120479)) * (lung_factor - air_factor)
    lung_soft = (density > 0.26) & (density < 0.90)
    result[lung_soft] = lung_factor + ((density[lung_soft] - 0.26) / 0.64) * (
        1.0 - lung_factor)
    bone_soft = (density > 1.20) & (density < 1.85)
    result[bone_soft] = 1.0 + ((density[bone_soft] - 1.20) / 0.65) * (
        bone_factor - 1.0)
    result[density >= 1.85] = bone_factor
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/ct/result/diagnostics/ct_sp_models.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tables = {
        material: load_stopping_power(
            repo / f"data/stopping_power_{material}_geant4_11_3_2.csv")
        for material in ("water", "air", "lung", "bone")
    }
    raw_lut = np.loadtxt(repo / "data/hu_stopping_power_lut_geant4_11_3_2.csv",
                         delimiter=",")
    energies, section_lut = raw_lut[0], raw_lut[1:]
    if section_lut.shape != (25, energies.size):
        raise ValueError(f"invalid TOPAS HU LUT shape: {raw_lut.shape}")

    result = {"definition": {
        "ratio": "TOPAS section mass-SP / legacy density-SPR",
        "density_weighting": "case voxel count within each Schneider section",
    }, "cases": []}
    for case, relative_ct in CASES.items():
        density, section = load_ct(repo / relative_ct)
        case_rows = []
        for energy in (50.0, 100.0, 200.0, 300.0):
            density_values = density_spr(density, energy, **tables)
            energy_index = int(np.abs(energies - energy).argmin())
            topas_values = section_lut[section, energy_index]
            rows = []
            for label, mask in (
                ("lung", section == 1),
                ("soft_tissue", (section >= 2) & (section <= 8)),
                ("bone", section >= 9),
            ):
                valid = mask & np.isfinite(density)
                rows.append({
                    "material": label,
                    "voxels": int(valid.sum()),
                    "mean_density_g_per_cm3": float(density[valid].mean()),
                    "mean_density_spr": float(density_values[valid].mean()),
                    "mean_topas_section_spr": float(topas_values[valid].mean()),
                    "topas_to_density_spr": float(
                        topas_values[valid].sum() / density_values[valid].sum()),
                })
            case_rows.append({"energy_MeVu": energy, "materials": rows})
        result["cases"].append({"case": case, "ct_grid": relative_ct,
                                "energies": case_rows})

    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
