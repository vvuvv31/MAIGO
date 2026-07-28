#!/usr/bin/env python3
"""Prepare matched TOPAS/GPU 1D minibeam cases for an energy sweep."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ENERGIES_MEVU = (100, 200, 300, 400)
ORIGIN_CATEGORIES = (
    ("PrimaryC12", "primary_c12"),
    ("SecondaryCarbon", "secondary_carbon"),
    ("Boron", "boron"),
    ("Beryllium", "beryllium"),
    ("Lithium", "lithium"),
    ("Helium", "helium"),
    ("Proton", "proton"),
    ("OtherCharged", "other_charged"),
    ("Neutron", "neutron"),
    ("Gamma", "gamma"),
    ("NeutralOther", "neutral_other"),
    ("Unresolved", "unresolved"),
)


def replace_yaml_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*:.*$")
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count != 1:
        raise ValueError(f"expected one YAML key {key!r}, found {count}")
    return updated


def replace_spot_energy(text: str, total_energy_mev: int) -> str:
    pattern = re.compile(
        r"(?m)^(dv:Tf/Scatterer1/L[12]/Values\s*=\s*1\s+)"
        r"[0-9.eE+-]+(\s+MeV\s*)$"
    )
    updated, count = pattern.subn(
        rf"\g<1>{total_energy_mev}\g<2>", text
    )
    if count != 2:
        raise ValueError(f"expected two spot energy rows, found {count}")
    return updated


def replace_spot_histories(text: str, histories: int) -> str:
    pattern = re.compile(
        r"(?m)^(iv:Tf/Scatterer1/L4/Values\s*=\s*1\s+)"
        r"\d+(\s*)$"
    )
    updated, count = pattern.subn(rf"\g<1>{histories}\g<2>", text)
    if count != 1:
        raise ValueError(f"expected one spot history row, found {count}")
    return updated


def topas_case(energy_mevu: int, histories: int, depth_mm: float) -> str:
    total_energy_mev = 12 * energy_mevu
    bins = round(depth_mm / 0.1)
    half_depth = 0.5 * depth_mm
    water_center_world_y = 60.0 + half_depth
    stem = f"minibeam_energy_e{energy_mevu}_{histories}"
    origin_scorers = []
    for scorer, category in ORIGIN_CATEGORIES:
        origin_scorers.append(
            f"""s:Sc/{scorer}/Quantity = "CarbonDoseOrigin"
s:Sc/{scorer}/Component = "Box2"
s:Sc/{scorer}/OutputFile = "output/{stem}_{category}"
s:Sc/{scorer}/OutputType = "binary"
s:Sc/{scorer}/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/{scorer}/OriginCategory = "{category}"
i:Sc/{scorer}/XBins = 1
i:Sc/{scorer}/YBins = {bins}
i:Sc/{scorer}/ZBins = 1
sv:Sc/{scorer}/Report = 1 "Sum"
"""
        )
    return f"""# Generated matched 1D minibeam validation case.
includeFile = run_dose_center_10k.txt

i:Ts/Seed = 20260728
i:Ts/NumberOfThreads = 40
i:Ts/ShowHistoryCountAtInterval = {max(1000, histories // 10)}
# Cover the full 4.8 GeV total kinetic energy of 400 MeV/u C-12 and avoid
# clamping the dedicated alpha/He3 electromagnetic tables.
d:Ph/Default/EMRangeMax = 6 GeV
i:Tf/NumberOfSequentialTimes = 1
d:Tf/TimelineEnd = 1 ms

dv:Tf/Scatterer1/L1/Values = 1 {total_energy_mev} MeV
dv:Tf/Scatterer1/L2/Values = 1 {total_energy_mev} MeV
iv:Tf/Scatterer1/L4/Values = 1 {histories}

d:Ge/Box/HLY = {half_depth:g} mm
d:Ge/Box/TransY = {water_center_world_y:g} mm
d:Ge/Box2/HLY = {half_depth:g} mm

s:Sc/DoseAtPhantomP/OutputFile = "output/{stem}_total"
i:Sc/DoseAtPhantomP/XBins = 1
i:Sc/DoseAtPhantomP/YBins = {bins}
i:Sc/DoseAtPhantomP/ZBins = 1

{chr(10).join(origin_scorers)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument("--depth-mm", type=float, default=400.0)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("out/minibeam/energy_sweep"),
    )
    parser.add_argument(
        "--gpu-template", type=Path,
        default=Path("config/beam_minibeam_center_copper_neutral_100k.yaml"),
    )
    parser.add_argument(
        "--spot-template", type=Path,
        default=Path("ct/minibeam/spots_single_center.txt"),
    )
    parser.add_argument(
        "--copper-reaction-package",
        type=Path,
        help="Optional multi-energy Copper reaction package override.",
    )
    args = parser.parse_args()
    if args.histories <= 0 or args.depth_mm <= 0.0:
        raise ValueError("histories and depth must be positive")

    config_dir = args.output_root / "config"
    result_dir = args.output_root / "gpu"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    topas_dir = Path("ct/minibeam")
    gpu_template = args.gpu_template.read_text(encoding="utf-8")
    spot_template = args.spot_template.read_text(encoding="utf-8")

    for energy in ENERGIES_MEVU:
        total_energy = 12 * energy
        spot_path = config_dir / f"spots_single_center_e{energy}.txt"
        spot_text = replace_spot_energy(spot_template, total_energy)
        spot_path.write_text(
            replace_spot_histories(spot_text, args.histories),
            encoding="utf-8",
        )

        case_dir = result_dir / f"e{energy}"
        case_dir.mkdir(parents=True, exist_ok=True)
        yaml = gpu_template
        replacements = {
            "number_of_histories": str(args.histories),
            "initial_energy_MeVu": str(energy),
            "phantom_length_mm": f"{args.depth_mm:g}",
            # A 1x1 transverse voxel is also the finite scorer boundary.
            # scorer_area_mm2 alone only sets dose mass and does not stop
            # particles that leave TOPAS Box2 (100 x 40 mm2).
            "enable_voxel_scoring": "true",
            "voxel_bins_x": "1",
            "voxel_bins_y": "1",
            "voxel_size_x_mm": "100",
            "voxel_size_y_mm": "40",
            # Low-energy C-12 loses more energy per Copper step. 0.10 mm
            # preserves the 100 MeV/u entrance spectrum; 0.25 mm is accurate
            # and substantially faster at 200--400 MeV/u.
            "minibeam_copper_max_step_mm": (
                "0.10" if energy <= 100 else "0.25"
            ),
            # The 1D total-dose accumulator shares the per-species scoring
            # buffer when secondary transport is enabled.  Keep that internal
            # scorer on, while leaving the optional species output paths empty.
            "enable_fragment_species_scoring": "true",
            "topas_spots_file": spot_path.as_posix(),
            "output_file": "",
            "dose_output_file": (
                case_dir / "depth_dose.csv"
            ).as_posix(),
            "fragment_species_dose_output_file": (
                case_dir / "species_dose.csv"
            ).as_posix(),
            "voxel_dose_output_file": "",
            "voxel_dose_Gy_output_file": "",
            "voxel_dose_mhd_output_file": "",
        }
        if args.copper_reaction_package is not None:
            replacements["minibeam_copper_reaction_package_file"] = (
                args.copper_reaction_package.as_posix()
            )
        for key, value in replacements.items():
            yaml = replace_yaml_value(yaml, key, value)
        yaml_path = config_dir / f"gpu_e{energy}.yaml"
        yaml_path.write_text(yaml, encoding="utf-8")

        topas_path = (
            topas_dir
            / f"run_energy_sweep_e{energy}_{args.histories}.txt"
        )
        topas_path.write_text(
            topas_case(energy, args.histories, args.depth_mm),
            encoding="utf-8",
        )
        print(yaml_path)
        print(topas_path)


if __name__ == "__main__":
    main()
