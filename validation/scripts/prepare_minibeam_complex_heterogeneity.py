#!/usr/bin/env python3
"""Generate matched GPU/TOPAS complex heterogeneous minibeam cases."""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = 0x47544343  # CCTG
VERSION_LEGACY = 1
NX, NY, NZ = 500, 1, 300
DX_MM, DY_MM, DZ_MM = 0.2, 40.0, 0.5
ORIGIN_X_MM, ORIGIN_Y_MM, ORIGIN_Z_MM = -50.0, -20.0, 0.0
PHANTOM_LENGTH_MM = NZ * DZ_MM


@dataclass(frozen=True)
class Material:
    class_id: int
    density_g_cm3: float
    topas_name: str
    color: str


@dataclass(frozen=True)
class Region:
    name: str
    material: str
    x_min_mm: float
    x_max_mm: float
    depth_min_mm: float
    depth_max_mm: float


MATERIALS = {
    "water": Material(2, 1.0, "G4_WATER", "#4c78a8"),
    "lung": Material(1, 1.04, "G4_LUNG_ICRP", "#72b7b2"),
    "bone": Material(3, 1.85, "G4_BONE_COMPACT_ICRU", "#b279a2"),
}

CASES = {
    "lateral_multimaterial": [
        Region("LungLeft", "lung", -15.0, -3.0, 15.0, 95.0),
        Region("BoneCenter", "bone", -3.0, 3.0, 15.0, 95.0),
    ],
    "longitudinal_multimaterial": [
        Region("LungLayer1", "lung", -20.0, 20.0, 18.0, 38.0),
        Region("BoneLayer", "bone", -20.0, 20.0, 52.0, 64.0),
        Region("LungLayer2", "lung", -20.0, 20.0, 78.0, 96.0),
    ],
    "combined_multimaterial": [
        Region("LungEntranceLeft", "lung", -15.0, -2.0, 12.0, 34.0),
        Region("BoneEntranceRight", "bone", 2.0, 15.0, 12.0, 34.0),
        Region("LungMiddleLeft", "lung", -15.0, -4.0, 34.0, 52.0),
        Region("BoneMiddle", "bone", -4.0, 4.0, 34.0, 52.0),
        Region("LungMiddleRight", "lung", 4.0, 15.0, 34.0, 52.0),
        Region("BoneDistalLeft", "bone", -15.0, 0.0, 52.0, 68.0),
        Region("LungDistalRight", "lung", 0.0, 15.0, 52.0, 68.0),
        Region("LungExitCenter", "lung", -5.0, 5.0, 68.0, 88.0),
    ],
}


def replace_yaml_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*:.*$")
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count == 1:
        return updated
    if count == 0:
        return text.rstrip() + f"\n{key}: {value}\n"
    raise ValueError(f"duplicate YAML key {key!r}")


def make_material_grid(regions: list[Region]) -> tuple[np.ndarray, np.ndarray]:
    density = np.full((NZ, NY, NX), MATERIALS["water"].density_g_cm3, dtype="<f4")
    material = np.full((NZ, NY, NX), MATERIALS["water"].class_id, dtype=np.uint8)
    x = ORIGIN_X_MM + (np.arange(NX) + 0.5) * DX_MM
    depth = ORIGIN_Z_MM + (np.arange(NZ) + 0.5) * DZ_MM
    occupied = np.zeros((NZ, NX), dtype=bool)
    for region in regions:
        x_mask = (x >= region.x_min_mm) & (x < region.x_max_mm)
        z_mask = (depth >= region.depth_min_mm) & (depth < region.depth_max_mm)
        selection = z_mask[:, None] & x_mask[None, :]
        if np.any(occupied & selection):
            raise ValueError(f"overlapping material region: {region.name}")
        occupied |= selection
        props = MATERIALS[region.material]
        density[:, 0, :][selection] = props.density_g_cm3
        material[:, 0, :][selection] = props.class_id
    return density, material


def write_cctg(path: Path, density: np.ndarray, material: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<IIIIIffffff",
        MAGIC,
        VERSION_LEGACY,
        NX,
        NY,
        NZ,
        ORIGIN_X_MM,
        ORIGIN_Y_MM,
        ORIGIN_Z_MM,
        DX_MM,
        DY_MM,
        DZ_MM,
    )
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(np.asarray(density, dtype="<f4").tobytes(order="C"))
        stream.write(np.asarray(material, dtype=np.uint8).tobytes(order="C"))


def topas_box(region: Region) -> str:
    props = MATERIALS[region.material]
    half_x = 0.5 * (region.x_max_mm - region.x_min_mm)
    half_depth = 0.5 * (region.depth_max_mm - region.depth_min_mm)
    center_x = 0.5 * (region.x_min_mm + region.x_max_mm)
    # Box2 local Y=-75 mm is the water entrance (GPU depth=0).
    center_y = 0.5 * (region.depth_min_mm + region.depth_max_mm) - 75.0
    return f"""
s:Ge/{region.name}/Parent = "Box2"
s:Ge/{region.name}/Type = "TsBox"
s:Ge/{region.name}/Material = "{props.topas_name}"
d:Ge/{region.name}/HLX = {half_x:g} mm
d:Ge/{region.name}/HLY = {half_depth:g} mm
d:Ge/{region.name}/HLZ = 19.999 mm
d:Ge/{region.name}/TransX = {center_x:g} mm
d:Ge/{region.name}/TransY = {center_y:g} mm
d:Ge/{region.name}/TransZ = 0 mm
d:Ge/{region.name}/MaxStepSize = 0.2 mm
"""


def topas_case(case_name: str, regions: list[Region], histories: int) -> str:
    boxes = "".join(topas_box(region) for region in regions)
    return f"""# Generated complex heterogeneous minibeam validation case.
includeFile = run_dose_center_10k.txt

i:Ts/Seed = 20260729
i:Ts/NumberOfThreads = 40
i:Ts/ShowHistoryCountAtInterval = {max(1000, histories // 10)}
i:Tf/NumberOfSequentialTimes = 1
d:Tf/TimelineEnd = 1 ms
d:Ph/Default/EMRangeMax = 6 GeV

dv:Tf/Scatterer1/L1/Values = 1 2400 MeV
dv:Tf/Scatterer1/L2/Values = 1 2400 MeV
iv:Tf/Scatterer1/L4/Values = 1 {histories}

d:Ge/Box/HLY = 75 mm
d:Ge/Box/TransY = 135 mm
d:Ge/Box2/HLY = 75 mm
{boxes}
s:Sc/DoseAtPhantomP/OutputFile = "output/minibeam_complex_{case_name}_{histories}_total"
i:Sc/DoseAtPhantomP/XBins = {NX}
i:Sc/DoseAtPhantomP/YBins = {NZ}
i:Sc/DoseAtPhantomP/ZBins = {NY}
"""


def gpu_case(
    template: str,
    case_name: str,
    grid_path: Path,
    spot_path: Path,
    histories: int,
    output_root: Path,
    precision: str,
) -> str:
    # Remove the mutually-exclusive slab description inherited from the
    # heterogeneous validation template.
    slab_keys = {
        "enable_layered_phantom",
        "slab_z_ends_mm",
        "slab_densities_g_per_cm3",
        "slab_radiation_lengths_g_per_cm2",
        "slab_stopping_power_files",
        "slab_cross_section_files",
    }
    lines = [
        line
        for line in template.splitlines()
        if line.split(":", 1)[0].strip() not in slab_keys
    ]
    text = "\n".join(lines) + "\n"
    high_accuracy = precision == "high_accuracy"
    case_output = (
        output_root
        / case_name
        / ("gpu_high_accuracy" if high_accuracy else "gpu")
    )
    overrides = {
        "number_of_histories": str(histories),
        "maximum_step_mm": "0.1" if high_accuracy else "0.2",
        "maximum_relative_energy_loss": "0.001" if high_accuracy else "0.005",
        "random_seed": "20260729",
        "enable_ct_grid": "true",
        "ct_grid_file": grid_path.as_posix(),
        "ct_skip_homogeneous_face_clamp": "true",
        "enable_ct_material_mcs": "true",
        "ct_air_stopping_power_file": "data/stopping_power_air_geant4_11_3_2.csv",
        "ct_lung_stopping_power_file": "data/stopping_power_lung_geant4_11_3_2.csv",
        "ct_water_stopping_power_file": "data/stopping_power_water_geant4_11_3_2.csv",
        "ct_bone_stopping_power_file": "data/stopping_power_bone_geant4_11_3_2.csv",
        "ct_lung_cross_section_file": "data/c12_inelastic_cross_sections_lung_geant4_11_3_2.csv",
        "ct_water_cross_section_file": "data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv",
        "ct_bone_cross_section_file": "data/c12_inelastic_cross_sections_bone_geant4_11_3_2.csv",
        "enable_voxel_scoring": "true",
        "voxel_scorer_clamps_transport": "false",
        "voxel_bins_x": str(NX),
        "voxel_bins_y": str(NY),
        "voxel_size_x_mm": f"{DX_MM:g}",
        "voxel_size_y_mm": f"{DY_MM:g}",
        "topas_spots_file": spot_path.as_posix(),
        "dose_output_file": (case_output / "depth_dose.csv").as_posix(),
        "fragment_species_dose_output_file": (case_output / "species_dose.csv").as_posix(),
        "voxel_dose_mhd_output_file": (case_output / "dose.mhd").as_posix(),
    }
    for key, value in overrides.items():
        text = replace_yaml_value(text, key, value)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/minibeam/complex_heterogeneity"),
    )
    parser.add_argument(
        "--precision",
        choices=("production", "high_accuracy"),
        default="production",
        help=(
            "production uses 0.2 mm/0.005 transport limits; high_accuracy "
            "uses 0.1 mm/0.001 for interface-sensitive reference runs"
        ),
    )
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=Path("ct/grid/minibeam_complex"),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config/generated/minibeam_complex"),
    )
    parser.add_argument(
        "--gpu-template",
        type=Path,
        default=Path("config/beam_minibeam_hetero_bone_e200_100k.yaml"),
    )
    args = parser.parse_args()
    if args.histories <= 0:
        raise ValueError("histories must be positive")

    template = args.gpu_template.read_text(encoding="utf-8")
    args.config_root.mkdir(parents=True, exist_ok=True)
    spot_template_path = Path(
        "out/minibeam/energy_sweep_100k/config/spots_single_center_e200.txt"
    )
    spot_template = spot_template_path.read_text(encoding="utf-8")
    spot_text, spot_count = re.subn(
        r"(?m)^(iv:Tf/Scatterer1/L4/Values\s*=\s*1\s+)\d+(\s*)$",
        rf"\g<1>{args.histories}\g<2>",
        spot_template,
    )
    if spot_count != 1:
        raise ValueError(
            f"expected one history row in {spot_template_path}, found {spot_count}"
        )
    spot_path = (
        args.config_root / f"spots_single_center_e200_{args.histories}.txt"
    )
    spot_path.write_text(spot_text, encoding="utf-8")
    topas_root = Path("ct/minibeam")
    for case_name in args.cases:
        regions = CASES[case_name]
        density, material = make_material_grid(regions)
        grid_path = args.grid_root / f"{case_name}.bin"
        write_cctg(grid_path, density, material)
        metadata = {
            "case": case_name,
            "grid": {
                "shape_xyz": [NX, NY, NZ],
                "origin_xyz_mm": [ORIGIN_X_MM, ORIGIN_Y_MM, ORIGIN_Z_MM],
                "spacing_xyz_mm": [DX_MM, DY_MM, DZ_MM],
            },
            "materials": {
                name: {
                    "class_id": value.class_id,
                    "density_g_cm3": value.density_g_cm3,
                    "topas_name": value.topas_name,
                    "voxel_count": int(np.count_nonzero(material == value.class_id)),
                }
                for name, value in MATERIALS.items()
            },
            "regions": [region.__dict__ for region in regions],
        }
        metadata_path = grid_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        precision_suffix = (
            "_high_accuracy" if args.precision == "high_accuracy" else ""
        )
        yaml_path = (
            args.config_root
            / f"{case_name}_{args.histories}{precision_suffix}.yaml"
        )
        yaml_path.write_text(
            gpu_case(
                template,
                case_name,
                grid_path,
                spot_path,
                args.histories,
                args.output_root,
                args.precision,
            ),
            encoding="utf-8",
        )
        topas_path = (
            topas_root / f"run_complex_{case_name}_{args.histories}.txt"
        )
        topas_path.write_text(
            topas_case(case_name, regions, args.histories), encoding="utf-8"
        )
        print(grid_path)
        print(metadata_path)
        print(yaml_path)
        print(topas_path)


if __name__ == "__main__":
    main()
