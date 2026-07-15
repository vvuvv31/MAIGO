#!/usr/bin/env python3
"""Validate GPU charged-origin sparse voxel category closure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CATEGORY_COLUMNS = (
    ("primary_c12", "primary_c12_MeV_per_primary"),
    ("secondary_carbon", "secondary_carbon_MeV_per_primary"),
    ("boron", "boron_MeV_per_primary"),
    ("beryllium", "beryllium_MeV_per_primary"),
    ("lithium", "lithium_MeV_per_primary"),
    ("helium", "helium_MeV_per_primary"),
    ("proton", "proton_MeV_per_primary"),
    ("other_charged", "other_charged_MeV_per_primary"),
)
SPECIES_IDD_COLUMNS = {
    "primary_c12": "primary_c12_MeV_per_primary",
    "secondary_carbon": "secondary_carbon_MeV_per_primary",
    "boron": "boron_MeV_per_primary",
    "beryllium": "beryllium_MeV_per_primary",
    "lithium": "lithium_MeV_per_primary",
    "helium": "helium_MeV_per_primary",
    "proton": "proton_MeV_per_primary",
    "other_charged": "other_MeV_per_primary",
}


def read_species_idd(path: Path, bins_z: int) -> tuple[list[float], dict[str, list[float]]]:
    total: list[float] = []
    categories = {name: [] for name, _ in CATEGORY_COLUMNS}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        for row in reader:
            total.append(float(row["total_MeV_per_primary"]))
            for name, _ in CATEGORY_COLUMNS:
                categories[name].append(float(row[SPECIES_IDD_COLUMNS[name]]))
    if len(total) != bins_z:
        raise ValueError(f"Expected {bins_z} IDD rows, found {len(total)}")
    return total, categories


def validate(
    voxel_path: Path,
    species_idd_path: Path,
    shape: tuple[int, int, int],
    tolerance: float,
) -> dict[str, object]:
    bins_x, bins_y, bins_z = shape
    voxel_count = bins_x * bins_y * bins_z
    seen = bytearray(voxel_count)
    plane_total = [0.0] * bins_z
    plane_categories = {
        name: [0.0] * bins_z for name, _ in CATEGORY_COLUMNS
    }
    category_integrals = {name: 0.0 for name, _ in CATEGORY_COLUMNS}
    maximum_voxel_closure = 0.0
    maximum_voxel_closure_index = [0, 0, 0]
    nonzero_voxels = 0

    with voxel_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "ix",
            "iy",
            "iz",
            "total_MeV_per_primary",
            *(column for _, column in CATEGORY_COLUMNS),
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"Missing voxel CSV columns: {missing}")
        for row in reader:
            x, y, z = int(row["ix"]), int(row["iy"]), int(row["iz"])
            if not (0 <= x < bins_x and 0 <= y < bins_y and 0 <= z < bins_z):
                raise ValueError(f"Voxel index outside {shape}: {(x, y, z)}")
            linear = (z * bins_y + y) * bins_x + x
            if seen[linear]:
                raise ValueError(f"Duplicate voxel index: {(x, y, z)}")
            seen[linear] = 1
            nonzero_voxels += 1

            total = float(row["total_MeV_per_primary"])
            category_values = {
                name: float(row[column]) for name, column in CATEGORY_COLUMNS
            }
            closure = math.fsum(category_values.values()) - total
            if abs(closure) > maximum_voxel_closure:
                maximum_voxel_closure = abs(closure)
                maximum_voxel_closure_index = [x, y, z]
            plane_total[z] += total
            for name, value in category_values.items():
                plane_categories[name][z] += value
                category_integrals[name] += value

    idd_total, idd_categories = read_species_idd(species_idd_path, bins_z)
    maximum_total_plane_closure = 0.0
    maximum_total_plane_bin = 0
    maximum_category_plane_closure = 0.0
    maximum_category_plane_location: list[object] = ["primary_c12", 0]
    for z in range(bins_z):
        total_closure = abs(plane_total[z] - idd_total[z])
        if total_closure > maximum_total_plane_closure:
            maximum_total_plane_closure = total_closure
            maximum_total_plane_bin = z
        for name, _ in CATEGORY_COLUMNS:
            category_closure = abs(
                plane_categories[name][z] - idd_categories[name][z]
            )
            if category_closure > maximum_category_plane_closure:
                maximum_category_plane_closure = category_closure
                maximum_category_plane_location = [name, z]

    integrated_total = math.fsum(plane_total)
    integrated_categories = math.fsum(category_integrals.values())
    integrated_closure = abs(integrated_categories - integrated_total)
    passed = max(
        maximum_voxel_closure,
        maximum_total_plane_closure,
        maximum_category_plane_closure,
        integrated_closure,
    ) <= tolerance
    return {
        "voxel_file": str(voxel_path),
        "species_idd_file": str(species_idd_path),
        "shape_xyz": list(shape),
        "nonzero_voxels": nonzero_voxels,
        "tolerance_MeV_per_primary": tolerance,
        "maximum_voxel_category_closure_MeV_per_primary": maximum_voxel_closure,
        "maximum_voxel_category_closure_index_xyz": maximum_voxel_closure_index,
        "maximum_total_plane_closure_MeV_per_primary_per_bin": maximum_total_plane_closure,
        "maximum_total_plane_closure_bin": maximum_total_plane_bin,
        "maximum_category_plane_closure_MeV_per_primary_per_bin": maximum_category_plane_closure,
        "maximum_category_plane_closure_location": maximum_category_plane_location,
        "integrated_category_closure_MeV_per_primary": integrated_closure,
        "category_integrals_MeV_per_primary": category_integrals,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-category-voxel", type=Path, required=True)
    parser.add_argument("--gpu-species-idd", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--bins-x", type=int, default=60)
    parser.add_argument("--bins-y", type=int, default=60)
    parser.add_argument("--bins-z", type=int, default=800)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    metrics = validate(
        args.gpu_category_voxel,
        args.gpu_species_idd,
        (args.bins_x, args.bins_y, args.bins_z),
        args.tolerance,
    )
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not metrics["passed"]:
        raise SystemExit("Charged-origin voxel closure validation failed")


if __name__ == "__main__":
    main()
