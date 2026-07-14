#!/usr/bin/env python3
"""Validate and package ancestor-attributed TOPAS 3D dose scorers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = (
    "primary_c12",
    "secondary_carbon",
    "boron",
    "beryllium",
    "lithium",
    "helium",
    "proton",
    "other_charged",
    "neutron",
    "gamma",
    "neutral_other",
    "unresolved",
)

DISPLAY_NAMES = {
    "primary_c12": "Primary C-12 lineage",
    "secondary_carbon": "Secondary carbon",
    "boron": "Boron",
    "beryllium": "Beryllium",
    "lithium": "Lithium",
    "helium": "Helium",
    "proton": "Proton",
    "other_charged": "Other charged",
    "neutron": "Neutron lineage",
    "gamma": "Gamma lineage",
    "neutral_other": "Other neutral lineage",
    "unresolved": "Unresolved",
}

MEV_J = 1.602176634e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sparse_topas(path: Path, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Read a sparse TOPAS i,j,k,value CSV, including a header-only file."""
    indices: list[tuple[int, int, int]] = []
    values: list[float] = []
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("# TOPAS Version:"):
                metadata["topas_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Parameter File:"):
                metadata["parameter_file"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Results for scorer:"):
                metadata["scorer"] = line.split(":", 1)[1].strip()
            elif line.startswith("#") or not line:
                continue
            else:
                fields = [value.strip() for value in line.split(",")]
                if len(fields) < 4:
                    raise ValueError(f"Expected i,j,k,value rows in {path}: {line}")
                indices.append(tuple(int(fields[column]) for column in range(3)))
                values.append(float(fields[3]))

    coordinates = np.asarray(indices, dtype=np.int32).reshape((-1, 3))
    dose = np.asarray(values, dtype=np.float64)
    if coordinates.size:
        if np.any(coordinates < 0) or any(
            np.any(coordinates[:, axis] >= shape[axis]) for axis in range(3)
        ):
            raise ValueError(f"Voxel index outside {shape} in {path}")
        linear = np.ravel_multi_index(coordinates.T, shape)
        if len(np.unique(linear)) != len(linear):
            raise ValueError(f"Duplicate voxel index in {path}")
        order = np.argsort(linear)
        coordinates = coordinates[order]
        dose = dose[order]
    return coordinates, dose, metadata


def sparse_to_dense(
    coordinates: np.ndarray, values: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float64)
    if coordinates.size:
        result[tuple(coordinates.T)] = values
    return result


def read_1d_energy(path: Path, bin_count: int) -> tuple[np.ndarray, dict[str, str]]:
    result = np.zeros(bin_count, dtype=np.float64)
    metadata: dict[str, str] = {}
    seen: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("# TOPAS Version:"):
                metadata["topas_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Parameter File:"):
                metadata["parameter_file"] = line.split(":", 1)[1].strip()
            elif line.startswith("#") or not line:
                continue
            else:
                fields = [value.strip() for value in line.split(",")]
                z_index = int(fields[2])
                if not 0 <= z_index < bin_count or z_index in seen:
                    raise ValueError(f"Invalid or duplicate depth index {z_index} in {path}")
                seen.add(z_index)
                result[z_index] = float(fields[3])
    if len(seen) != bin_count:
        raise ValueError(f"Expected {bin_count} depth bins in {path}, found {len(seen)}")
    return result, metadata


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, object] = {"path": path.as_posix(), "sha256": sha256(path)}
    geant4 = re.search(r"Geant4 version Name:\s+(\S+)", text)
    threads = re.search(r"setting number of threads to:\s+(\d+)", text)
    elapsed = re.search(r"^\s*Total:\s+User=[^\n]*?Real=([0-9.]+)s", text, re.MULTILINE)
    if geant4:
        result["geant4_version"] = geant4.group(1)
    if threads:
        result["threads"] = int(threads.group(1))
    if elapsed:
        result["elapsed_real_s"] = float(elapsed.group(1))
    return result


def output_path(directory: Path, case_name: str, category: str) -> Path:
    return directory / f"ancestor_{case_name}_{category}_dose_3d.csv"


def write_idd(
    path: Path,
    depth_mm: np.ndarray,
    total: np.ndarray,
    components: dict[str, np.ndarray],
    independent_total: np.ndarray,
    direct_total: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "depth_mm",
                "total_from_3d_MeV_per_primary",
                *(f"{name}_MeV_per_primary" for name in CATEGORIES),
                "topas_independent_3d_total_MeV_per_primary",
                "topas_direct_total_MeV_per_primary",
                "category_closure_MeV_per_primary",
                "independent_total_closure_MeV_per_primary",
                "dose_energy_closure_MeV_per_primary",
            ]
        )
        reconstructed = np.sum([components[name] for name in CATEGORIES], axis=0)
        for index, depth in enumerate(depth_mm):
            writer.writerow(
                [
                    f"{depth:.12g}",
                    f"{total[index]:.12g}",
                    *(f"{components[name][index]:.12g}" for name in CATEGORIES),
                    f"{independent_total[index]:.12g}",
                    f"{direct_total[index]:.12g}",
                    f"{reconstructed[index] - total[index]:.12g}",
                    f"{reconstructed[index] - independent_total[index]:.12g}",
                    f"{independent_total[index] - direct_total[index]:.12g}",
                ]
            )


def write_plot(
    path: Path,
    total_dose: np.ndarray,
    depth_mm: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    idd: np.ndarray,
    components: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak_z = int(np.argmax(idd))
    center_x = len(x_mm) // 2
    center_y = len(y_mm) // 2
    maximum = float(np.max(total_dose))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    image = axes[0, 0].imshow(
        total_dose[:, center_y, :],
        origin="lower",
        aspect="auto",
        extent=(depth_mm[0] - 0.25, depth_mm[-1] + 0.25, x_mm[0] - 2.5, x_mm[-1] + 2.5),
        cmap="inferno",
        vmin=0.0,
        vmax=maximum,
    )
    axes[0, 0].set(title="Central XZ total dose", xlabel="Depth (mm)", ylabel="x (mm)")
    figure.colorbar(image, ax=axes[0, 0], label="Gy / primary")

    transverse = axes[0, 1].imshow(
        total_dose[:, :, peak_z].T,
        origin="lower",
        extent=(x_mm[0] - 2.5, x_mm[-1] + 2.5, y_mm[0] - 2.5, y_mm[-1] + 2.5),
        cmap="inferno",
        vmin=0.0,
        vmax=maximum,
    )
    axes[0, 1].set(
        title=f"Transverse dose at {depth_mm[peak_z]:.2f} mm",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )
    figure.colorbar(transverse, ax=axes[0, 1], label="Gy / primary")

    axes[1, 0].plot(depth_mm, idd, color="black", linewidth=1.4, label="Total")
    charged = CATEGORIES[:8]
    axes[1, 0].stackplot(
        depth_mm,
        *(components[name] for name in charged),
        labels=[DISPLAY_NAMES[name] for name in charged],
        alpha=0.8,
    )
    axes[1, 0].set(
        title="Ancestor-attributed IDD derived from 3D dose",
        xlabel="Depth (mm)",
        ylabel="MeV / primary / 0.5 mm",
    )
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=7, ncol=2)

    for name in CATEGORIES[8:]:
        axes[1, 1].semilogy(
            depth_mm,
            np.maximum(components[name], max(float(np.max(idd)) * 1.0e-10, 1.0e-14)),
            label=DISPLAY_NAMES[name],
        )
    axes[1, 1].set(
        title="Neutral-source lineages and diagnostic",
        xlabel="Depth (mm)",
        ylabel="MeV / primary / 0.5 mm",
    )
    axes[1, 1].grid(alpha=0.2, which="both")
    axes[1, 1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/topas/output"))
    parser.add_argument("--case", choices=("smoke", "development", "reference"), required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(60, 60, 800), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--voxel-mm", type=float, nargs=3, default=(5.0, 5.0, 0.5), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--tail-start-mm", type=float, default=90.0)
    parser.add_argument("--closure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--dose-energy-bin-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--dose-energy-integral-relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-idd", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.histories <= 0:
        raise SystemExit("--histories must be positive")
    shape = tuple(args.shape)
    voxel_mm = tuple(args.voxel_mm)
    if any(value <= 0 for value in (*shape, *voxel_mm)):
        raise SystemExit("Shape and voxel dimensions must be positive")

    total_path = output_path(args.input_dir, args.case, "total")
    total_coordinates, total_sum_gy, header = read_sparse_topas(total_path, shape)
    independent_total_dose = sparse_to_dense(
        total_coordinates, total_sum_gy / args.histories, shape
    )
    reconstructed_sum_gy = np.zeros(shape, dtype=np.float64)
    category_dose: dict[str, np.ndarray] = {}
    category_idd: dict[str, np.ndarray] = {}
    package: dict[str, np.ndarray] = {}
    input_files: dict[str, dict[str, str]] = {
        "total_3d": {"path": total_path.as_posix(), "sha256": sha256(total_path)}
    }

    voxel_mass_kg = args.density_kg_m3 * np.prod(voxel_mm) * 1.0e-9
    gy_to_mev = voxel_mass_kg / MEV_J
    for name in CATEGORIES:
        path = output_path(args.input_dir, args.case, name)
        coordinates, sum_gy, metadata = read_sparse_topas(path, shape)
        if metadata.get("topas_version") != header.get("topas_version"):
            raise SystemExit(f"TOPAS version mismatch between {total_path} and {path}")
        dense_sum_gy = sparse_to_dense(coordinates, sum_gy, shape)
        reconstructed_sum_gy += dense_sum_gy
        dose_per_primary = dense_sum_gy / args.histories
        category_dose[name] = dose_per_primary
        category_idd[name] = dose_per_primary.sum(axis=(0, 1)) * gy_to_mev
        package[f"{name}_linear_index"] = (
            np.ravel_multi_index(coordinates.T, shape).astype(np.int32)
            if coordinates.size
            else np.empty(0, dtype=np.int32)
        )
        package[f"{name}_dose_Gy_per_primary"] = (sum_gy / args.histories).astype(np.float32)
        input_files[name] = {"path": path.as_posix(), "sha256": sha256(path)}

    independent_closure_sum_gy = reconstructed_sum_gy - sparse_to_dense(
        total_coordinates, total_sum_gy, shape
    )
    # The authoritative attributed total is the mutually exclusive category
    # sum. The separately accumulated built-in total remains an independent
    # floating-point and scoring-geometry cross-check.
    total_dose = reconstructed_sum_gy / args.histories
    total_idd = total_dose.sum(axis=(0, 1)) * gy_to_mev
    independent_total_idd = independent_total_dose.sum(axis=(0, 1)) * gy_to_mev
    category_reconstructed_idd = np.sum([category_idd[name] for name in CATEGORIES], axis=0)
    category_idd_closure = category_reconstructed_idd - total_idd
    independent_total_closure = category_reconstructed_idd - independent_total_idd

    direct_total_path = args.input_dir / f"ancestor_{args.case}_total_energy_deposit.csv"
    direct_total_sum, direct_header = read_1d_energy(direct_total_path, shape[2])
    if direct_header.get("topas_version") != header.get("topas_version"):
        raise SystemExit("TOPAS version mismatch between 3D dose and direct IDD")
    direct_total_idd = direct_total_sum / args.histories
    dose_energy_closure = independent_total_idd - direct_total_idd

    maximum_idd_closure = float(np.max(np.abs(category_idd_closure)))
    maximum_independent_total_closure = float(np.max(np.abs(independent_total_closure)))
    maximum_dose_energy_closure = float(np.max(np.abs(dose_energy_closure)))
    dose_energy_integral_relative = float(
        np.sum(independent_total_idd) / np.sum(direct_total_idd) - 1.0
    )
    if maximum_idd_closure >= args.closure_tolerance:
        raise SystemExit(
            f"Ancestor IDD closure {maximum_idd_closure:.6g} is not below "
            f"{args.closure_tolerance:.6g} MeV/primary/bin"
        )
    # The independent 3D and 1D parallel scoring worlds can assign a boundary
    # step to adjacent z bins. Their energy integral must remain invariant;
    # the per-bin difference is retained as a navigation/binning diagnostic.
    if maximum_dose_energy_closure >= args.dose_energy_bin_tolerance:
        raise SystemExit(
            f"3D dose-derived IDD differs from original total by {maximum_dose_energy_closure:.6g}; "
            f"required < {args.dose_energy_bin_tolerance:.6g} MeV/primary/bin"
        )
    if abs(dose_energy_integral_relative) >= args.dose_energy_integral_relative_tolerance:
        raise SystemExit(
            "3D dose/direct-energy integral relative difference "
            f"{dose_energy_integral_relative:.6g} is not below "
            f"{args.dose_energy_integral_relative_tolerance:.6g}"
        )

    depth_mm = (np.arange(shape[2]) + 0.5) * voxel_mm[2]
    x_mm = (np.arange(shape[0]) + 0.5) * voxel_mm[0] - shape[0] * voxel_mm[0] / 2.0
    y_mm = (np.arange(shape[1]) + 0.5) * voxel_mm[1] - shape[1] * voxel_mm[1] / 2.0
    package.update(
        {
            "total_dose_Gy_per_primary": total_dose.astype(np.float32),
            "topas_independent_total_dose_Gy_per_primary": independent_total_dose.astype(np.float32),
            "x_center_mm": x_mm.astype(np.float32),
            "y_center_mm": y_mm.astype(np.float32),
            "depth_center_mm": depth_mm.astype(np.float32),
            "shape_xyz": np.asarray(shape, dtype=np.int32),
            "voxel_size_mm": np.asarray(voxel_mm, dtype=np.float32),
        }
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **package)
    write_idd(
        args.output_idd,
        depth_mm,
        total_idd,
        category_idd,
        independent_total_idd,
        direct_total_idd,
    )
    write_plot(args.plot, total_dose, depth_mm, x_mm, y_mm, total_idd, category_idd)

    tail_mask = depth_mm >= args.tail_start_mm
    category_integrals = {name: float(np.sum(category_idd[name])) for name in CATEGORIES}
    tail_integrals = {name: float(np.sum(category_idd[name][tail_mask])) for name in CATEGORIES}
    total_integral = float(np.sum(total_idd))
    tail_total = float(np.sum(total_idd[tail_mask]))
    log_path = args.log or args.input_dir / f"ancestor-{args.case}_topas.log"
    if not log_path.exists():
        raise SystemExit(f"TOPAS log not found: {log_path}")
    metadata = {
        "case": args.case,
        "histories": args.histories,
        "topas_version": header.get("topas_version", "unknown"),
        "parameter_file": header.get("parameter_file", "unknown"),
        "scoring_semantics": "dose-to-medium attributed by track ancestry; electron/positron dose inherits its charged parent, while neutral-source descendants retain neutron/gamma/neutral_other origin",
        "attributed_total_semantics": "authoritative attributed total is the sum of all mutually exclusive ancestor categories; the separately accumulated TOPAS DoseToMedium total is retained as an independent numerical cross-check",
        "array_axis_order": "x,y,z",
        "shape_xyz": list(shape),
        "voxel_size_mm": list(voxel_mm),
        "phantom_origin_world_mm": [-shape[0] * voxel_mm[0] / 2.0, -shape[1] * voxel_mm[1] / 2.0, -shape[2] * voxel_mm[2] / 2.0],
        "depth_origin_world_mm": -shape[2] * voxel_mm[2] / 2.0,
        "density_kg_m3": args.density_kg_m3,
        "voxel_mass_kg": voxel_mass_kg,
        "total_deposited_MeV_per_primary": total_integral,
        "category_deposited_MeV_per_primary": category_integrals,
        "category_fraction": {name: value / total_integral for name, value in category_integrals.items()},
        "tail_start_mm": args.tail_start_mm,
        "tail_deposited_MeV_per_primary": tail_total,
        "tail_category_deposited_MeV_per_primary": tail_integrals,
        "tail_category_fraction": {name: value / tail_total for name, value in tail_integrals.items()},
        "closure_tolerance_MeV_per_primary_per_bin": args.closure_tolerance,
        "category_closure_max_abs_MeV_per_primary_per_bin": maximum_idd_closure,
        "independent_total_closure_max_abs_MeV_per_primary_per_bin": maximum_independent_total_closure,
        "dose_energy_closure_max_abs_MeV_per_primary_per_bin": maximum_dose_energy_closure,
        "dose_energy_integral_relative_difference": dose_energy_integral_relative,
        "independent_total_voxel_closure_max_abs_Gy_sum": float(np.max(np.abs(independent_closure_sum_gy))),
        "unresolved_MeV_per_primary": category_integrals["unresolved"],
        "original_total_energy_integral_unchanged": abs(dose_energy_integral_relative) < args.dose_energy_integral_relative_tolerance,
        "input_files": {
            **input_files,
            "direct_total_idd": {"path": direct_total_path.as_posix(), "sha256": sha256(direct_total_path)},
        },
        "topas_log": parse_log(log_path),
        "output_npz": args.output_npz.as_posix(),
        "output_idd": args.output_idd.as_posix(),
        "plot": args.plot.as_posix(),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"Wrote {args.output_npz}")
    print(f"Wrote {args.output_idd}")
    print(f"Wrote {args.metadata}")
    print(f"Wrote {args.plot}")
    print(f"Total deposited energy: {total_integral:.6f} MeV/primary")
    print(f"Maximum category closure: {maximum_idd_closure:.3e} MeV/primary/bin")
    print(f"Maximum independent-total cross-check difference: {maximum_independent_total_closure:.3e} MeV/primary/bin")
    print(f"Maximum 3D dose/direct-energy closure: {maximum_dose_energy_closure:.3e} MeV/primary/bin")
    print(f"Unresolved: {category_integrals['unresolved']:.3e} MeV/primary")


if __name__ == "__main__":
    main()
