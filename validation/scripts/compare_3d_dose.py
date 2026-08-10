#!/usr/bin/env python3
"""Compare absolute GPU charged-origin 3D dose with TOPAS ancestor-attributed dose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np


CHARGED_CATEGORIES = (
    "primary_c12",
    "secondary_carbon",
    "boron",
    "beryllium",
    "lithium",
    "helium",
    "proton",
    "other_charged",
)
MEV_J = 1.602176634e-13
WATER_DENSITY_KG_M3 = 1000.0


def read_gpu_voxels(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 6), ndmin=2)
    result = np.zeros(shape, dtype=np.float64)
    if rows.size:
        indices = rows[:, :3].astype(np.int64)
        if np.any(indices < 0) or any(
            np.any(indices[:, axis] >= shape[axis]) for axis in range(3)
        ):
            raise ValueError(f"GPU voxel index outside {shape}")
        linear = np.ravel_multi_index(indices.T, shape)
        if np.unique(linear).size != linear.size:
            raise ValueError("Duplicate GPU sparse voxel index")
        result[tuple(indices.T)] = rows[:, 3]
    return result


def read_gpu_idd(path: Path, bins: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,), ndmin=1)
    if values.size != bins:
        raise ValueError(f"Expected {bins} GPU IDD bins, found {values.size}")
    return values.astype(np.float64)


def read_topas_charged_energy(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as package:
        shape = tuple(int(value) for value in package["shape_xyz"])
        voxel_size_mm = package["voxel_size_mm"].astype(np.float64)
        voxel_mass_kg = (
            WATER_DENSITY_KG_M3 * float(np.prod(voxel_size_mm)) * 1.0e-9
        )
        gy_to_mev = voxel_mass_kg / MEV_J
        result = np.zeros(shape, dtype=np.float64)
        flat = result.ravel()
        for category in CHARGED_CATEGORIES:
            linear = package[f"{category}_linear_index"].astype(np.int64)
            dose = package[f"{category}_dose_Gy_per_primary"].astype(np.float64)
            flat[linear] += dose * gy_to_mev
        x_mm = package["x_center_mm"].astype(np.float64)
        y_mm = package["y_center_mm"].astype(np.float64)
        depth_mm = package["depth_center_mm"].astype(np.float64)
    return result, x_mm, y_mm, depth_mm, voxel_size_mm


def weighted_sigma(coordinate: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(coordinate * weights) / total)
    variance = float(np.sum((coordinate - mean) ** 2 * weights) / total)
    return float(np.sqrt(max(0.0, variance)))


def lateral_widths(
    dose: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    depth_mm: np.ndarray,
    requested_depths_mm: list[float],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for requested in requested_depths_mm:
        index = int(np.argmin(np.abs(depth_mm - requested)))
        plane = dose[:, :, index]
        result[f"{depth_mm[index]:.2f}"] = {
            "sigma_x_mm": weighted_sigma(x_mm, np.sum(plane, axis=1)),
            "sigma_y_mm": weighted_sigma(y_mm, np.sum(plane, axis=0)),
            "energy_MeV_per_primary": float(np.sum(plane)),
        }
    return result


def finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else fallback


def write_plot(
    path: Path,
    gpu: np.ndarray,
    topas: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    depth_mm: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_idd = np.sum(gpu, axis=(0, 1))
    topas_idd = np.sum(topas, axis=(0, 1))
    peak = int(np.argmax(topas_idd))
    gpu_xz = np.sum(gpu, axis=1)
    topas_xz = np.sum(topas, axis=1)
    difference_xz = gpu_xz - topas_xz
    common_max = max(float(np.max(gpu_xz)), float(np.max(topas_xz)))
    positive = np.concatenate((gpu_xz[gpu_xz > 0.0], topas_xz[topas_xz > 0.0]))
    common_min = max(
        common_max * 1.0e-6,
        finite_percentile(positive, 5.0, common_max * 1.0e-6),
    )
    difference_limit = max(
        finite_percentile(np.abs(difference_xz), 99.5, 0.0),
        common_max * 1.0e-4,
    )
    extent_xz = (
        depth_mm[0] - 0.25,
        depth_mm[-1] + 0.25,
        x_mm[0] - 2.5,
        x_mm[-1] + 2.5,
    )
    extent_xy = (
        x_mm[0] - 2.5,
        x_mm[-1] + 2.5,
        y_mm[0] - 2.5,
        y_mm[-1] + 2.5,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes[0, 0].plot(depth_mm, topas_idd, label="TOPAS charged-origin", linewidth=1.5)
    axes[0, 0].plot(depth_mm, gpu_idd, label="GPU charged-origin", linewidth=1.2)
    axes[0, 0].axvline(depth_mm[peak], color="0.4", linestyle=":", linewidth=0.9)
    axes[0, 0].set(
        title="Absolute charged-origin IDD",
        xlabel="Depth (mm)",
        ylabel="MeV / primary / 0.5 mm",
    )
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend()

    norm = LogNorm(vmin=common_min, vmax=common_max)
    topas_image = axes[0, 1].imshow(
        topas_xz,
        origin="lower",
        aspect="auto",
        extent=extent_xz,
        cmap="inferno",
        norm=norm,
    )
    axes[0, 1].set(title="TOPAS x-z projection", xlabel="Depth (mm)", ylabel="x (mm)")
    gpu_image = axes[0, 2].imshow(
        gpu_xz,
        origin="lower",
        aspect="auto",
        extent=extent_xz,
        cmap="inferno",
        norm=norm,
    )
    axes[0, 2].set(title="GPU x-z projection", xlabel="Depth (mm)", ylabel="x (mm)")
    figure.colorbar(gpu_image, ax=(axes[0, 1], axes[0, 2]), label="MeV / primary / x-z cell")

    difference_image = axes[1, 0].imshow(
        difference_xz,
        origin="lower",
        aspect="auto",
        extent=extent_xz,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-difference_limit, vmax=difference_limit),
    )
    axes[1, 0].set(title="GPU - TOPAS x-z", xlabel="Depth (mm)", ylabel="x (mm)")
    figure.colorbar(difference_image, ax=axes[1, 0], label="MeV / primary / x-z cell")

    topas_plane = topas[:, :, peak].T
    gpu_plane = gpu[:, :, peak].T
    transverse_max = max(float(np.max(topas_plane)), float(np.max(gpu_plane)))
    transverse_positive = np.concatenate(
        (topas_plane[topas_plane > 0.0], gpu_plane[gpu_plane > 0.0])
    )
    transverse_min = max(
        transverse_max * 1.0e-5,
        finite_percentile(transverse_positive, 5.0, transverse_max * 1.0e-5),
    )
    transverse_norm = LogNorm(vmin=transverse_min, vmax=transverse_max)
    topas_transverse = axes[1, 1].imshow(
        topas_plane,
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        norm=transverse_norm,
    )
    axes[1, 1].set(
        title=f"TOPAS transverse @ {depth_mm[peak]:.2f} mm",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )
    gpu_transverse = axes[1, 2].imshow(
        gpu_plane,
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        norm=transverse_norm,
    )
    axes[1, 2].set(
        title=f"GPU transverse @ {depth_mm[peak]:.2f} mm",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )
    figure.colorbar(
        gpu_transverse,
        ax=(axes[1, 1], axes[1, 2]),
        label="MeV / primary / voxel",
    )
    figure.suptitle("200 MeV/u C-12: GPU MCS vs TOPAS ancestor-attributed charged dose")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-voxel", type=Path, required=True)
    parser.add_argument("--gpu-idd", type=Path, required=True)
    parser.add_argument("--topas-npz", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--tail-start-mm", type=float, default=90.0)
    args = parser.parse_args()

    topas, x_mm, y_mm, depth_mm, voxel_size_mm = read_topas_charged_energy(
        args.topas_npz
    )
    gpu = read_gpu_voxels(args.gpu_voxel, topas.shape)
    gpu_idd_file = read_gpu_idd(args.gpu_idd, topas.shape[2])
    gpu_idd = np.sum(gpu, axis=(0, 1))
    topas_idd = np.sum(topas, axis=(0, 1))
    closure = gpu_idd - gpu_idd_file
    difference = gpu - topas
    topas_integral = float(np.sum(topas))
    gpu_integral = float(np.sum(gpu))
    tail_mask = depth_mm >= args.tail_start_mm
    topas_tail = float(np.sum(topas[:, :, tail_mask]))
    gpu_tail = float(np.sum(gpu[:, :, tail_mask]))
    high_dose_mask = topas >= float(np.max(topas)) * 0.01
    high_topas = topas[high_dose_mask]
    high_gpu = gpu[high_dose_mask]
    correlation = (
        float(np.corrcoef(high_gpu, high_topas)[0, 1])
        if high_topas.size > 1 and np.std(high_topas) > 0.0 and np.std(high_gpu) > 0.0
        else 0.0
    )
    peak = int(np.argmax(topas_idd))
    requested_depths = [20.0, 50.0, float(depth_mm[peak])]

    metrics = {
        "comparison_semantics": (
            "GPU transported charged dose versus the sum of TOPAS primary_c12, "
            "secondary C/B/Be/Li/He, proton, and other_charged ancestor categories"
        ),
        "normalization": "absolute MeV/primary; no global scale",
        "shape_xyz": list(topas.shape),
        "voxel_size_mm": voxel_size_mm.tolist(),
        "gpu_voxel_idd_closure_max_abs_MeV_per_primary_per_bin": float(
            np.max(np.abs(closure))
        ),
        "integral": {
            "gpu_MeV_per_primary": gpu_integral,
            "topas_MeV_per_primary": topas_integral,
            "signed_percent": 100.0 * (gpu_integral / topas_integral - 1.0),
        },
        "tail_ge_90mm": {
            "gpu_MeV_per_primary": gpu_tail,
            "topas_MeV_per_primary": topas_tail,
            "signed_percent": 100.0 * (gpu_tail / topas_tail - 1.0),
        },
        "voxel": {
            "normalized_L1_percent": 100.0 * float(np.sum(np.abs(difference))) / topas_integral,
            "normalized_RMSE_to_topas_max_percent": (
                100.0 * float(np.sqrt(np.mean(difference * difference))) / float(np.max(topas))
            ),
            "high_dose_threshold_percent_of_topas_max": 1.0,
            "high_dose_voxel_count": int(np.count_nonzero(high_dose_mask)),
            "high_dose_mean_abs_percent": (
                100.0 * float(np.mean(np.abs(high_gpu - high_topas) / high_topas))
            ),
            "high_dose_pearson_r": correlation,
        },
        "topas_peak_depth_mm": float(depth_mm[peak]),
        "gpu_lateral_widths": lateral_widths(
            gpu, x_mm, y_mm, depth_mm, requested_depths
        ),
        "topas_lateral_widths": lateral_widths(
            topas, x_mm, y_mm, depth_mm, requested_depths
        ),
        "inputs": {
            "gpu_voxel": args.gpu_voxel.as_posix(),
            "gpu_idd": args.gpu_idd.as_posix(),
            "topas_npz": args.topas_npz.as_posix(),
        },
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.output_metrics.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metrics, indent=2, ensure_ascii=False))
        stream.write("\n")
    write_plot(args.output_plot, gpu, topas, x_mm, y_mm, depth_mm)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_metrics}")
    print(f"Wrote {args.output_plot}")


if __name__ == "__main__":
    main()
