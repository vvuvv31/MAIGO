#!/usr/bin/env python3
"""Compare TOPAS and GPU species-resolved 3D dose scorers by X/Y summation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = (
    ("primary_C", "primary", "primary"),
    ("secondary_C", "secondary_carbon", "secondary_carbon"),
    ("Z5_B", "boron", "secondary_boron"),
    ("Z4_Be", "beryllium", "secondary_beryllium"),
    ("Z3_Li", "lithium", "secondary_lithium"),
    ("Z2_He", "helium", "secondary_helium"),
    ("Z1_H", "proton", "secondary_proton"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_idd(path: Path, dtype: str, shape: tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    volume = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return volume.sum(axis=(1, 2), dtype=np.float64)


def metrics(topas: np.ndarray, gpu: np.ndarray, z_mm: np.ndarray) -> tuple[dict, np.ndarray]:
    peak_index = int(np.argmax(topas))
    peak = float(topas[peak_index])
    valid = topas >= 0.01 * peak
    prepeak = valid & (np.arange(topas.size) <= peak_index)
    tail = (z_mm >= z_mm[peak_index] + 5.0) & (z_mm <= z_mm[peak_index] + 20.0)
    relative = np.full(topas.size, np.nan)
    relative[valid] = 100.0 * (gpu[valid] / topas[valid] - 1.0)
    tail_topas = float(topas[tail].sum())
    return {
        "topas_integral_Gy": float(topas.sum()),
        "gpu_integral_Gy": float(gpu.sum()),
        "integral_difference_percent": float(100.0 * (gpu.sum() / topas.sum() - 1.0)),
        "topas_peak_depth_mm": float(z_mm[peak_index]),
        "gpu_peak_depth_mm": float(z_mm[int(np.argmax(gpu))]),
        "peak_at_topas_depth_difference_percent": float(
            100.0 * (gpu[peak_index] / topas[peak_index] - 1.0)
        ),
        "idd_nrmse_percent_above_1pct": float(
            100.0 * np.sqrt(np.mean((gpu[valid] - topas[valid]) ** 2)) / peak
        ),
        "prepeak_pointwise_max_abs_percent_above_1pct": float(
            np.max(np.abs(relative[prepeak]))
        ),
        "prepeak_pointwise_mean_abs_percent_above_1pct": float(
            np.mean(np.abs(relative[prepeak]))
        ),
        "fraction_prepeak_bins_within_1pct": float(
            np.mean(np.abs(relative[prepeak]) < 1.0)
        ),
        "tail_5_to_20mm_integral_difference_percent": (
            float(100.0 * (gpu[tail].sum() / tail_topas - 1.0)) if tail_topas > 0.0 else None
        ),
    }, relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topas-root", type=Path, required=True)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=100_000)
    parser.add_argument("--energy-mevu", type=int, default=400)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=800)
    parser.add_argument("--dx-mm", type=float, default=0.4)
    parser.add_argument("--dy-mm", type=float, default=0.4)
    parser.add_argument("--dz-mm", type=float, default=0.5)
    parser.add_argument(
        "--provenance", type=Path, action="append", default=[],
        help="Additional configuration/package/rate input to hash into analysis.json",
    )
    args = parser.parse_args()
    if args.histories <= 0 or args.energy_mevu <= 0 or min(args.nx, args.ny, args.nz) <= 0:
        raise ValueError("histories, energy, and grid dimensions must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.nz, args.ny, args.nx)
    z_mm = (np.arange(args.nz) + 0.5) * args.dz_mm
    report = {
        "schema_version": 1,
        "histories": args.histories,
        "energy_MeV_per_u": args.energy_mevu,
        "grid": {"shape": [args.nx, args.ny, args.nz],
                 "spacing_mm": [args.dx_mm, args.dy_mm, args.dz_mm]},
        "scorer": "3D DoseToMedium; IDD is X/Y sum",
        "statistics": {
            "confidence_intervals": "unavailable_single_seed",
            "required_for_ci": "two or more independent TOPAS and GPU seeds",
        },
        "inputs": [],
        "categories": [],
    }
    for path in args.provenance:
        report["inputs"].append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    curves = []
    topas_category_sum = np.zeros(args.nz, dtype=np.float64)
    gpu_category_sum = np.zeros(args.nz, dtype=np.float64)
    for label, topas_name, gpu_name in CATEGORIES:
        topas_path = args.topas_root / f"topas_{topas_name}_e{args.energy_mevu}.bin"
        gpu_path = args.gpu_root / f"species_{gpu_name}.raw"
        topas_idd = load_idd(topas_path, "<f8", shape)
        gpu_idd = load_idd(gpu_path, "<f4", shape)
        row, relative = metrics(topas_idd, gpu_idd, z_mm)
        row["category"] = label
        report["categories"].append(row)
        report["inputs"].extend((
            {"path": str(topas_path), "bytes": topas_path.stat().st_size,
             "sha256": sha256(topas_path)},
            {"path": str(gpu_path), "bytes": gpu_path.stat().st_size,
             "sha256": sha256(gpu_path)},
        ))
        topas_category_sum += topas_idd
        gpu_category_sum += gpu_idd
        curves.append((label, topas_idd, gpu_idd, relative, row["topas_peak_depth_mm"]))

    other_path = args.gpu_root / "species_secondary_other_charged.raw"
    total_path = args.gpu_root / "voxel_dose.raw"
    other_idd = load_idd(other_path, "<f4", shape)
    total_idd = load_idd(total_path, "<f4", shape)
    reconstructed = gpu_category_sum + other_idd
    topas_total_path = args.topas_root / f"topas_total_e{args.energy_mevu}.bin"
    topas_total_idd = load_idd(topas_total_path, "<f8", shape)
    total_row, total_relative = metrics(topas_total_idd, total_idd, z_mm)
    total_row["category"] = "charged_total"
    report["charged_total"] = total_row
    curves.insert(0, ("Total dose", topas_total_idd, total_idd, total_relative,
                      total_row["topas_peak_depth_mm"]))
    report["topas_species_sum_vs_total"] = {
        "species_sum_integral_Gy": float(topas_category_sum.sum()),
        "total_integral_Gy": float(topas_total_idd.sum()),
        "unassigned_integral_Gy": float(topas_total_idd.sum() - topas_category_sum.sum()),
        "species_sum_difference_percent": float(
            100.0 * (topas_category_sum.sum() / topas_total_idd.sum() - 1.0)
        ),
        "semantics": "The unfiltered TOPAS DoseToMedium scorer is the total reference; filtered species scorers do not form a complete total because electron/positron dose is excluded.",
    }
    report["gpu_charged_category_closure"] = {
        "other_charged_integral_Gy": float(other_idd.sum()),
        "total_integral_Gy": float(total_idd.sum()),
        "reconstructed_integral_Gy": float(reconstructed.sum()),
        "integral_difference_percent": float(
            100.0 * (reconstructed.sum() / total_idd.sum() - 1.0)
        ),
        "maximum_depth_bin_abs_Gy": float(np.max(np.abs(reconstructed - total_idd))),
    }
    z1 = next(row for row in report["categories"] if row["category"] == "Z1_H")
    report["z1_classification_bound"] = {
        "semantics": "GPU Z1 plus all other charged versus TOPAS atomic-number-1; compatibility diagnostic, exact only for pre-Z1-alignment outputs",
        "topas_Z1_integral_Gy": z1["topas_integral_Gy"],
        "gpu_proton_integral_Gy": z1["gpu_integral_Gy"],
        "gpu_other_charged_integral_Gy": float(other_idd.sum()),
        "proton_plus_all_other_difference_percent": float(
            100.0 * ((z1["gpu_integral_Gy"] + other_idd.sum()) / z1["topas_integral_Gy"] - 1.0)
        ),
    }
    for path in (other_path, total_path, topas_total_path):
        report["inputs"].append({"path": str(path), "bytes": path.stat().st_size,
                                 "sha256": sha256(path)})

    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    plot_depth_max_mm = 1.2 * float(report["charged_total"]["topas_peak_depth_mm"])
    axes = axes.ravel()
    for axis, (label, topas_idd, gpu_idd, relative, peak_depth) in zip(axes, curves):
        error_axis = axis.twinx()
        axis.plot(z_mm, topas_idd, label="TOPAS 100k", linewidth=1.4)
        axis.plot(z_mm, gpu_idd, label="GPU CINEL02 100k", linewidth=1.2)
        valid = np.isfinite(relative)
        error_axis.plot(z_mm[valid], relative[valid], color="tab:red", alpha=0.45,
                        linewidth=0.8)
        axis.axvline(peak_depth, color="0.6", linestyle=":", linewidth=0.8)
        axis.set_title(label)
        axis.set_ylabel("IDD (Gy)")
        error_axis.set_ylabel("Error (%)", color="tab:red")
        error_axis.set_ylim(-40.0, 40.0)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Depth (mm)")
        axis.set_xlim(0.0, plot_depth_max_mm)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.suptitle(
        f"{args.energy_mevu} MeV/u species-resolved 3D IDD: GPU CINEL02 vs TOPAS",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(args.output_dir / "species_idd_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
