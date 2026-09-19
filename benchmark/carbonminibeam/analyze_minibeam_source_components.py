#!/usr/bin/env python3
"""Diagnose minibeam dose by ion species and immediate birth material.

The GPU input is the optional 17-category MHD output written beside the
charged-origin maps.  TOPAS total and charged-component grids are optional;
when supplied, every comparison is normalized per incident history and no
fitted dose scale is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from compare_gpu_topas_dose import (
    fixed_minibeam_region_masks, parse_mhd,
)


GPU_LABELS = (
    "primary_c12",
    "copper_secondary_c12", "copper_p", "copper_d", "copper_t",
    "copper_he3", "copper_he4", "copper_heavy", "copper_other",
    "water_secondary_c12", "water_p", "water_d", "water_t",
    "water_he3", "water_he4", "water_heavy", "water_other",
)

TOPAS_COMPONENTS = {
    "primary_c12": "component_primary_c.bin",
    "z1": "component_z1.bin",
    "helium": "component_helium.bin",
    # The existing TOPAS scorer groups every secondary carbon isotope, while
    # the GPU diagnostic intentionally separates only C12.  Compare their
    # common, taxonomy-safe aggregate instead of mislabelling C10/C11 as C12.
    "z_ge_3_secondary": (
        "component_secondary_c.bin", "component_boron.bin",
        "component_beryllium.bin", "component_lithium.bin"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--component-prefix", default="components")
    parser.add_argument("--gpu-histories", type=int, required=True)
    parser.add_argument("--topas-total", type=Path)
    parser.add_argument("--topas-component-dir", type=Path)
    parser.add_argument("--topas-total-histories", type=int)
    parser.add_argument("--topas-component-histories", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--depth-spacing-mm", type=float, default=0.25)
    parser.add_argument("--lateral-spacing-mm", type=float, default=0.1)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    return parser.parse_args()


def read_grid(path: Path, dtype: str, bins: int) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype).astype(np.float64)
    if values.size != bins * bins:
        raise ValueError(
            f"Expected {bins * bins} values in {path}, got {values.size}")
    return values.reshape(bins, bins)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bins = args.bins
    gpu_total = read_grid(args.gpu_dir / "dose.raw", "<f4", bins)
    gpu = {
        label: read_grid(
            args.gpu_dir /
            f"{args.component_prefix}_component_{label}.raw", "<f4", bins)
        for label in GPU_LABELS
    }
    groups = {
        "primary_c12": gpu["primary_c12"],
        "secondary_c12": (gpu["copper_secondary_c12"] +
                          gpu["water_secondary_c12"]),
        "p": gpu["copper_p"] + gpu["water_p"],
        "d": gpu["copper_d"] + gpu["water_d"],
        "t": gpu["copper_t"] + gpu["water_t"],
        "he3": gpu["copper_he3"] + gpu["water_he3"],
        "he4": gpu["copper_he4"] + gpu["water_he4"],
        "heavy": gpu["copper_heavy"] + gpu["water_heavy"],
        "other": gpu["copper_other"] + gpu["water_other"],
    }
    groups["z1"] = groups["p"] + groups["d"] + groups["t"]
    groups["helium"] = groups["he3"] + groups["he4"]
    groups["copper_born"] = sum(
        gpu[label] for label in GPU_LABELS if label.startswith("copper_"))
    groups["water_born"] = sum(
        gpu[label] for label in GPU_LABELS if label.startswith("water_"))

    reconstructed = sum(gpu.values())
    difference = reconstructed - gpu_total
    nonzero = gpu_total != 0.0
    mhd = parse_mhd(args.gpu_dir / "dose.mhd")
    spacing = mhd["spacing_mm"]
    offset = mhd["offset_mm"]
    assert isinstance(spacing, list) and isinstance(offset, list)
    depth_mm = offset[2] + np.arange(bins) * spacing[2]
    x_mm = offset[0] + np.arange(bins) * spacing[0]
    region_masks = fixed_minibeam_region_masks(x_mm, args.pitch_mm)
    slab_bins = max(1, int(round(1.0 / spacing[2])))
    smooth_total = uniform_filter1d(
        gpu_total, size=slab_bins, axis=0, mode="nearest")
    smooth_groups = {
        name: uniform_filter1d(values, size=slab_bins, axis=0, mode="nearest")
        for name, values in groups.items()
    }

    topas_total = None
    if args.topas_total is not None:
        if args.topas_total_histories is None:
            raise ValueError("--topas-total requires --topas-total-histories")
        topas_total = read_grid(args.topas_total, "<f8", bins)
        topas_total *= args.gpu_histories / args.topas_total_histories
        bragg_search = depth_mm >= 20.0
        bragg_index = np.flatnonzero(bragg_search)[
            np.argmax(topas_total.sum(axis=1)[bragg_search])]
    else:
        bragg_search = depth_mm >= 20.0
        bragg_index = np.flatnonzero(bragg_search)[
            np.argmax(gpu_total.sum(axis=1)[bragg_search])]
    bragg_depth = float(depth_mm[bragg_index])

    metrics: dict = {
        "normalization": "absolute Gy per incident history; no fitted scale",
        "gpu_histories": args.gpu_histories,
        "bragg_reference_depth_mm": bragg_depth,
        "closure": {
            "global_relative_difference": ratio(
                abs(float(reconstructed.sum() - gpu_total.sum())),
                float(abs(gpu_total).sum())),
            "maximum_absolute_voxel_difference_Gy": float(
                np.max(np.abs(difference))),
            "maximum_relative_nonzero_voxel_difference": float(
                np.max(np.abs(difference[nonzero]) /
                       np.abs(gpu_total[nonzero]))) if np.any(nonzero) else 0.0,
        },
        "global_gpu_fraction": {
            name: float(values.sum() / gpu_total.sum())
            for name, values in groups.items()
        },
        "fixed_regions_at_bragg": {},
        "bragg_pm2mm_component_comparison": {},
        "topas_component_taxonomy_note": (
            "TOPAS secondary carbon contains all carbon isotopes; GPU "
            "secondary_c12 and heavy are therefore compared only through the "
            "combined z_ge_3_secondary category."
        ),
    }

    smooth_topas = (uniform_filter1d(
        topas_total, size=slab_bins, axis=0, mode="nearest")
                    if topas_total is not None else None)
    for region, mask in region_masks.items():
        gpu_region_total = float(smooth_total[bragg_index, mask].sum())
        entry = {
            "gpu_total_Gy": gpu_region_total,
            "gpu_fraction": {
                name: ratio(float(values[bragg_index, mask].sum()),
                            gpu_region_total)
                for name, values in smooth_groups.items()
            },
        }
        if smooth_topas is not None:
            topas_value = float(smooth_topas[bragg_index, mask].sum())
            entry.update({
                "topas_total_Gy": topas_value,
                "gpu_over_topas_total": ratio(gpu_region_total, topas_value),
                "topas_minus_gpu_Gy": topas_value - gpu_region_total,
            })
        metrics["fixed_regions_at_bragg"][region] = entry

    if args.topas_component_dir is not None:
        if args.topas_component_histories is None:
            raise ValueError(
                "--topas-component-dir requires --topas-component-histories")
        topas_components = {}
        component_scale = args.gpu_histories / args.topas_component_histories
        for name, filenames in TOPAS_COMPONENTS.items():
            if isinstance(filenames, str):
                values = read_grid(
                    args.topas_component_dir / filenames, "<f8", bins)
            else:
                values = sum(read_grid(
                    args.topas_component_dir / filename, "<f8", bins)
                             for filename in filenames)
            topas_components[name] = values * component_scale
        gpu_comparable = {
            "primary_c12": groups["primary_c12"],
            "z1": groups["z1"],
            "helium": groups["helium"],
            "z_ge_3_secondary": groups["secondary_c12"] + groups["heavy"],
        }
        all_helium_path = args.gpu_dir / (
            f"{args.component_prefix}_secondary_helium.raw")
        if all_helium_path.is_file():
            # The 17-way map isolates He3/He4 and leaves rarer He isotopes in
            # `other`; the legacy charged-origin map provides taxonomy parity
            # with TOPAS's all-Z=2 helium scorer.
            gpu_comparable["helium"] = read_grid(
                all_helium_path, "<f4", bins)
        depth_mask = np.abs(depth_mm - bragg_depth) <= 2.0
        for region, lateral_mask in region_masks.items():
            metrics["bragg_pm2mm_component_comparison"][region] = {}
            for name in TOPAS_COMPONENTS:
                gpu_value = float(
                    gpu_comparable[name][depth_mask][:, lateral_mask].sum())
                topas_value = float(
                    topas_components[name][depth_mask][:, lateral_mask].sum())
                metrics["bragg_pm2mm_component_comparison"][region][name] = {
                    "gpu_Gy": gpu_value,
                    "topas_Gy": topas_value,
                    "gpu_over_topas": ratio(gpu_value, topas_value),
                }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in ("primary_c12", "copper_born", "water_born"):
        axes[0].plot(depth_mm, smooth_groups[name][:, region_masks["peak"]].sum(1),
                     label=name)
        axes[1].plot(depth_mm, smooth_groups[name][:, region_masks["valley"]].sum(1),
                     label=name)
    axes[0].plot(depth_mm, smooth_total[:, region_masks["peak"]].sum(1),
                 color="black", label="GPU total")
    axes[1].plot(depth_mm, smooth_total[:, region_masks["valley"]].sum(1),
                 color="black", label="GPU total")
    if smooth_topas is not None:
        axes[0].plot(depth_mm, smooth_topas[:, region_masks["peak"]].sum(1),
                     color="black", linestyle="--", label="TOPAS total")
        axes[1].plot(depth_mm, smooth_topas[:, region_masks["valley"]].sum(1),
                     color="black", linestyle="--", label="TOPAS total")
    for axis, title in zip(axes, ("fixed peak", "fixed valley")):
        axis.axvline(bragg_depth, color="0.6", linewidth=1)
        axis.set_xlim(max(0.0, bragg_depth - 20.0), bragg_depth + 15.0)
        axis.set_xlabel("water depth [mm]")
        axis.set_ylabel("1 mm-smoothed region dose [Gy]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(args.output_dir / "bragg_source_components.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
