#!/usr/bin/env python3
"""Compare matched 2-D TOPAS and GPU Copper-minibeam dose planes.

TOPAS scores X versus beam-depth Y and writes float64; the GPU MHD plane is
depth Z versus X and writes float32. No dose fitting is applied. Optional
history-count scaling supports otherwise identical unequal-statistics runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import uniform_filter1d


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topas", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-mhd", type=Path)
    parser.add_argument("--topas-header", type=Path)
    parser.add_argument("--topas-histories", type=int)
    parser.add_argument("--gpu-histories", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--lateral-spacing-mm", type=float)
    parser.add_argument("--depth-spacing-mm", type=float)
    parser.add_argument("--slab-mm", type=float, default=1.0)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    parser.add_argument(
        "--normalization-description",
        default="absolute Gy; equal incident histories; no fitted scale",
    )
    parser.add_argument(
        "--profile-depths-mm", type=float, nargs="+"
    )
    return parser.parse_args()


def parse_mhd(path: Path) -> dict[str, list[float] | list[int] | str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    try:
        dimensions = [int(value) for value in values["DimSize"].split()]
        spacing = [float(value) for value in values["ElementSpacing"].split()]
        offset = [float(value) for value in values["Offset"].split()]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Incomplete MHD grid metadata in {path}") from error
    if len(dimensions) != 3 or len(spacing) != 3 or len(offset) != 3:
        raise ValueError(f"Expected 3-D MHD grid metadata in {path}")
    if dimensions[1] != 1:
        raise ValueError(f"Expected one integrated MHD y bin in {path}")
    return {
        "dimensions": dimensions,
        "spacing_mm": spacing,
        "offset_mm": offset,
        "element_file": values.get("ElementDataFile", ""),
    }


def parse_topas_header(path: Path) -> dict[str, tuple[int, float]]:
    axes: dict[str, tuple[int, float]] = {}
    pattern = re.compile(
        r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins?\s+of\s+"
        r"([0-9.eE+-]+)\s+cm\s*$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            axes[match.group(1)] = (
                int(match.group(2)), float(match.group(3)) * 10.0)
    if "X" not in axes or "Y" not in axes:
        raise ValueError(f"Missing TOPAS X/Y grid metadata in {path}")
    return axes


def assert_close(name: str, left: float, right: float) -> None:
    if not np.isclose(left, right, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(f"{name} mismatch: {left} versus {right}")


def resolve_grid(args: argparse.Namespace) -> dict:
    mhd_path = args.gpu_mhd or args.gpu.with_suffix(".mhd")
    topas_header_path = args.topas_header or Path(str(args.topas) + "header")
    mhd = parse_mhd(mhd_path) if mhd_path.is_file() else None
    topas = (parse_topas_header(topas_header_path)
             if topas_header_path.is_file() else None)
    if mhd is None and topas is None:
        if (args.bins is None or args.lateral_spacing_mm is None or
                args.depth_spacing_mm is None):
            raise ValueError(
                "Grid metadata not found; explicitly provide --bins, "
                "--lateral-spacing-mm, and --depth-spacing-mm")
        lateral_bins = depth_bins = args.bins
        lateral_spacing = args.lateral_spacing_mm
        depth_spacing = args.depth_spacing_mm
        lateral_origin = -0.5 * (lateral_bins - 1) * lateral_spacing
        depth_origin = 0.5 * depth_spacing
    elif mhd is not None:
        dimensions = mhd["dimensions"]
        spacing = mhd["spacing_mm"]
        offset = mhd["offset_mm"]
        assert isinstance(dimensions, list)
        assert isinstance(spacing, list)
        assert isinstance(offset, list)
        lateral_bins, depth_bins = int(dimensions[0]), int(dimensions[2])
        lateral_spacing, depth_spacing = spacing[0], spacing[2]
        lateral_origin, depth_origin = offset[0], offset[2]
        element_file = str(mhd["element_file"])
        if element_file and Path(element_file).name != args.gpu.name:
            raise ValueError(
                f"{mhd_path} describes {element_file}, not {args.gpu.name}")
    else:
        assert topas is not None
        lateral_bins, lateral_spacing = topas["X"]
        depth_bins, depth_spacing = topas["Y"]
        lateral_origin = -0.5 * (lateral_bins - 1) * lateral_spacing
        depth_origin = 0.5 * depth_spacing

    if args.bins is not None and (
            lateral_bins != args.bins or depth_bins != args.bins):
        raise ValueError(
            f"--bins={args.bins} disagrees with metadata "
            f"{depth_bins}x{lateral_bins}")
    if args.lateral_spacing_mm is not None:
        assert_close("lateral spacing", args.lateral_spacing_mm, lateral_spacing)
    if args.depth_spacing_mm is not None:
        assert_close("depth spacing", args.depth_spacing_mm, depth_spacing)
    if topas is not None:
        if topas["X"][0] != lateral_bins or topas["Y"][0] != depth_bins:
            raise ValueError(
                "TOPAS and GPU dimensions disagree: "
                f"TOPAS={topas['Y'][0]}x{topas['X'][0]}, "
                f"GPU={depth_bins}x{lateral_bins}")
        assert_close(
            "TOPAS/GPU lateral spacing", topas["X"][1], lateral_spacing)
        assert_close(
            "TOPAS/GPU depth spacing", topas["Y"][1], depth_spacing)
    return {
        "lateral_bins": lateral_bins,
        "depth_bins": depth_bins,
        "lateral_spacing_mm": lateral_spacing,
        "depth_spacing_mm": depth_spacing,
        "lateral_origin_mm": lateral_origin,
        "depth_origin_mm": depth_origin,
        "gpu_mhd": str(mhd_path) if mhd is not None else None,
        "topas_header": str(topas_header_path) if topas is not None else None,
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def local_feature_depth_curves(
    dose: np.ndarray, x_mm: np.ndarray, pitch_mm: float, slab_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return median central peak/valley dose and their median positions.

    Nine central peaks and eight intervening valleys are used.  For every
    expected feature, the local extremum is located within +/-0.7 mm and the
    reported value is the mean of the extremum voxel and its two neighbours.
    A 1 mm moving depth average suppresses finite-history fluctuations without
    mixing adjacent minibeams laterally.
    """
    depth_smoothed = uniform_filter1d(
        dose, size=slab_bins, axis=0, mode="nearest"
    )
    peak_values = []
    peak_positions = []
    valley_values = []
    valley_positions = []
    half_search = 0.7
    for expected in np.arange(-4, 5, dtype=float) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= half_search)
        local = depth_smoothed[:, indices]
        chosen = indices[np.argmax(local, axis=1)]
        rows = np.arange(dose.shape[0])
        values = np.stack(
            [
                depth_smoothed[rows, np.clip(chosen + shift, 0, dose.shape[1] - 1)]
                for shift in (-1, 0, 1)
            ],
            axis=1,
        ).mean(axis=1)
        peak_values.append(values)
        peak_positions.append(x_mm[chosen])
    for expected in (np.arange(-4, 4, dtype=float) + 0.5) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= half_search)
        local = depth_smoothed[:, indices]
        chosen = indices[np.argmin(local, axis=1)]
        rows = np.arange(dose.shape[0])
        values = np.stack(
            [
                depth_smoothed[rows, np.clip(chosen + shift, 0, dose.shape[1] - 1)]
                for shift in (-1, 0, 1)
            ],
            axis=1,
        ).mean(axis=1)
        valley_values.append(values)
        valley_positions.append(x_mm[chosen])
    return (
        np.median(np.stack(peak_values), axis=0),
        np.median(np.stack(valley_values), axis=0),
        np.median(np.stack(peak_positions), axis=0),
        np.median(np.stack(valley_positions), axis=0),
    )


def slab_profile(
    dose: np.ndarray, requested_depth_mm: float,
    depth_coordinates_mm: np.ndarray, slab_bins: int
) -> np.ndarray:
    center = int(np.argmin(np.abs(depth_coordinates_mm - requested_depth_mm)))
    lo = max(0, center - slab_bins // 2)
    hi = min(dose.shape[0], lo + slab_bins)
    return dose[lo:hi].mean(axis=0)


def fixed_minibeam_region_masks(
    x_mm: np.ndarray, pitch_mm: float, field_half_width_mm: float = 18.0
) -> dict[str, np.ndarray]:
    """Return the canonical fixed ROIs, including their boundary rule.

    Coordinates must come from grid metadata. Peak owns |folded x| < 0.25 mm,
    shoulder owns [0.25, 0.9), and valley owns [0.9, 1.8].
    """
    folded_x = (x_mm + 0.5 * pitch_mm) % pitch_mm - 0.5 * pitch_mm
    central = np.abs(x_mm) <= field_half_width_mm
    return {
        "peak": central & (np.abs(folded_x) < 0.25),
        "shoulder": central & (np.abs(folded_x) >= 0.25) &
                    (np.abs(folded_x) < 0.9),
        "valley": central & (np.abs(folded_x) >= 0.9) &
                  (np.abs(folded_x) <= 1.8),
    }


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = resolve_grid(args)
    lateral_bins = grid["lateral_bins"]
    depth_bins = grid["depth_bins"]
    expected = lateral_bins * depth_bins
    topas_flat = np.fromfile(args.topas, dtype="<f8")
    gpu_flat = np.fromfile(args.gpu, dtype="<f4").astype(np.float64)
    if topas_flat.size != expected or gpu_flat.size != expected:
        raise ValueError(
            f"Expected {expected} values, got TOPAS={topas_flat.size}, GPU={gpu_flat.size}"
        )
    if (args.topas_histories is None) != (args.gpu_histories is None):
        raise ValueError(
            "--topas-histories and --gpu-histories must be provided together"
        )
    if args.topas_histories is not None:
        if args.topas_histories <= 0 or args.gpu_histories <= 0:
            raise ValueError("history counts must be positive")
        topas_flat *= args.gpu_histories / args.topas_histories
    topas = topas_flat.reshape(depth_bins, lateral_bins)
    gpu = gpu_flat.reshape(depth_bins, lateral_bins)
    lateral_spacing = grid["lateral_spacing_mm"]
    depth_spacing = grid["depth_spacing_mm"]
    x_mm = (grid["lateral_origin_mm"] +
            np.arange(lateral_bins) * lateral_spacing)
    depth_mm = (grid["depth_origin_mm"] +
                np.arange(depth_bins) * depth_spacing)
    slab_bins = max(1, int(round(args.slab_mm / depth_spacing)))

    topas_idd = topas.sum(axis=1)
    gpu_idd = gpu.sum(axis=1)
    # At high energy the entrance plateau can narrowly exceed the distal peak
    # after collimator attenuation.  Bragg depth is therefore the maximum
    # beyond the first 20 mm, rather than the global maximum including entrance.
    bragg_search = depth_mm >= 20.0
    topas_bragg_index = np.flatnonzero(bragg_search)[np.argmax(topas_idd[bragg_search])]
    gpu_bragg_index = np.flatnonzero(bragg_search)[np.argmax(gpu_idd[bragg_search])]
    topas_bragg_depth = float(depth_mm[topas_bragg_index])
    if args.profile_depths_mm is None:
        args.profile_depths_mm = [
            0.5, 0.5 * topas_bragg_depth, topas_bragg_depth,
            min(depth_mm[-1], topas_bragg_depth + 10.0),
        ]
    topas_lateral = topas.sum(axis=0)
    gpu_lateral = gpu.sum(axis=0)
    topas_peak, topas_valley, topas_peak_x, topas_valley_x = local_feature_depth_curves(
        topas, x_mm, args.pitch_mm, slab_bins
    )
    gpu_peak, gpu_valley, gpu_peak_x, gpu_valley_x = local_feature_depth_curves(
        gpu, x_mm, args.pitch_mm, slab_bins
    )
    fixed_masks = fixed_minibeam_region_masks(x_mm, args.pitch_mm)
    topas_smoothed = uniform_filter1d(
        topas, size=slab_bins, axis=0, mode="nearest")
    gpu_smoothed = uniform_filter1d(
        gpu, size=slab_bins, axis=0, mode="nearest")
    fixed_region_curves = {
        name: {
            "topas_integral_Gy": topas_smoothed[:, mask].sum(axis=1),
            "gpu_integral_Gy": gpu_smoothed[:, mask].sum(axis=1),
            "voxel_count": int(np.count_nonzero(mask)),
        }
        for name, mask in fixed_masks.items()
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        topas_pvdr = np.divide(topas_peak, topas_valley)
        gpu_pvdr = np.divide(gpu_peak, gpu_valley)

    topas_sum = float(topas.sum())
    gpu_sum = float(gpu.sum())
    high_depth = topas_idd >= 0.01 * topas_idd.max()
    metrics = {
        "comparison": args.normalization_description,
        "grid": {
            "shape_depth_lateral": [depth_bins, lateral_bins],
            "depth_spacing_mm": depth_spacing,
            "lateral_spacing_mm": lateral_spacing,
            "depth_origin_mm": grid["depth_origin_mm"],
            "lateral_origin_mm": grid["lateral_origin_mm"],
            "gpu_mhd": grid["gpu_mhd"],
            "topas_header": grid["topas_header"],
            "peak_valley_depth_smoothing_mm": args.slab_mm,
            "central_peak_count": 9,
            "central_valley_count": 8,
        },
        "dose_sum_Gy": {"topas": topas_sum, "gpu": gpu_sum},
        "gpu_over_topas_dose_sum": gpu_sum / topas_sum,
        "two_dimensional_pearson": pearson(topas, gpu),
        "two_dimensional_l1_over_topas": float(np.abs(gpu - topas).sum() / topas_sum),
        "idd_pearson": pearson(topas_idd, gpu_idd),
        "idd_l1_over_topas": float(np.abs(gpu_idd - topas_idd).sum() / topas_idd.sum()),
        "lateral_integral_pearson": pearson(topas_lateral, gpu_lateral),
        "lateral_integral_l1_over_topas": float(
            np.abs(gpu_lateral - topas_lateral).sum() / topas_lateral.sum()
        ),
        "bragg_peak_depth_mm": {
            "topas": topas_bragg_depth,
            "gpu": float(depth_mm[gpu_bragg_index]),
        },
        "median_peak_position_absolute_difference_mm_high_dose": float(
            np.median(np.abs(gpu_peak_x[high_depth] - topas_peak_x[high_depth]))
        ),
        "median_valley_position_absolute_difference_mm_high_dose": float(
            np.median(np.abs(gpu_valley_x[high_depth] - topas_valley_x[high_depth]))
        ),
        "selected_depths": [],
        "fixed_regions": {
            "central_half_width_mm": 18.0,
            "peak_abs_folded_x_mm": [0.0, 0.25],
            "shoulder_abs_folded_x_mm": [0.25, 0.9],
            "valley_abs_folded_x_mm": [0.9, 1.8],
            "selected_depths": [],
        },
    }
    selected_rows = []
    for requested_depth in args.profile_depths_mm:
        index = int(np.argmin(np.abs(depth_mm - requested_depth)))
        topas_profile = slab_profile(topas, requested_depth, depth_mm, slab_bins)
        gpu_profile = slab_profile(gpu, requested_depth, depth_mm, slab_bins)
        profile_roi = np.abs(x_mm) <= 30.0
        profile_pearson = (
            pearson(topas_profile[profile_roi], gpu_profile[profile_roi])
            if np.std(gpu_profile[profile_roi]) > 0.0
            else None
        )
        row = {
            "requested_depth_mm": requested_depth,
            "grid_depth_mm": float(depth_mm[index]),
            "topas_peak_Gy": float(topas_peak[index]),
            "gpu_peak_Gy": float(gpu_peak[index]),
            "peak_gpu_over_topas": float(gpu_peak[index] / topas_peak[index])
            if topas_peak[index] > 0
            else None,
            "topas_valley_Gy": float(topas_valley[index]),
            "gpu_valley_Gy": float(gpu_valley[index]),
            "valley_gpu_over_topas": float(gpu_valley[index] / topas_valley[index])
            if topas_valley[index] > 0
            else None,
            "topas_pvdr": float(topas_pvdr[index]),
            "gpu_pvdr": float(gpu_pvdr[index]) if np.isfinite(gpu_pvdr[index]) else None,
            "pvdr_gpu_over_topas": float(gpu_pvdr[index] / topas_pvdr[index])
            if np.isfinite(gpu_pvdr[index]) and topas_pvdr[index] > 0
            else None,
            "lateral_profile_pearson_within_30mm": profile_pearson,
            "lateral_profile_l1_over_topas_within_30mm": float(
                np.abs(gpu_profile[profile_roi] - topas_profile[profile_roi]).sum()
                / topas_profile[profile_roi].sum()
            ),
        }
        selected_rows.append(row)
        metrics["selected_depths"].append(row)
        fixed_row = {"depth_mm": float(depth_mm[index])}
        for name, values in fixed_region_curves.items():
            topas_value = float(values["topas_integral_Gy"][index])
            gpu_value = float(values["gpu_integral_Gy"][index])
            fixed_row[f"topas_{name}_integral_Gy"] = topas_value
            fixed_row[f"gpu_{name}_integral_Gy"] = gpu_value
            fixed_row[f"{name}_gpu_over_topas"] = (
                gpu_value / topas_value if topas_value > 0.0 else None)
        metrics["fixed_regions"]["selected_depths"].append(fixed_row)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "selected_depth_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    with (args.output_dir / "depth_curves.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "depth_mm", "topas_idd_Gy", "gpu_idd_Gy", "topas_peak_Gy",
                "gpu_peak_Gy", "topas_valley_Gy", "gpu_valley_Gy",
                "topas_pvdr", "gpu_pvdr",
            ]
        )
        writer.writerows(
            zip(
                depth_mm, topas_idd, gpu_idd, topas_peak, gpu_peak,
                topas_valley, gpu_valley, topas_pvdr, gpu_pvdr,
            )
        )
    with (args.output_dir / "fixed_region_curves.csv").open(
            "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "depth_mm",
            "topas_peak_integral_Gy", "gpu_peak_integral_Gy",
            "topas_shoulder_integral_Gy", "gpu_shoulder_integral_Gy",
            "topas_valley_integral_Gy", "gpu_valley_integral_Gy",
        ])
        writer.writerows(zip(
            depth_mm,
            fixed_region_curves["peak"]["topas_integral_Gy"],
            fixed_region_curves["peak"]["gpu_integral_Gy"],
            fixed_region_curves["shoulder"]["topas_integral_Gy"],
            fixed_region_curves["shoulder"]["gpu_integral_Gy"],
            fixed_region_curves["valley"]["topas_integral_Gy"],
            fixed_region_curves["valley"]["gpu_integral_Gy"],
        ))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    positive = np.concatenate((topas[topas > 0], gpu[gpu > 0]))
    vmax = float(max(topas.max(), gpu.max()))
    vmin = max(float(np.percentile(positive, 2)), vmax * 1.0e-5)
    extent = [x_mm[0] - 0.5 * lateral_spacing,
              x_mm[-1] + 0.5 * lateral_spacing,
              depth_mm[-1] + 0.5 * depth_spacing,
              depth_mm[0] - 0.5 * depth_spacing]
    displayed_depth = min(depth_mm[-1], max(90.0, topas_bragg_depth + 20.0))
    for axis, dose, title in zip(axes[0, :2], (topas, gpu), ("TOPAS", "GPU Copper candidate")):
        image = axis.imshow(dose, extent=extent, aspect="auto", cmap="magma",
                            norm=LogNorm(vmin=vmin, vmax=vmax))
        axis.set_xlim(-30, 30)
        axis.set_ylim(displayed_depth, 0)
        axis.set_title(f"{title}: absolute dose")
        axis.set_xlabel("lateral x (mm)")
        axis.set_ylabel("water depth (mm)")
    fig.colorbar(image, ax=axes[0, :2], label="dose per voxel (Gy)", shrink=0.85)
    difference = 100.0 * (gpu - topas) / topas.max()
    diff_image = axes[0, 2].imshow(difference, extent=extent, aspect="auto", cmap="RdBu_r",
                                   vmin=-20, vmax=20)
    axes[0, 2].set_xlim(-30, 30)
    axes[0, 2].set_ylim(displayed_depth, 0)
    axes[0, 2].set_title("GPU - TOPAS (% TOPAS max voxel)")
    axes[0, 2].set_xlabel("lateral x (mm)")
    axes[0, 2].set_ylabel("water depth (mm)")
    fig.colorbar(diff_image, ax=axes[0, 2], label="percentage points", shrink=0.85)

    axes[1, 0].plot(depth_mm, topas_idd, label="TOPAS", lw=1.7)
    axes[1, 0].plot(depth_mm, gpu_idd, label="GPU", lw=1.4)
    axes[1, 0].set_xlim(0, displayed_depth)
    axes[1, 0].set_xlabel("water depth (mm)")
    axes[1, 0].set_ylabel("lateral-integrated dose (Gy)")
    axes[1, 0].set_title("Integrated depth dose (absolute)")
    axes[1, 0].legend()

    axes[1, 1].semilogy(depth_mm, topas_peak, label="TOPAS peak", lw=1.7)
    axes[1, 1].semilogy(depth_mm, gpu_peak, label="GPU peak", lw=1.4)
    axes[1, 1].semilogy(depth_mm, topas_valley, "--", label="TOPAS valley", lw=1.7)
    axes[1, 1].semilogy(depth_mm, gpu_valley, "--", label="GPU valley", lw=1.4)
    axes[1, 1].set_xlim(0, displayed_depth)
    axes[1, 1].set_ylim(max(vmin * 0.1, 1e-10), None)
    axes[1, 1].set_xlabel("water depth (mm)")
    axes[1, 1].set_ylabel("median local dose (Gy)")
    axes[1, 1].set_title(f"Peak and valley dose vs depth ({args.slab_mm:g} mm slab)")
    axes[1, 1].legend(fontsize=8)

    pvdr_mask = high_depth & np.isfinite(topas_pvdr) & np.isfinite(gpu_pvdr)
    axes[1, 2].plot(depth_mm[pvdr_mask], topas_pvdr[pvdr_mask], label="TOPAS", lw=1.7)
    axes[1, 2].plot(depth_mm[pvdr_mask], gpu_pvdr[pvdr_mask], label="GPU", lw=1.4)
    axes[1, 2].set_xlabel("water depth (mm)")
    axes[1, 2].set_ylabel("PVDR")
    axes[1, 2].set_title("Peak-to-valley dose ratio (>1% TOPAS IDD)")
    axes[1, 2].legend()
    fig.suptitle("3 x 3 cm Copper minibeam: TOPAS vs GPU (matched histories, no dose scaling)")
    fig.savefig(args.output_dir / "dose_comparison_overview.png", dpi=180)
    plt.close(fig)

    count = len(args.profile_depths_mm)
    columns = 2
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, requested_depth in zip(axes, args.profile_depths_mm):
        topas_profile = slab_profile(topas, requested_depth, depth_mm, slab_bins)
        gpu_profile = slab_profile(gpu, requested_depth, depth_mm, slab_bins)
        axis.plot(x_mm, topas_profile * 1e3, label="TOPAS", lw=1.6)
        axis.plot(x_mm, gpu_profile * 1e3, label="GPU", lw=1.3)
        axis.set_xlim(-30, 30)
        axis.set_xlabel("lateral x (mm)")
        axis.set_ylabel("mean dose (mGy)")
        axis.set_title(f"depth {requested_depth:g} mm; {args.slab_mm:g} mm slab")
        axis.legend()
    for axis in axes[count:]:
        axis.set_visible(False)
    fig.suptitle("Lateral minibeam profiles (absolute dose, no fitted scale)")
    fig.savefig(args.output_dir / "lateral_profiles.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(x_mm, topas_lateral, label="TOPAS", lw=1.6)
    axis.plot(x_mm, gpu_lateral, label="GPU", lw=1.3)
    axis.set_xlim(-35, 35)
    axis.set_xlabel("lateral x (mm)")
    axis.set_ylabel("depth-integrated dose (Gy)")
    axis.set_title("Depth-integrated lateral profile")
    axis.legend()
    fig.savefig(args.output_dir / "lateral_integral.png", dpi=180)
    plt.close(fig)

    ratio_mask = high_depth & (topas_peak > 0) & (topas_valley > 0)
    ratio_depth = depth_mm[ratio_mask]
    ratio_series = (
        (gpu_idd[ratio_mask] / topas_idd[ratio_mask], "IDD"),
        (gpu_peak[ratio_mask] / topas_peak[ratio_mask], "peak dose"),
        (gpu_valley[ratio_mask] / topas_valley[ratio_mask], "valley dose"),
        (gpu_pvdr[ratio_mask] / topas_pvdr[ratio_mask], "PVDR"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (ratio, title) in zip(axes.ravel(), ratio_series):
        axis.plot(ratio_depth, ratio, lw=1.5)
        axis.axhline(1.0, color="black", lw=0.9, linestyle="--")
        axis.set_xlim(0, displayed_depth)
        axis.set_xlabel("water depth (mm)")
        axis.set_ylabel("GPU / TOPAS")
        axis.set_title(title)
    fig.suptitle("Depth-dependent dose ratios (>1% TOPAS IDD)")
    fig.savefig(args.output_dir / "dose_ratios_vs_depth.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, values in fixed_region_curves.items():
        ratio = np.divide(
            values["gpu_integral_Gy"], values["topas_integral_Gy"],
            out=np.full_like(values["gpu_integral_Gy"], np.nan),
            where=values["topas_integral_Gy"] > 0.0)
        axis.plot(depth_mm[high_depth], ratio[high_depth], label=name)
    axis.axhline(1.0, color="black", lw=0.9, linestyle="--")
    axis.set(xlabel="water depth (mm)", ylabel="GPU / TOPAS",
             title="Fixed folded-x region dose integrals (>1% TOPAS IDD)")
    axis.legend()
    fig.savefig(args.output_dir / "fixed_region_dose_ratios.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
