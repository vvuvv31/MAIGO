#!/usr/bin/env python3
"""Compare aligned TOPAS/GPU minibeam dose in a heterogeneous slab phantom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_minibeam_dose import read_mhd
from compare_minibeam_planar_gamma import gamma_line


def smooth(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return values.copy()
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(values, kernel, mode="same")


def distal_r80(depth: np.ndarray, dose: np.ndarray) -> float:
    peak = int(np.argmax(dose))
    target = 0.8 * float(dose[peak])
    for index in range(peak, dose.size - 1):
        left, right = float(dose[index]), float(dose[index + 1])
        if left >= target > right:
            fraction = (target - left) / (right - left)
            return float(depth[index] + fraction * (depth[index + 1] - depth[index]))
    return float("nan")


def profile_metrics(reference: np.ndarray, evaluation: np.ndarray) -> dict[str, float]:
    maximum = float(np.max(reference))
    return {
        "integral_ratio": float(np.sum(evaluation) / np.sum(reference)),
        "normalized_L1_percent": float(
            100.0 * np.sum(np.abs(evaluation - reference)) / np.sum(reference)
        ),
        "RMSE_percent_of_reference_max": float(
            100.0 * np.sqrt(np.mean((evaluation - reference) ** 2)) / maximum
        ),
        "pearson_r": float(np.corrcoef(reference, evaluation)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-smoothing-mm", type=float, default=1.5)
    parser.add_argument("--lateral-slab-mm", type=float, default=5.0)
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        help="Optional output prefix for *_depth.png and *_lateral.png",
    )
    parser.add_argument(
        "--lateral-depths-mm", type=float, nargs="+", default=[10.0, 60.0, 70.0, 75.0]
    )
    args = parser.parse_args()

    reference = read_mhd(args.reference)
    evaluation = read_mhd(args.evaluation)
    for field in ("shape_xyz", "spacing_xyz_mm", "offset_xyz_mm"):
        if getattr(reference, field) != getattr(evaluation, field):
            raise ValueError(f"Grid mismatch for {field}")

    sx, _, sz = reference.spacing_xyz_mm
    ox, _, oz = reference.offset_xyz_mm
    nx, _, nz = reference.shape_xyz
    x = ox + np.arange(nx, dtype=np.float64) * sx
    depth = oz + np.arange(nz, dtype=np.float64) * sz
    topas_depth = np.sum(reference.data_zyx, axis=(1, 2), dtype=np.float64)
    gpu_depth = np.sum(evaluation.data_zyx, axis=(1, 2), dtype=np.float64)
    smoothing_bins = max(1, int(round(args.depth_smoothing_mm / sz)))
    topas_depth_s = smooth(topas_depth, smoothing_bins)
    gpu_depth_s = smooth(gpu_depth, smoothing_bins)

    result: dict[str, object] = {
        "normalization": "absolute DoseToMedium; no fitted dose scale",
        "grid": {
            "shape_xyz": list(reference.shape_xyz),
            "spacing_xyz_mm": list(reference.spacing_xyz_mm),
        },
        "depth": profile_metrics(topas_depth_s, gpu_depth_s),
        "regions": {},
        "lateral_slabs": [],
    }
    result["depth"].update(
        {
            "smoothing_mm": args.depth_smoothing_mm,
            "topas_R80_mm": distal_r80(depth, topas_depth_s),
            "gpu_R80_mm": distal_r80(depth, gpu_depth_s),
            "global_gamma_3pct_1mm_thr10": gamma_line(
                topas_depth_s, gpu_depth_s, sz, 0.10, 3.0, 1.0, False
            )["pass_percent"],
            "local_gamma_3pct_1mm_thr10": gamma_line(
                topas_depth_s, gpu_depth_s, sz, 0.10, 3.0, 1.0, True
            )["pass_percent"],
        }
    )
    result["depth"]["delta_R80_mm"] = (
        result["depth"]["gpu_R80_mm"] - result["depth"]["topas_R80_mm"]
    )

    for name, begin, end in (
        ("water_before", 0.0, 50.0),
        ("bone", 50.0, 70.0),
        ("water_after", 70.0, 150.0),
    ):
        selected = (depth >= begin) & (depth < end)
        result["regions"][name] = {
            "depth_span_mm": [begin, end],
            **profile_metrics(topas_depth[selected], gpu_depth[selected]),
        }

    lateral_profiles: list[tuple[float, np.ndarray, np.ndarray]] = []
    for requested_depth in args.lateral_depths_mm:
        selected = np.abs(depth - requested_depth) <= 0.5 * args.lateral_slab_mm
        topas = np.sum(reference.data_zyx[selected, 0, :], axis=0, dtype=np.float64)
        gpu = np.sum(evaluation.data_zyx[selected, 0, :], axis=0, dtype=np.float64)
        lateral_profiles.append((requested_depth, topas, gpu))
        entry = {
            "depth_mm": requested_depth,
            "summed_depth_span_mm": [
                float(depth[np.flatnonzero(selected)[0]]),
                float(depth[np.flatnonzero(selected)[-1]]),
            ],
            **profile_metrics(topas, gpu),
            "global_gamma_3pct_0p4mm_thr5": gamma_line(
                topas, gpu, sx, 0.05, 3.0, 0.4, False
            )["pass_percent"],
            "local_gamma_3pct_0p4mm_thr5": gamma_line(
                topas, gpu, sx, 0.05, 3.0, 0.4, True
            )["pass_percent"],
        }
        # Central five slit peaks and the four intervening valleys. A ±0.2 mm
        # average is robust to the 0.2 mm scorer grid and MC noise.
        peak_positions = np.arange(-2, 3, dtype=np.float64) * 3.6
        valley_positions = (np.arange(-2, 2, dtype=np.float64) + 0.5) * 3.6

        def mean_at(values: np.ndarray, positions: np.ndarray) -> float:
            samples = [
                float(np.mean(values[np.abs(x - position) <= 0.2 + 1.0e-12]))
                for position in positions
            ]
            return float(np.mean(samples))

        for label, values in (("topas", topas), ("gpu", gpu)):
            peak = mean_at(values, peak_positions)
            valley = mean_at(values, valley_positions)
            entry[f"{label}_peak_mean_Gy"] = peak
            entry[f"{label}_valley_mean_Gy"] = valley
            entry[f"{label}_PVDR"] = peak / valley
        entry["PVDR_relative_difference_percent"] = (
            100.0 * entry["gpu_PVDR"] / entry["topas_PVDR"] - 100.0
        )
        result["lateral_slabs"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.plot_prefix is not None:
        args.plot_prefix.parent.mkdir(parents=True, exist_ok=True)
        figure, (dose_axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(9.0, 7.0),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
        dose_axis.plot(depth, topas_depth_s, label="TOPAS", linewidth=1.8)
        dose_axis.plot(depth, gpu_depth_s, label="GPU", linewidth=1.5)
        dose_axis.axvspan(50.0, 70.0, color="0.75", alpha=0.45, label="bone")
        dose_axis.set_ylabel("Integrated transverse dose (Gy)")
        dose_axis.set_title(
            "200 MeV/u minibeam: water / compact bone / water depth dose"
        )
        dose_axis.legend()
        ratio = np.divide(
            gpu_depth_s,
            topas_depth_s,
            out=np.full_like(gpu_depth_s, np.nan),
            where=topas_depth_s > 0.02 * np.max(topas_depth_s),
        )
        ratio_axis.plot(depth, ratio, color="tab:purple", linewidth=1.2)
        ratio_axis.axhline(1.0, color="black", linewidth=0.8)
        ratio_axis.axhspan(0.97, 1.03, color="tab:green", alpha=0.15)
        ratio_axis.axvspan(50.0, 70.0, color="0.75", alpha=0.45)
        ratio_axis.set_ylim(0.7, 1.3)
        ratio_axis.set_xlabel("Depth (mm)")
        ratio_axis.set_ylabel("GPU / TOPAS")
        depth_plot = Path(f"{args.plot_prefix}_depth.png")
        figure.savefig(depth_plot, dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(
            2, 2, figsize=(10.5, 7.5), sharex=True, constrained_layout=True
        )
        for axis, (requested_depth, topas, gpu), entry in zip(
            axes.flat, lateral_profiles, result["lateral_slabs"], strict=True
        ):
            axis.plot(x, topas, label="TOPAS", linewidth=1.5)
            axis.plot(x, gpu, label="GPU", linewidth=1.3)
            axis.set_title(
                f"{requested_depth:g} mm; PVDR Δ "
                f"{entry['PVDR_relative_difference_percent']:+.1f}%"
            )
            axis.set_ylabel("Dose in 5 mm slab (Gy)")
            axis.grid(alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("Position across slits (mm)")
        axes[0, 0].legend()
        figure.suptitle("Absolute lateral minibeam dose (common range within each depth)")
        lateral_plot = Path(f"{args.plot_prefix}_lateral.png")
        figure.savefig(lateral_plot, dpi=180)
        plt.close(figure)
        print(f"Wrote {depth_plot}")
        print(f"Wrote {lateral_plot}")

    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
