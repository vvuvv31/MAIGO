#!/usr/bin/env python3
"""Analyze a patient-Y planar minibeam dose comparison in a CT grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_minibeam_planar_gamma import gamma_line
from match_gpu_to_physical_dose import read_mhd


def load_volume(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, values = read_mhd(path)
    nx, ny, nz = (int(value) for value in metadata["DimSize"].split())
    return metadata, np.asarray(values, dtype=np.float64).reshape(nz, ny, nx)


def distal_crossing(depth: np.ndarray, dose: np.ndarray, fraction: float) -> float:
    peak = int(np.argmax(dose))
    target = fraction * float(dose[peak])
    for index in range(peak + 1, dose.size):
        if dose[index] <= target < dose[index - 1]:
            denominator = float(dose[index] - dose[index - 1])
            if denominator == 0.0:
                return float(depth[index])
            weight = (target - dose[index - 1]) / denominator
            return float(depth[index - 1] + weight * (depth[index] - depth[index - 1]))
    return float("nan")


def lateral_profile(
    dose_zyx: np.ndarray,
    depth_centers_mm: np.ndarray,
    requested_depth_mm: float,
    slab_width_mm: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    selected = np.flatnonzero(
        np.abs(depth_centers_mm - requested_depth_mm)
        <= 0.5 * slab_width_mm + 1.0e-12
    )
    if selected.size == 0:
        selected = np.asarray(
            [int(np.argmin(np.abs(depth_centers_mm - requested_depth_mm)))]
        )
    profile = np.sum(
        dose_zyx[:, int(selected[0]) : int(selected[-1]) + 1, :],
        axis=(0, 1),
        dtype=np.float64,
    )
    return profile, (
        float(depth_centers_mm[selected[0]]),
        float(depth_centers_mm[selected[-1]]),
    )


def peak_valley_metrics(
    x_mm: np.ndarray,
    profile: np.ndarray,
    pitch_mm: float,
    slit_count: int,
) -> dict[str, float | int]:
    half = slit_count // 2
    slit_centers = pitch_mm * np.arange(-half, half + 1, dtype=np.float64)
    peak_values: list[float] = []
    for center in slit_centers:
        selected = np.abs(x_mm - center) <= 0.75
        peak_values.append(float(np.max(profile[selected])))
    valley_values = [
        float(profile[int(np.argmin(np.abs(x_mm - midpoint)))])
        for midpoint in 0.5 * (slit_centers[:-1] + slit_centers[1:])
    ]
    mean_peak = float(np.mean(peak_values))
    mean_valley = float(np.mean(valley_values))
    return {
        "peak_count": len(peak_values),
        "mean_peak_Gy": mean_peak,
        "mean_valley_Gy": mean_valley,
        "pvdr": mean_peak / mean_valley if mean_valley > 0.0 else float("inf"),
        "peak_cv": (
            float(np.std(peak_values, ddof=1) / mean_peak)
            if len(peak_values) > 1 and mean_peak > 0.0
            else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--depths-mm", type=float, nargs="+", default=[5.0, 50.0, 100.0, 150.0]
    )
    parser.add_argument("--slab-width-mm", type=float, default=2.0)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    parser.add_argument("--slit-count", type=int, default=15)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_meta, reference = load_volume(args.reference)
    gpu_meta, gpu = load_volume(args.gpu)
    if reference.shape != gpu.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {gpu.shape}")
    for key in ("DimSize", "ElementSpacing", "Offset"):
        if ref_meta[key] != gpu_meta[key]:
            raise ValueError(f"grid mismatch for {key}")

    spacing_x, spacing_depth, _spacing_z = (
        float(value) for value in ref_meta["ElementSpacing"].split()
    )
    nz, ndepth, nx = reference.shape
    x_mm = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * spacing_x
    depth_mm = (np.arange(ndepth, dtype=np.float64) + 0.5) * spacing_depth
    reference_idd = np.sum(reference, axis=(0, 2), dtype=np.float64)
    gpu_idd = np.sum(gpu, axis=(0, 2), dtype=np.float64)
    reference_peak = float(np.max(reference_idd))
    selected_idd = reference_idd >= args.threshold_percent * 0.01 * reference_peak
    idd_rmse = float(
        np.sqrt(np.mean((gpu_idd[selected_idd] - reference_idd[selected_idd]) ** 2))
        / reference_peak
    )
    idd_correlation = float(np.corrcoef(reference_idd, gpu_idd)[0, 1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 1, figsize=(9.2, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    axes[0].plot(depth_mm, reference_idd, label="TOPAS", color="#1967B3")
    axes[0].plot(depth_mm, gpu_idd, label="GPU", color="#D55E00", linestyle="--")
    axes[0].set_ylabel("Integrated dose (Gy)")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    difference = 100.0 * (gpu_idd - reference_idd) / reference_peak
    axes[1].axhspan(-3.0, 3.0, color="#d9ead3", alpha=0.75)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].plot(depth_mm, difference, color="#6A3D9A")
    axes[1].set_xlabel("Depth from CT bounding-box entrance (mm)")
    axes[1].set_ylabel("GPU − TOPAS\n(% TOPAS peak)")
    axes[1].grid(alpha=0.22)
    figure.suptitle(
        "20022516 lung CT: integrated planar-minibeam depth dose\n"
        f"IDD correlation={idd_correlation:.5f}, peak-region NRMSE={100*idd_rmse:.2f}%"
    )
    figure.tight_layout()
    depth_path = args.output_dir / "depth_dose.png"
    figure.savefig(depth_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    metrics: dict[str, object] = {
        "reference": str(args.reference),
        "gpu": str(args.gpu),
        "normalization": "absolute same histories; no fitted dose scale",
        "depth_axis": "patient +Y",
        "depth_spacing_mm": spacing_depth,
        "idd": {
            "correlation": idd_correlation,
            "nrmse_fraction_of_peak_thr10": idd_rmse,
            "peak_depth_topas_mm": float(depth_mm[int(np.argmax(reference_idd))]),
            "peak_depth_gpu_mm": float(depth_mm[int(np.argmax(gpu_idd))]),
            "r80_topas_mm": distal_crossing(depth_mm, reference_idd, 0.8),
            "r80_gpu_mm": distal_crossing(depth_mm, gpu_idd, 0.8),
        },
        "profiles": [],
    }
    figure, axes = plt.subplots(
        len(args.depths_mm), 2, figsize=(11.0, 3.0 * len(args.depths_mm)),
        squeeze=False,
    )
    for row, requested_depth in enumerate(args.depths_mm):
        ref_profile, span = lateral_profile(
            reference, depth_mm, requested_depth, args.slab_width_mm
        )
        gpu_profile, _ = lateral_profile(
            gpu, depth_mm, requested_depth, args.slab_width_mm
        )
        topas_pv = peak_valley_metrics(
            x_mm, ref_profile, args.pitch_mm, args.slit_count
        )
        gpu_pv = peak_valley_metrics(
            x_mm, gpu_profile, args.pitch_mm, args.slit_count
        )
        gamma_metrics: dict[str, object] = {}
        threshold = args.threshold_percent * 0.01
        for dose_percent, distance_mm in (
            (3.0, 0.3),
            (3.0, 0.5),
            (2.0, 0.5),
        ):
            label = f"{dose_percent:g}pct_{distance_mm:g}mm"
            gamma_metrics[f"global_{label}"] = gamma_line(
                ref_profile, gpu_profile, spacing_x, threshold,
                dose_percent, distance_mm, False,
            )
            gamma_metrics[f"local_{label}"] = gamma_line(
                ref_profile, gpu_profile, spacing_x, threshold,
                dose_percent, distance_mm, True,
            )
        metrics["profiles"].append(
            {
                "requested_depth_mm": requested_depth,
                "sampled_depth_span_mm": list(span),
                "topas": topas_pv,
                "gpu": gpu_pv,
                "pvdr_relative_difference": (
                    gpu_pv["pvdr"] / topas_pv["pvdr"] - 1.0
                ),
                "gamma_1d_across_slits_thr10": gamma_metrics,
            }
        )

        axes[row, 0].plot(x_mm, ref_profile, color="#1967B3", label="TOPAS")
        axes[row, 0].plot(
            x_mm, gpu_profile, color="#D55E00", linestyle="--", label="GPU"
        )
        axes[row, 0].set_xlim(-30.0, 30.0)
        axes[row, 0].set_ylabel("Dose (Gy)")
        axes[row, 0].set_title(
            f"Depth {requested_depth:g} mm ({span[0]:g}–{span[1]:g} mm)"
        )
        axes[row, 0].grid(alpha=0.22)
        axes[row, 1].axhspan(-3.0, 3.0, color="#d9ead3", alpha=0.75)
        axes[row, 1].axhline(0.0, color="black", linewidth=0.7)
        axes[row, 1].plot(
            x_mm, 100.0 * (gpu_profile - ref_profile) / max(ref_profile),
            color="#6A3D9A",
        )
        axes[row, 1].set_xlim(-30.0, 30.0)
        axes[row, 1].set_ylabel("Difference\n(% profile max)")
        axes[row, 1].grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Across-slit position relative to collimator centre (mm)")
    axes[-1, 1].set_xlabel("Across-slit position relative to collimator centre (mm)")
    figure.suptitle("Absolute lateral minibeam peak/valley profiles in lung CT")
    figure.tight_layout()
    lateral_path = args.output_dir / "lateral_peak_valley.png"
    figure.savefig(lateral_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(depth_path)
    print(lateral_path)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
