#!/usr/bin/env python3
"""Plot matched TOPAS/GPU minibeam depth dose across incident energies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENERGIES = (100, 200, 300, 400)


def load_topas(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size % 8 == 0:
        return np.fromfile(path, dtype="<f8")
    if size % 4 == 0:
        return np.fromfile(path, dtype="<f4").astype(np.float64)
    raise ValueError(f"{path}: unsupported TOPAS binary size")


def load_gpu(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(
        values, np.ones(width, dtype=np.float64) / width, mode="same"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--root", type=Path, default=Path("out/minibeam/energy_sweep")
    )
    parser.add_argument(
        "--topas-root", type=Path, default=Path("ct/minibeam/output")
    )
    parser.add_argument("--smoothing-mm", type=float, default=1.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "out/minibeam/energy_sweep/"
            "multienergy_depth_dose_smoothed.png"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help=(
            "Optional metric-trend figure. By default it is written next to "
            "--output as multienergy_metrics_trend.png."
        ),
    )
    args = parser.parse_args()
    metrics = json.loads(
        (args.root / "metrics.json").read_text(encoding="utf-8")
    )["energies"]
    smoothing_bins = max(1, int(round(args.smoothing_mm / 0.1)))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1

    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.0), constrained_layout=True
    )
    for axis, energy in zip(axes.flat, ENERGIES, strict=True):
        stem = f"minibeam_energy_e{energy}_{args.histories}"
        topas_total = load_topas(
            args.topas_root / f"{stem}_total.bin"
        )
        topas_primary = load_topas(
            args.topas_root / f"{stem}_primary_c12.bin"
        )
        gpu = load_gpu(
            args.root / "gpu" / f"e{energy}" / "species_dose.csv"
        )
        gpu_total = np.asarray(gpu["total_Gy"], dtype=np.float64)
        gpu_primary = np.asarray(gpu["primary_c12_Gy"], dtype=np.float64)
        if not (
            topas_total.size
            == topas_primary.size
            == gpu_total.size
            == gpu_primary.size
        ):
            raise ValueError(f"grid size mismatch at {energy} MeV/u")
        depth = (np.arange(topas_total.size) + 0.5) * 0.1
        normalization = float(np.max(smooth(topas_primary, smoothing_bins)))
        topas_total_s = smooth(topas_total, smoothing_bins) / normalization
        gpu_total_s = smooth(gpu_total, smoothing_bins) / normalization
        topas_primary_s = smooth(topas_primary, smoothing_bins) / normalization
        gpu_primary_s = smooth(gpu_primary, smoothing_bins) / normalization

        axis.plot(
            depth, topas_total_s, color="black", linewidth=1.35,
            label="TOPAS total",
        )
        axis.plot(
            depth, gpu_total_s, color="tab:blue", linewidth=1.15,
            label="GPU total",
        )
        axis.plot(
            depth, topas_primary_s, color="0.35", linewidth=1.0,
            linestyle="--", label="TOPAS primary C-12",
        )
        axis.plot(
            depth, gpu_primary_s, color="tab:red", linewidth=0.95,
            linestyle="--", label="GPU primary C-12",
        )
        item = metrics[str(energy)]
        axis.axvline(
            item["topas_R80_mm"], color="black", linewidth=0.9,
            linestyle=":",
        )
        axis.axvline(
            item["gpu_R80_mm"], color="tab:red", linewidth=0.9,
            linestyle=":",
        )
        axis.set_xlim(
            0.0,
            min(
                float(depth[-1]),
                1.12 * max(item["topas_R80_mm"], item["gpu_R80_mm"]),
            ),
        )
        axis.set_ylim(bottom=0.0)
        axis.set_title(
            f"{energy} MeV/u   "
            f"ΔR80 {item['delta_R80_mm']:+.2f} mm\n"
            f"integral {item['integral_difference_percent']:+.2f}%   "
            f"L1 {item['depth_normalized_L1_percent']:.2f}%"
        )
        axis.set_xlabel("Depth in water (mm)")
        axis.set_ylabel("Dose / TOPAS primary maximum")
        axis.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Minibeam multi-energy TOPAS vs GPU "
        f"({args.histories:,} histories/energy; "
        f"{args.smoothing_mm:g} mm display smoothing)",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)
    print(args.output)

    metrics_output = args.metrics_output
    if metrics_output is None:
        metrics_output = args.output.with_name("multienergy_metrics_trend.png")
    energy_values = np.asarray(ENERGIES, dtype=np.float64)
    items = [metrics[str(energy)] for energy in ENERGIES]
    delta_r80 = np.asarray(
        [item["delta_R80_mm"] for item in items], dtype=np.float64
    )
    primary_difference = np.asarray(
        [item["primary_integral_difference_percent"] for item in items],
        dtype=np.float64,
    )
    nonprimary_difference = np.asarray(
        [item["nonprimary_integral_difference_percent"] for item in items],
        dtype=np.float64,
    )
    total_difference = np.asarray(
        [item["integral_difference_percent"] for item in items],
        dtype=np.float64,
    )
    normalized_l1 = np.asarray(
        [item["depth_normalized_L1_percent"] for item in items],
        dtype=np.float64,
    )

    trend, trend_axes = plt.subplots(
        1, 3, figsize=(13.5, 4.0), constrained_layout=True
    )
    trend_axes[0].axhspan(-0.2, 0.2, color="tab:green", alpha=0.10)
    trend_axes[0].axhline(0.0, color="0.3", linewidth=0.8)
    trend_axes[0].plot(
        energy_values, delta_r80, marker="o", color="tab:blue"
    )
    trend_axes[0].set_ylabel("GPU - TOPAS R80 (mm)")
    trend_axes[0].set_title("Range agreement")

    trend_axes[1].axhspan(-3.0, 3.0, color="tab:green", alpha=0.10)
    trend_axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    trend_axes[1].plot(
        energy_values, total_difference, marker="o", label="Total"
    )
    trend_axes[1].plot(
        energy_values, primary_difference, marker="s", label="Primary C-12"
    )
    trend_axes[1].plot(
        energy_values, nonprimary_difference, marker="^",
        label="Non-primary",
    )
    trend_axes[1].set_ylabel("Integral difference (%)")
    trend_axes[1].set_title("Dose integral")
    trend_axes[1].legend(fontsize=8)

    trend_axes[2].axhspan(0.0, 3.0, color="tab:green", alpha=0.10)
    trend_axes[2].axhline(3.0, color="tab:green", linewidth=0.8, linestyle="--")
    trend_axes[2].plot(
        energy_values, normalized_l1, marker="o", color="tab:purple"
    )
    trend_axes[2].set_ylabel("Depth-normalized L1 (%)")
    trend_axes[2].set_ylim(bottom=0.0)
    trend_axes[2].set_title("Depth-dose shape")

    for axis in trend_axes:
        axis.set_xlabel("Incident energy (MeV/u)")
        axis.set_xticks(ENERGIES)
        axis.grid(alpha=0.2)
    trend.suptitle(
        "Minibeam TOPAS vs GPU: multi-energy accuracy "
        f"({args.histories:,} histories/energy)",
        fontsize=13,
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    trend.savefig(metrics_output, dpi=190)
    plt.close(trend)
    print(metrics_output)


if __name__ == "__main__":
    main()
