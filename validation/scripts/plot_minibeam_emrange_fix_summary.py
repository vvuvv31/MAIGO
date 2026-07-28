#!/usr/bin/env python3
"""Plot the high-energy EM-table fix and the matched 400 MeV/u dose result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA_PDG = 1_000_020_040


def load_primary_alpha_energy(path: Path) -> np.ndarray:
    phase = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if phase.shape[1] != 14:
        raise ValueError(f"{path}: expected 14 columns, found {phase.shape[1]}")
    pdg = phase[:, 7].astype(np.int64)
    track = phase[:, 12].astype(np.int64)
    parent = phase[:, 13].astype(np.int64)
    return phase[(pdg == ALPHA_PDG) & (track == 1) & (parent == 0), 5]


def load_topas_binary(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size % 8 == 0:
        return np.fromfile(path, dtype="<f8")
    if size % 4 == 0:
        return np.fromfile(path, dtype="<f4").astype(np.float64)
    raise ValueError(f"{path}: unsupported binary size")


def load_gpu_dose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return (
        np.asarray(data["depth_mm"], dtype=np.float64),
        np.asarray(data["dose_Gy"], dtype=np.float64),
    )


def load_gpu_primary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return (
        np.asarray(data["depth_mm"], dtype=np.float64),
        np.asarray(data["primary_c12_Gy"], dtype=np.float64),
    )


def smooth(values: np.ndarray, bins: int = 11) -> np.ndarray:
    return np.convolve(
        values, np.ones(bins, dtype=np.float64) / bins, mode="same"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-alpha",
        type=Path,
        default=Path(
            "ct/minibeam/output/"
            "secondary_ion_copper_he4_e300_t1_10000_exit.phsp"
        ),
    )
    parser.add_argument(
        "--fixed-alpha",
        type=Path,
        default=Path(
            "ct/minibeam/output/"
            "secondary_ion_copper_he4_e300_t1_10000_emmax6000_exit.phsp"
        ),
    )
    parser.add_argument(
        "--topas-root", type=Path, default=Path("ct/minibeam/output")
    )
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--gpu-root",
        type=Path,
        default=Path("out/minibeam/energy_sweep/gpu/e400"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("out/minibeam/energy_sweep/metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "out/minibeam/emrange_fix_summary/"
            "emrange_fix_and_e400_dose.png"
        ),
    )
    args = parser.parse_args()

    old_alpha = load_primary_alpha_energy(args.old_alpha)
    fixed_alpha = load_primary_alpha_energy(args.fixed_alpha)
    gpu_depth, gpu_total = load_gpu_dose(
        args.gpu_root / "depth_dose.csv"
    )
    gpu_primary_depth, gpu_primary = load_gpu_primary(
        args.gpu_root / "species_dose.csv"
    )
    stem = f"minibeam_energy_e400_{args.histories}"
    topas_total = load_topas_binary(
        args.topas_root / f"{stem}_total.bin"
    )
    topas_primary = load_topas_binary(
        args.topas_root / f"{stem}_primary_c12.bin"
    )
    topas_depth = (
        np.arange(topas_total.size, dtype=np.float64) + 0.5
    ) * 0.1
    if (
        gpu_total.size != topas_total.size
        or gpu_primary.size != topas_primary.size
        or not np.allclose(gpu_depth, topas_depth)
        or not np.allclose(gpu_primary_depth, topas_depth)
    ):
        raise ValueError("GPU and TOPAS depth grids differ")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))[
        "energies"
    ]["400"]
    normalization = float(np.max(topas_primary))
    topas_total_smooth = smooth(topas_total)
    gpu_total_smooth = smooth(gpu_total)
    valid = topas_total_smooth > 0.01 * float(np.max(topas_total_smooth))
    difference = np.full(topas_total.shape, np.nan, dtype=np.float64)
    difference[valid] = 100.0 * (
        gpu_total_smooth[valid] / topas_total_smooth[valid] - 1.0
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(13.2, 8.5), constrained_layout=True
    )
    bins = np.linspace(1178.0, 1195.0, 100)
    axes[0, 0].hist(
        old_alpha,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="tab:orange",
        label="TOPAS default: EMRangeMax 600 MeV",
    )
    axes[0, 0].hist(
        fixed_alpha,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="tab:blue",
        label="TOPAS fixed: EMRangeMax 6 GeV",
    )
    axes[0, 0].axvline(
        1191.415,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="GPU mean-loss prediction",
    )
    axes[0, 0].set_title(
        "He-4, 300 MeV/u after 1 mm Copper"
    )
    axes[0, 0].set_xlabel("Exit kinetic energy (MeV)")
    axes[0, 0].set_ylabel("Probability density")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        topas_depth,
        topas_total / normalization,
        color="black",
        linewidth=0.45,
        alpha=0.18,
    )
    axes[0, 1].plot(
        gpu_depth,
        gpu_total / normalization,
        color="tab:blue",
        linewidth=0.45,
        alpha=0.18,
    )
    axes[0, 1].plot(
        topas_depth,
        topas_total_smooth / normalization,
        color="black",
        linewidth=1.25,
        label="TOPAS total (1.1 mm smoothing)",
    )
    axes[0, 1].plot(
        gpu_depth,
        gpu_total_smooth / normalization,
        color="tab:blue",
        linewidth=1.1,
        label="GPU total (1.1 mm smoothing)",
    )
    axes[0, 1].set_title(
        f"400 MeV/u total depth dose, {args.histories:,} histories"
    )
    axes[0, 1].set_xlim(0.0, 310.0)
    axes[0, 1].set_ylim(bottom=0.0)
    axes[0, 1].set_xlabel("Depth in water (mm)")
    axes[0, 1].set_ylabel("Dose / TOPAS primary maximum")
    axes[0, 1].legend(fontsize=8)

    topas_primary_smooth = smooth(topas_primary)
    gpu_primary_smooth = smooth(gpu_primary)
    axes[1, 0].plot(
        topas_depth,
        topas_primary / normalization,
        color="black",
        linewidth=0.45,
        alpha=0.18,
    )
    axes[1, 0].plot(
        gpu_primary_depth,
        gpu_primary / normalization,
        color="tab:red",
        linewidth=0.45,
        alpha=0.18,
    )
    axes[1, 0].plot(
        topas_depth,
        topas_primary_smooth / normalization,
        color="black",
        linewidth=1.25,
        label="TOPAS primary C-12 (1.1 mm smoothing)",
    )
    axes[1, 0].plot(
        gpu_primary_depth,
        gpu_primary_smooth / normalization,
        color="tab:red",
        linewidth=1.1,
        label="GPU primary C-12 (1.1 mm smoothing)",
    )
    axes[1, 0].axvline(
        metrics["topas_R80_mm"],
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 0].axvline(
        metrics["gpu_R80_mm"],
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 0].set_title(
        f"Primary range: ΔR80 = {metrics['delta_R80_mm']:+.3f} mm"
    )
    axes[1, 0].set_xlim(0.0, 310.0)
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 0].set_xlabel("Depth in water (mm)")
    axes[1, 0].set_ylabel("Dose / TOPAS primary maximum")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].axhspan(
        -3.0, 3.0, color="tab:green", alpha=0.12, label="±3%"
    )
    axes[1, 1].plot(
        topas_depth,
        difference,
        color="tab:purple",
        linewidth=1.0,
        label="GPU − TOPAS (1.1 mm smoothing)",
    )
    axes[1, 1].set_title(
        "Total depth-dose relative difference\n"
        f"integral {metrics['integral_difference_percent']:+.2f}%, "
        f"L1 {metrics['depth_normalized_L1_percent']:.2f}%"
    )
    axes[1, 1].set_xlim(0.0, 310.0)
    axes[1, 1].set_ylim(-30.0, 30.0)
    axes[1, 1].set_xlabel("Depth in water (mm)")
    axes[1, 1].set_ylabel("Relative difference (%)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Minibeam high-energy EM-table fix: independent physics check",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
