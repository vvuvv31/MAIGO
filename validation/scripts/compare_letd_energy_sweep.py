#!/usr/bin/env python3
"""Compare GPU and TOPAS primary/all-hadron LET_d from 100 to 400 MeV/u."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENERGIES = (100, 200, 300, 400)


def topas_values(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", delimiter=",")[:, 3]


def score(gpu: np.ndarray, topas: np.ndarray, mask: np.ndarray) -> dict:
    delta = gpu[mask] - topas[mask]
    relative = 100.0 * delta / np.maximum(np.abs(topas[mask]), 1.0e-12)
    return {
        "bins": int(mask.sum()),
        "mean_bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_relative_percent": float(np.mean(relative)),
        "median_absolute_relative_percent": float(np.median(np.abs(relative))),
        "p95_absolute_relative_percent": float(np.percentile(np.abs(relative), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-dir", type=Path, default=Path("out/letd_energy_sweep/gpu")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/letd_energy_sweep/comparison_g4_11_3_2"),
    )
    parser.add_argument("--gpu-label", default="GPU 100k")
    args = parser.parse_args()
    root = Path("out/letd_energy_sweep")
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    metrics = {
        "definition": "dose-weighted electronic LET, MeV/mm/(g/cm3)",
        "gpu_histories": 100_000,
        "topas_histories": 50_000,
        "mask": "matching TOPAS energy deposition >= 1% of its maximum",
        "energies": {},
    }
    for row, energy in enumerate(ENERGIES):
        gpu = np.genfromtxt(
            args.gpu_dir / f"e{energy}" / "gpu_letd.csv",
            delimiter=",",
            names=True,
        )
        topas_primary = topas_values(
            Path(f"validation/topas/output/e{energy}_letd_50k_primary_c12.csv")
        )
        topas_all = topas_values(
            Path(f"validation/topas/output/e{energy}_letd_50k_all_hadron.csv")
        )
        primary_energy = topas_values(
            Path(f"validation/topas/output/e{energy}_letd_50k_primary_energy.csv")
        )
        total_energy = topas_values(
            Path(f"validation/topas/output/e{energy}_letd_50k_total_energy.csv")
        )
        depth = gpu["depth_mm"]
        gpu_primary = gpu["primary_c12_letd_MeV_per_mm_per_g_cm3"]
        gpu_all = gpu["all_hadron_letd_MeV_per_mm_per_g_cm3"]
        primary_mask = (primary_energy >= 0.01 * primary_energy.max()) & (
            topas_primary > 0
        )
        all_mask = (total_energy >= 0.01 * total_energy.max()) & (topas_all > 0)
        metrics["energies"][str(energy)] = {
            "primary_c12": score(gpu_primary, topas_primary, primary_mask),
            "all_hadron": score(gpu_all, topas_all, all_mask),
        }
        for col, (gpu_curve, topas_curve, title) in enumerate(
            [
                (gpu_primary, topas_primary, "Primary C-12"),
                (gpu_all, topas_all, "All hadrons"),
            ]
        ):
            axis = axes[row, col]
            axis.plot(depth, topas_curve, label="TOPAS 50k")
            axis.plot(depth, gpu_curve, "--", label=args.gpu_label)
            axis.set_title(f"{energy} MeV/u — {title}")
            axis.set_xlabel("Depth in water [mm]")
            axis.set_ylabel(r"LET$_d$ [MeV/mm/(g/cm$^3$)]")
            axis.grid(alpha=0.25)
            axis.legend()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figure.suptitle("C-12 water LET$_d$ energy sweep: GPU vs TOPAS")
    figure.savefig(output / "gpu_vs_topas_letd_energy_sweep.png", dpi=180)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
