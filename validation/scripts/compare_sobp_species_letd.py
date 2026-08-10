#!/usr/bin/env python3
"""Diagnose SOBP LET_d by charged-particle atomic-number group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPECIES = {
    "primary_c12": Path("validation/topas/output/sobp_letd_100k_primary_c12.csv"),
    "secondary_carbon": Path(
        "validation/topas/output/sobp_letd_species_100k_secondary_carbon.csv"
    ),
    "boron": Path("validation/topas/output/sobp_letd_species_100k_boron.csv"),
    "beryllium": Path(
        "validation/topas/output/sobp_letd_species_100k_beryllium.csv"
    ),
    "lithium": Path("validation/topas/output/sobp_letd_species_100k_lithium.csv"),
    "helium": Path("validation/topas/output/sobp_letd_species_100k_helium.csv"),
    "hydrogen": Path(
        "validation/topas/output/sobp_letd_species_100k_hydrogen.csv"
    ),
}


def score(gpu: np.ndarray, topas: np.ndarray, mask: np.ndarray) -> dict:
    delta = gpu[mask] - topas[mask]
    relative = 100.0 * delta / np.maximum(np.abs(topas[mask]), 1.0e-12)
    return {
        "bins": int(mask.sum()),
        "mean_bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "mean_relative_percent": float(np.mean(relative)),
        "median_absolute_relative_percent": float(np.median(np.abs(relative))),
        "p95_absolute_relative_percent": float(np.percentile(np.abs(relative), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("out/letd_sobp/gpu_sobp_species_letd.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/letd_sobp/species_comparison_g4_11_3_2"),
    )
    parser.add_argument("--gpu-label", default="GPU 100k")
    args = parser.parse_args()
    gpu = np.genfromtxt(
        args.gpu,
        delimiter=",",
        names=True,
    )
    depth = gpu["depth_mm"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    figure, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    for axis, (name, topas_path) in zip(axes.flat, SPECIES.items()):
        topas = np.loadtxt(topas_path, comments="#", delimiter=",")[:, 3]
        gpu_let = gpu[f"{name}_letd_MeV_per_mm_per_g_cm3"]
        denominator = gpu[f"{name}_denominator_MeV"]
        support = (denominator >= 0.01 * denominator.max()) & (topas > 0.0)
        sobp = support & (depth >= 50.0) & (depth <= 100.0)
        metrics[name] = {
            "global_species_support": score(gpu_let, topas, support),
            "sobp_50_100_mm": score(gpu_let, topas, sobp),
        }
        axis.plot(depth, topas, label="TOPAS 100k")
        axis.plot(depth, gpu_let, "--", label=args.gpu_label)
        axis.axvspan(50, 100, color="0.85", alpha=0.5)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Depth in water [mm]")
        axis.set_ylabel(r"LET$_d$ [MeV/mm/(g/cm$^3$)]")
        axis.grid(alpha=0.25)
        axis.legend()
    axes.flat[-1].axis("off")
    figure.suptitle("21-layer SOBP species-resolved LET$_d$: GPU vs TOPAS")
    figure.savefig(output / "gpu_vs_topas_sobp_species_letd.png", dpi=180)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
