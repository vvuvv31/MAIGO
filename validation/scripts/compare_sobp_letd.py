#!/usr/bin/env python3
"""Compare GPU and TOPAS depth LETd for the 21-layer water SOBP."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/maigo-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def read_topas(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values.append(float(line.split(",")[3]))
    return np.asarray(values)


def read_gpu(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def metrics(
    reference: np.ndarray, test: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    delta = test[mask] - reference[mask]
    relative = delta / reference[mask]
    return {
        "bins": int(mask.sum()),
        "mean_bias": float(delta.mean()),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_relative_percent": float(100.0 * relative.mean()),
        "median_absolute_relative_percent": float(
            100.0 * np.median(np.abs(relative))
        ),
        "p95_absolute_relative_percent": float(
            100.0 * np.percentile(np.abs(relative), 95.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("out/letd_sobp/gpu_sobp_letd.csv"),
    )
    parser.add_argument(
        "--topas-primary",
        type=Path,
        default=Path("validation/topas/output/sobp_letd_100k_primary_c12.csv"),
    )
    parser.add_argument(
        "--topas-all",
        type=Path,
        default=Path("validation/topas/output/sobp_letd_100k_all_hadron.csv"),
    )
    parser.add_argument(
        "--topas-primary-energy",
        type=Path,
        default=Path(
            "validation/topas/output/"
            "sobp_letd_100k_primary_c12_energy_deposit.csv"
        ),
    )
    parser.add_argument(
        "--topas-total-energy",
        type=Path,
        default=Path(
            "validation/topas/output/sobp_letd_100k_total_energy_deposit.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/letd_sobp/comparison")
    )
    parser.add_argument("--sobp-start-mm", type=float, default=50.0)
    parser.add_argument("--sobp-end-mm", type=float, default=100.0)
    args = parser.parse_args()

    gpu = read_gpu(args.gpu)
    depth = gpu["depth_mm"]
    gpu_primary = gpu["primary_c12_letd_MeV_per_mm_per_g_cm3"]
    gpu_all = gpu["all_hadron_letd_MeV_per_mm_per_g_cm3"]
    topas_primary = read_topas(args.topas_primary)
    topas_all = read_topas(args.topas_all)
    primary_energy = read_topas(args.topas_primary_energy)
    total_energy = read_topas(args.topas_total_energy)
    arrays = (
        depth,
        gpu_primary,
        gpu_all,
        topas_primary,
        topas_all,
        primary_energy,
        total_energy,
    )
    if len({len(array) for array in arrays}) != 1:
        raise SystemExit("GPU and TOPAS depth grids differ")

    primary_valid = primary_energy > 0.01 * primary_energy.max()
    all_valid = total_energy > 0.01 * total_energy.max()
    sobp = (depth >= args.sobp_start_mm) & (depth <= args.sobp_end_mm)
    report = {
        "definition": "dose-weighted electronic LET, MeV/mm/(g/cm3)",
        "histories": {"gpu": 100_000, "topas": 100_000},
        "depth_bin_width_mm": float(np.median(np.diff(depth))),
        "global_above_1pct_energy": {
            "primary_c12": metrics(
                topas_primary, gpu_primary, primary_valid
            ),
            "all_hadron": metrics(topas_all, gpu_all, all_valid),
        },
        "sobp_50_100_mm": {
            "primary_c12": metrics(
                topas_primary, gpu_primary, primary_valid & sobp
            ),
            "all_hadron": metrics(topas_all, gpu_all, all_valid & sobp),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8.5), constrained_layout=True
    )
    for axis, reference, test, title in (
        (axes[0, 0], topas_primary, gpu_primary, "Primary C-12 LET$_d$"),
        (axes[0, 1], topas_all, gpu_all, "All-hadron LET$_d$"),
    ):
        axis.plot(depth, reference, label="TOPAS 100k", linewidth=1.8)
        axis.plot(depth, test, label="GPU 100k", linewidth=1.5, linestyle="--")
        axis.axvspan(
            args.sobp_start_mm,
            args.sobp_end_mm,
            color="#999999",
            alpha=0.12,
            label="SOBP 50–100 mm",
        )
        axis.set_xlim(0, 140)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Depth in water [mm]")
        axis.set_ylabel(r"LET$_d$ [MeV/mm/(g/cm$^3$)]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1, 0].plot(
        depth, gpu_primary - topas_primary, label="Primary C-12"
    )
    axes[1, 0].plot(depth, gpu_all - topas_all, label="All hadrons")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].axvspan(
        args.sobp_start_mm, args.sobp_end_mm, color="#999999", alpha=0.12
    )
    axes[1, 0].set_xlim(0, 140)
    axes[1, 0].set_xlabel("Depth in water [mm]")
    axes[1, 0].set_ylabel(r"GPU - TOPAS LET$_d$")
    axes[1, 0].set_title("Absolute difference")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        depth, total_energy / total_energy.max(), label="TOPAS total"
    )
    axes[1, 1].plot(
        depth,
        primary_energy / primary_energy.max(),
        label="TOPAS primary C-12",
    )
    axes[1, 1].axvspan(
        args.sobp_start_mm, args.sobp_end_mm, color="#999999", alpha=0.12
    )
    axes[1, 1].set_xlim(0, 140)
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_xlabel("Depth in water [mm]")
    axes[1, 1].set_ylabel("Relative energy deposition")
    axes[1, 1].set_title("SOBP dose support / metric mask")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle("21-layer C-12 SOBP LET$_d$: GPU vs TOPAS")
    figure.savefig(args.output_dir / "gpu_vs_topas_sobp_letd.png", dpi=190)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
