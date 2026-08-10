#!/usr/bin/env python3
"""Compare GPU LET_d against TOPAS HadronLET in the 200 MeV/u water case."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/maigo-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def read_topas(path: Path, value_column: int = 3) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("#") or not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            rows.append((int(fields[2]), float(fields[value_column])))
    if not rows:
        raise ValueError(f"No TOPAS data rows in {path}")
    result = np.zeros(max(index for index, _ in rows) + 1)
    for index, value in rows:
        result[index] = value
    return result


def read_gpu(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def metrics(reference: np.ndarray, test: np.ndarray, mask: np.ndarray) -> dict[str, float]:
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=Path, default=Path("out/letd_water/benchmark/gpu_letd_on_2.csv")
    )
    parser.add_argument(
        "--topas-primary",
        type=Path,
        default=Path("validation/topas/output/topas_primary_c12_letd.csv"),
    )
    parser.add_argument(
        "--topas-all",
        type=Path,
        default=Path("validation/topas/output/topas_all_hadron_letd.csv"),
    )
    parser.add_argument(
        "--topas-primary-energy",
        type=Path,
        default=Path(
            "validation/topas/output/topas_primary_c12_energy_deposit.csv"
        ),
    )
    parser.add_argument(
        "--topas-total-energy",
        type=Path,
        default=Path("validation/topas/output/topas_energy_deposit.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/letd_water/comparison")
    )
    args = parser.parse_args()

    gpu = read_gpu(args.gpu)
    depth = gpu["depth_mm"]
    topas_primary = read_topas(args.topas_primary)
    topas_all = read_topas(args.topas_all)
    primary_energy = read_topas(args.topas_primary_energy)
    total_energy = read_topas(args.topas_total_energy)
    if not (
        len(depth)
        == len(topas_primary)
        == len(topas_all)
        == len(primary_energy)
        == len(total_energy)
    ):
        raise ValueError("GPU and TOPAS depth grids differ")

    gpu_primary = gpu["primary_c12_letd_MeV_per_mm_per_g_cm3"]
    gpu_all = gpu["all_hadron_letd_MeV_per_mm_per_g_cm3"]
    primary_mask = primary_energy > 0.01 * primary_energy.max()
    all_mask = total_energy > 0.01 * total_energy.max()
    report = {
        "definition": "dose-weighted electronic LET, MeV/mm/(g/cm3)",
        "mask": "corresponding TOPAS energy deposition > 1% of maximum",
        "primary_c12": metrics(topas_primary, gpu_primary, primary_mask),
        "all_hadron": metrics(topas_all, gpu_all, all_mask),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, reference, test, title in (
        (axes[0, 0], topas_primary, gpu_primary, "Primary C-12 LET$_d$"),
        (axes[0, 1], topas_all, gpu_all, "All-hadron LET$_d$"),
    ):
        axis.plot(depth, reference, label="TOPAS HadronLET", linewidth=1.8)
        axis.plot(depth, test, label="GPU", linewidth=1.4)
        axis.set_xlim(0, 120)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Depth in water (mm)")
        axis.set_ylabel(r"LET$_d$ (MeV/mm/(g/cm$^3$))")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1, 0].plot(depth, gpu_primary - topas_primary, label="Primary C-12")
    axes[1, 0].plot(depth, gpu_all - topas_all, label="All hadrons")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xlim(0, 120)
    axes[1, 0].set_xlabel("Depth in water (mm)")
    axes[1, 0].set_ylabel(r"GPU - TOPAS LET$_d$")
    axes[1, 0].set_title("Absolute difference")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(depth, total_energy / total_energy.max(), label="TOPAS total")
    axes[1, 1].plot(
        depth, primary_energy / primary_energy.max(), label="TOPAS primary C-12"
    )
    axes[1, 1].set_xlim(0, 120)
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_xlabel("Depth in water (mm)")
    axes[1, 1].set_ylabel("Relative energy deposition")
    axes[1, 1].set_title("Metric masks / Bragg region")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    figure.savefig(args.output_dir / "gpu_vs_topas_water_letd.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
