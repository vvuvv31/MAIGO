#!/usr/bin/env python3
"""Plot central-axis depth LETd for the 200 MeV/u water validation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/maigo-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def read_gpu(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def read_topas(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values.append(float(line.split(",")[3]))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("out/letd_water/fp64_moments_500k/gpu_letd_on_0.csv"),
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
        "--output",
        type=Path,
        default=Path(
            "out/letd_water/central_axis_depth_letd_gpu500k_topas100k.png"
        ),
    )
    parser.add_argument("--maximum-depth-mm", type=float, default=140.0)
    args = parser.parse_args()

    gpu = read_gpu(args.gpu)
    depth = gpu["depth_mm"]
    gpu_primary = gpu["primary_c12_letd_MeV_per_mm_per_g_cm3"]
    gpu_all = gpu["all_hadron_letd_MeV_per_mm_per_g_cm3"]
    topas_primary = read_topas(args.topas_primary)
    topas_all = read_topas(args.topas_all)
    if not (len(depth) == len(topas_primary) == len(topas_all)):
        raise SystemExit("GPU and TOPAS depth grids differ")

    curves = (
        ("Primary C-12", topas_primary, gpu_primary),
        ("All hadrons", topas_all, gpu_all),
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(10.5, 8.5), sharex=True, constrained_layout=True
    )
    for axis, (title, topas, gpu_curve) in zip(axes, curves):
        topas_peak = int(np.argmax(topas))
        gpu_peak = int(np.argmax(gpu_curve))
        axis.plot(
            depth,
            topas,
            color="#1f77b4",
            linewidth=2.1,
            label="TOPAS 100k",
        )
        axis.plot(
            depth,
            gpu_curve,
            color="#e66101",
            linewidth=1.7,
            linestyle="--",
            label="GPU 500k",
        )
        axis.axvline(
            depth[topas_peak],
            color="#1f77b4",
            linewidth=0.9,
            alpha=0.45,
        )
        axis.scatter(
            [depth[topas_peak], depth[gpu_peak]],
            [topas[topas_peak], gpu_curve[gpu_peak]],
            color=["#1f77b4", "#e66101"],
            s=28,
            zorder=4,
        )
        axis.text(
            0.985,
            0.94,
            (
                f"Peak depth: TOPAS {depth[topas_peak]:.2f} mm, "
                f"GPU {depth[gpu_peak]:.2f} mm"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        axis.set_xlim(0.0, args.maximum_depth_mm)
        axis.set_ylim(bottom=0.0)
        axis.set_ylabel(r"LET$_d$ [MeV/mm/(g/cm$^3$)]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Depth along central beam axis in water [mm]")
    figure.suptitle(
        r"200 MeV/u C-12 central-axis depth LET$_d$ in water",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
