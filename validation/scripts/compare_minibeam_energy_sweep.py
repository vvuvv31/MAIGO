#!/usr/bin/env python3
"""Compare matched TOPAS/GPU 1D minibeam energy-sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ENERGIES_MEVU = (100, 200, 300, 400)


def load_gpu(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def load_topas(path: Path, depth_bin_mm: float) -> tuple[np.ndarray, np.ndarray]:
    size = path.stat().st_size
    if size % 8 == 0:
        values = np.fromfile(path, dtype="<f8")
    elif size % 4 == 0:
        values = np.fromfile(path, dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"unsupported TOPAS binary size: {path}")
    depth = (np.arange(values.size, dtype=np.float64) + 0.5) * depth_bin_mm
    return depth, values


def distal_peak_metrics(
    depth: np.ndarray, dose: np.ndarray, fraction: float = 0.8
) -> tuple[float, float]:
    """Return the deepest primary-C12 Bragg peak and its distal crossing.

    Copper traversal produces a broad primary-energy spectrum.  At 300--400
    MeV/u its entrance dose can exceed the distal full-energy Bragg peak, so a
    global argmax is not a valid range definition.  Smooth over 1.1 mm, find
    the deepest region retaining at least 20% of the global signal, then seek
    its peak in the preceding 20 mm.
    """
    spacing = float(np.median(np.diff(depth)))
    window_bins = max(3, int(round(1.0 / spacing)))
    if window_bins % 2 == 0:
        window_bins += 1
    smooth = np.convolve(
        dose, np.ones(window_bins, dtype=np.float64) / window_bins, mode="same"
    )
    distal_end = int(np.flatnonzero(smooth >= 0.2 * np.max(smooth))[-1])
    search_bins = max(1, int(round(20.0 / spacing)))
    begin = max(0, distal_end - search_bins)
    peak = begin + int(np.argmax(smooth[begin : distal_end + 1]))
    target = fraction * float(smooth[peak])
    for index in range(peak + 1, smooth.size):
        if smooth[index] <= target < smooth[index - 1]:
            denominator = float(smooth[index] - smooth[index - 1])
            weight = (
                0.0 if denominator == 0.0
                else (target - float(smooth[index - 1])) / denominator
            )
            crossing = float(
                depth[index - 1]
                + weight * (depth[index] - depth[index - 1])
            )
            return float(depth[peak]), crossing
    return float(depth[peak]), math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--root", type=Path, default=Path("out/minibeam/energy_sweep")
    )
    parser.add_argument("--depth-bin-mm", type=float, default=0.1)
    parser.add_argument(
        "--energies",
        type=int,
        nargs="+",
        choices=ENERGIES_MEVU,
        default=list(ENERGIES_MEVU),
        help="Subset of prepared energies to compare.",
    )
    args = parser.parse_args()
    report: dict[str, object] = {
        "histories_per_energy": args.histories,
        "depth_bin_mm": args.depth_bin_mm,
        "energies": {},
    }
    rows: list[dict[str, float]] = []
    column_count = min(2, len(args.energies))
    row_count = math.ceil(len(args.energies) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 4 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    for energy in args.energies:
        gpu_path = args.root / "gpu" / f"e{energy}" / "depth_dose.csv"
        stem = f"minibeam_energy_e{energy}_{args.histories}"
        topas_path = (
            Path("ct/minibeam/output") / f"{stem}_total.bin"
        )
        gpu_depth, gpu = load_gpu(gpu_path)
        topas_depth, topas = load_topas(topas_path, args.depth_bin_mm)
        gpu_primary_depth, gpu_primary = load_gpu_primary(
            args.root / "gpu" / f"e{energy}" / "species_dose.csv"
        )
        _, topas_primary = load_topas(
            Path("ct/minibeam/output") / f"{stem}_primary_c12.bin",
            args.depth_bin_mm,
        )
        if gpu.size != topas.size or not np.allclose(gpu_depth, topas_depth):
            raise ValueError(f"depth grid mismatch at {energy} MeV/u")
        if (
            gpu_primary.size != topas_primary.size
            or not np.allclose(gpu_primary_depth, topas_depth)
        ):
            raise ValueError(f"primary C-12 depth grid mismatch at {energy} MeV/u")
        # At high energy, collimator fragments can make the entrance region the
        # global maximum of total dose.  Primary-C12 R80 remains an unambiguous
        # range metric, while total dose is retained for entrance/integral tests.
        topas_peak, topas_r80 = distal_peak_metrics(
            topas_depth, topas_primary
        )
        gpu_peak, gpu_r80 = distal_peak_metrics(gpu_depth, gpu_primary)
        entrance_1 = topas_depth < 1.0
        entrance_5 = topas_depth < 5.0
        topas_nonprimary = np.maximum(0.0, topas - topas_primary)
        gpu_nonprimary = np.maximum(0.0, gpu - gpu_primary)
        item = {
            "energy_MeVu": float(energy),
            "topas_R80_mm": topas_r80,
            "gpu_R80_mm": gpu_r80,
            "delta_R80_mm": gpu_r80 - topas_r80,
            "topas_primary_peak_mm": topas_peak,
            "gpu_primary_peak_mm": gpu_peak,
            "primary_integral_difference_percent": 100.0 * (
                float(np.sum(gpu_primary)) / float(np.sum(topas_primary)) - 1.0
            ),
            "nonprimary_integral_difference_percent": 100.0 * (
                float(np.sum(gpu_nonprimary))
                / float(np.sum(topas_nonprimary)) - 1.0
            ),
            "integral_difference_percent": 100.0 * (
                float(np.sum(gpu)) / float(np.sum(topas)) - 1.0
            ),
            "entrance_0_1mm_difference_percent": 100.0 * (
                float(np.sum(gpu[entrance_1]))
                / float(np.sum(topas[entrance_1])) - 1.0
            ),
            "entrance_0_5mm_difference_percent": 100.0 * (
                float(np.sum(gpu[entrance_5]))
                / float(np.sum(topas[entrance_5])) - 1.0
            ),
            "depth_normalized_L1_percent": 100.0
            * float(np.sum(np.abs(gpu - topas)))
            / float(np.sum(topas)),
        }
        report["energies"][str(energy)] = item
        rows.append(item)
        axis = axes.flat[len(rows) - 1]
        normalization = float(np.max(topas_primary))
        axis.plot(
            topas_depth, topas_primary / normalization,
            color="black", linewidth=1.4, label="TOPAS primary C-12",
        )
        axis.plot(
            gpu_depth, gpu_primary / normalization,
            color="tab:red", linewidth=1.1, label="GPU primary C-12",
        )
        axis.plot(
            topas_depth, topas / normalization,
            color="0.45", linewidth=0.8, alpha=0.65, label="TOPAS total",
        )
        axis.plot(
            gpu_depth, gpu / normalization,
            color="tab:blue", linewidth=0.8, alpha=0.65, label="GPU total",
        )
        axis.axvline(topas_r80, color="black", linestyle=":", linewidth=0.9)
        axis.axvline(gpu_r80, color="tab:red", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"{energy} MeV/u  ΔR80={gpu_r80 - topas_r80:+.2f} mm"
        )
        axis.set_xlim(0.0, min(400.0, 1.12 * max(topas_r80, gpu_r80)))
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Depth in water (mm)")
        axis.set_ylabel("Dose / TOPAS primary maximum")
    args.root.mkdir(parents=True, exist_ok=True)
    json_path = args.root / "metrics.json"
    csv_path = args.root / "metrics.csv"
    plot_path = args.root / "depth_dose_profiles.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for axis in axes.flat[len(rows):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))
    print(json_path)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
