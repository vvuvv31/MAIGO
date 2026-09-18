#!/usr/bin/env python3
"""Compare matched-history GPU/TOPAS origin-resolved minibeam dose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


COMPONENTS = {
    "primary": ("component_primary_c.bin", "origin_primary.raw"),
    "secondary_carbon": ("component_secondary_c.bin", "origin_secondary_carbon.raw"),
    "boron": ("component_boron.bin", "origin_secondary_boron.raw"),
    "beryllium": ("component_beryllium.bin", "origin_secondary_beryllium.raw"),
    "lithium": ("component_lithium.bin", "origin_secondary_lithium.raw"),
    "helium": ("component_helium.bin", "origin_secondary_helium.raw"),
    "z1": ("component_z1.bin", "origin_secondary_proton.raw"),
    "other_charged": ("component_other_charged.bin", "origin_secondary_other_charged.raw"),
}


def read_grid(path: Path, dtype: str, bins: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=dtype).astype(np.float64)
    if flat.size != bins * bins:
        raise ValueError(f"Expected {bins*bins} values in {path}, got {flat.size}")
    return flat.reshape(bins, bins)


def l1(gpu: np.ndarray, topas: np.ndarray) -> float | None:
    denominator = float(np.abs(topas).sum())
    return float(np.abs(gpu - topas).sum() / denominator) if denominator > 0 else None


def feature_curves(dose: np.ndarray, x_mm: np.ndarray, pitch_mm: float,
                   slab_bins: int) -> tuple[np.ndarray, np.ndarray]:
    smooth = uniform_filter1d(dose, size=slab_bins, axis=0, mode="nearest")
    peaks = []
    valleys = []
    rows = np.arange(dose.shape[0])
    for expected in np.arange(-4, 5) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= 0.7)
        chosen = indices[np.argmax(smooth[:, indices], axis=1)]
        peaks.append(np.mean(np.stack([
            smooth[rows, np.clip(chosen + shift, 0, dose.shape[1] - 1)]
            for shift in (-1, 0, 1)
        ]), axis=0))
    for expected in (np.arange(-4, 4) + 0.5) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= 0.7)
        chosen = indices[np.argmin(smooth[:, indices], axis=1)]
        valleys.append(np.mean(np.stack([
            smooth[rows, np.clip(chosen + shift, 0, dose.shape[1] - 1)]
            for shift in (-1, 0, 1)
        ]), axis=0))
    return np.median(np.stack(peaks), axis=0), np.median(np.stack(valleys), axis=0)


def safe_ratio(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topas-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-total", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--depth-spacing-mm", type=float, default=0.25)
    parser.add_argument("--lateral-spacing-mm", type=float, default=0.1)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topas_total = read_grid(args.topas_dir / "dose.bin", "<f8", args.bins)
    gpu_total = read_grid(args.gpu_total, "<f4", args.bins)
    topas = {}
    gpu = {}
    for name, (topas_name, gpu_name) in COMPONENTS.items():
        topas[name] = read_grid(args.topas_dir / topas_name, "<f8", args.bins)
        gpu[name] = read_grid(args.gpu_dir / gpu_name, "<f4", args.bins)
    topas_neutral = read_grid(
        args.topas_dir / "component_neutral_origin.bin", "<f8", args.bins
    )
    topas_unclassified = read_grid(
        args.topas_dir / "component_unclassified.bin", "<f8", args.bins
    )
    topas_charged = sum(topas.values())
    gpu_charged = sum(gpu.values())
    topas_fragments = topas_charged - topas["primary"]
    gpu_fragments = gpu_charged - gpu["primary"]

    depth = (np.arange(args.bins) + 0.5) * args.depth_spacing_mm
    x_mm = ((np.arange(args.bins) + 0.5) * args.lateral_spacing_mm
            - 0.5 * args.bins * args.lateral_spacing_mm)
    search = depth >= 20.0
    bragg_index = np.flatnonzero(search)[np.argmax(topas_total.sum(axis=1)[search])]
    bragg_depth = float(depth[bragg_index])
    regions = {
        "entrance_0_5mm": depth < 5.0,
        "pre_bragg": (depth >= 0.45 * bragg_depth) & (depth < 0.55 * bragg_depth),
        "bragg_pm2mm": np.abs(depth - bragg_depth) <= 2.0,
        "distal_20mm_after_bragg": depth >= bragg_depth + 20.0,
    }

    metrics: dict = {
        "comparison": "absolute Gy, 1,024,000 histories per engine, no fitted scale",
        "bragg_depth_topas_total_mm": bragg_depth,
        "closure": {
            "topas_charged_plus_neutral_plus_unclassified_over_total": float(
                (topas_charged + topas_neutral + topas_unclassified).sum()
                / topas_total.sum()
            ),
            "topas_neutral_fraction": float(topas_neutral.sum() / topas_total.sum()),
            "topas_unclassified_fraction": float(topas_unclassified.sum() / topas_total.sum()),
            "gpu_charged_categories_over_total": float(gpu_charged.sum() / gpu_total.sum()),
        },
        "components": {},
        "aggregates": {},
        "selected_depth_features": [],
    }

    for name in COMPONENTS:
        topas_sum = float(topas[name].sum())
        gpu_sum = float(gpu[name].sum())
        entry = {
            "topas_sum_Gy": topas_sum,
            "gpu_sum_Gy": gpu_sum,
            "gpu_over_topas_sum": safe_ratio(gpu_sum, topas_sum),
            "topas_fraction_of_total": topas_sum / float(topas_total.sum()),
            "gpu_fraction_of_total": gpu_sum / float(gpu_total.sum()),
            "two_dimensional_l1_over_topas": l1(gpu[name], topas[name]),
            "idd_l1_over_topas": l1(gpu[name].sum(axis=1), topas[name].sum(axis=1)),
            "lateral_l1_over_topas": l1(gpu[name].sum(axis=0), topas[name].sum(axis=0)),
            "regional_gpu_over_topas": {},
        }
        for region_name, mask in regions.items():
            entry["regional_gpu_over_topas"][region_name] = safe_ratio(
                float(gpu[name][mask].sum()), float(topas[name][mask].sum())
            )
        metrics["components"][name] = entry

    for name, topas_grid, gpu_grid in (
        ("total", topas_total, gpu_total),
        ("primary", topas["primary"], gpu["primary"]),
        ("charged_fragments", topas_fragments, gpu_fragments),
        ("charged_all", topas_charged, gpu_charged),
    ):
        metrics["aggregates"][name] = {
            "gpu_over_topas_sum": safe_ratio(float(gpu_grid.sum()), float(topas_grid.sum())),
            "two_dimensional_l1_over_topas": l1(gpu_grid, topas_grid),
            "idd_l1_over_topas": l1(gpu_grid.sum(axis=1), topas_grid.sum(axis=1)),
            "lateral_l1_over_topas": l1(gpu_grid.sum(axis=0), topas_grid.sum(axis=0)),
            "regional_gpu_over_topas": {
                region_name: safe_ratio(float(gpu_grid[mask].sum()), float(topas_grid[mask].sum()))
                for region_name, mask in regions.items()
            },
        }

    slab_bins = max(1, int(round(1.0 / args.depth_spacing_mm)))
    curves = {}
    for name, tg, gg in (
        ("total", topas_total, gpu_total),
        ("primary", topas["primary"], gpu["primary"]),
        ("charged_fragments", topas_fragments, gpu_fragments),
    ):
        tp, tv = feature_curves(tg, x_mm, args.pitch_mm, slab_bins)
        gp, gv = feature_curves(gg, x_mm, args.pitch_mm, slab_bins)
        curves[name] = (tp, tv, gp, gv)
    for requested in (0.5, 0.5 * bragg_depth, bragg_depth, bragg_depth + 10.0):
        index = int(np.argmin(np.abs(depth - requested)))
        row = {"depth_mm": float(depth[index]), "components": {}}
        for name, (tp, tv, gp, gv) in curves.items():
            row["components"][name] = {
                "peak_gpu_over_topas": safe_ratio(float(gp[index]), float(tp[index])),
                "valley_gpu_over_topas": safe_ratio(float(gv[index]), float(tv[index])),
                "pvdr_gpu_over_topas": safe_ratio(
                    safe_ratio(float(gp[index]), float(gv[index])) or 0.0,
                    safe_ratio(float(tp[index]), float(tv[index])) or 0.0,
                ),
            }
        metrics["selected_depth_features"].append(row)

    (args.output_dir / "component_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    shown = ["primary", "secondary_carbon", "boron", "beryllium", "lithium", "helium", "z1"]
    for name in shown:
        axes[0, 0].plot(depth, topas[name].sum(axis=1), label=name)
        axes[0, 1].plot(depth, gpu[name].sum(axis=1), label=name)
    axes[0, 0].plot(depth, topas_neutral.sum(axis=1), label="neutral", ls="--", color="black")
    for axis, title in zip(axes[0], ("TOPAS origin IDD", "GPU origin IDD")):
        axis.set_yscale("log")
        axis.set_xlim(0, min(220, depth[-1]))
        axis.set_ylim(bottom=1e-9)
        axis.set(xlabel="water depth [mm]", ylabel="laterally integrated dose [Gy]", title=title)
        axis.legend(fontsize=7, ncol=2)
    for name, tg, gg in (
        ("total", topas_total, gpu_total),
        ("primary", topas["primary"], gpu["primary"]),
        ("charged fragments", topas_fragments, gpu_fragments),
    ):
        ratio = np.divide(gg.sum(axis=1), tg.sum(axis=1), out=np.full(args.bins, np.nan),
                          where=tg.sum(axis=1) > 0)
        axes[1, 0].plot(depth, ratio, label=name)
    axes[1, 0].axhline(1.0, color="black", lw=0.8)
    axes[1, 0].set(xlim=(0, min(220, depth[-1])), ylim=(0.5, 1.6), xlabel="water depth [mm]",
                   ylabel="GPU / TOPAS IDD", title="Aggregate depth-dose ratios")
    axes[1, 0].legend()
    labels = list(COMPONENTS)
    ratios = [metrics["components"][name]["gpu_over_topas_sum"] for name in labels]
    axes[1, 1].bar(np.arange(len(labels)), ratios)
    axes[1, 1].axhline(1.0, color="black", lw=0.8)
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
    axes[1, 1].set(ylabel="GPU / TOPAS integrated dose", title="Component normalization")
    fig.savefig(args.output_dir / "component_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
