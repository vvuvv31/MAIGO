#!/usr/bin/env python3
"""Compare GPU vs TOPAS density-slab IDD (absolute MeV/primary)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data["energy_deposition_MeV_per_primary"], dtype=float)),
    )


def r80(depth: np.ndarray, dose: np.ndarray) -> float:
    n = dose / np.max(dose)
    peak = int(np.argmax(n))
    for i in range(peak + 1, len(depth)):
        if n[i] <= 0.8 < n[i - 1]:
            return float(
                depth[i - 1]
                + (0.8 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--topas", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()

    gz, gd = load_idd(args.gpu)
    tz, td = load_idd(args.topas)
    n = min(len(gz), len(tz))
    gz, gd, tz, td = gz[:n], gd[:n], tz[:n], td[:n]
    diff = gd - td
    ref_max = float(np.max(td))
    dense = (tz >= 50.0) & (tz < 70.0)
    report = {
        "gpu": str(args.gpu),
        "topas": str(args.topas),
        "integral_gpu": float(np.trapz(gd, gz)),
        "integral_topas": float(np.trapz(td, tz)),
        "integral_rel_percent": 100.0
        * (float(np.trapz(gd, gz)) - float(np.trapz(td, tz)))
        / float(np.trapz(td, tz)),
        "R80_gpu_mm": r80(gz, gd),
        "R80_topas_mm": r80(tz, td),
        "delta_R80_mm": r80(gz, gd) - r80(tz, td),
        "peak_z_gpu_mm": float(gz[int(np.argmax(gd))]),
        "peak_z_topas_mm": float(tz[int(np.argmax(td))]),
        "nrmse": float(np.sqrt(np.mean(diff * diff)) / ref_max),
        "dense_region_mean_gpu": float(np.mean(gd[dense])),
        "dense_region_mean_topas": float(np.mean(td[dense])),
        "dense_region_mean_rel_percent": 100.0
        * (float(np.mean(gd[dense])) - float(np.mean(td[dense])))
        / max(float(np.mean(td[dense])), 1e-12),
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(tz, td, lw=2, label="TOPAS")
    axes[0].plot(gz, gd, lw=1.4, label="GPU")
    axes[0].axvspan(50, 70, color="0.85", label="dense ρ=1.85")
    axes[0].set_ylabel("MeV/primary")
    axes[0].set_title(
        f"Density slab 200 MeV/u  ΔR80={report['delta_R80_mm']:+.2f} mm  "
        f"∫rel={report['integral_rel_percent']:+.2f}%"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(gz, 100.0 * diff / np.maximum(td, 1e-12), lw=1.2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axvspan(50, 70, color="0.85")
    axes[1].set_xlabel("Depth (mm)")
    axes[1].set_ylabel("GPU−TOPAS (%)")
    axes[1].set_ylim(-20, 20)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output_metrics}")
    print(f"Wrote {args.output_plot}")


if __name__ == "__main__":
    main()
