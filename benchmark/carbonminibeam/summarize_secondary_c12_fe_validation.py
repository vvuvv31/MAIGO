#!/usr/bin/env python3
"""Summarize same-source primary/secondary C12 water-transport validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-phase-metrics", type=Path)
    parser.add_argument("--primary-dose-metrics", type=Path)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--fe010", type=Path, required=True)
    parser.add_argument("--fe005", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_case(
    root: Path, phase_metrics: Path | None = None, dose_metrics: Path | None = None
) -> dict:
    phase_metrics = phase_metrics or (
        root / "phase_comparison/water_primary_phase_space_metrics.json"
    )
    dose_metrics = dose_metrics or (root / "dose_comparison/metrics.json")
    phase = json.loads(
        phase_metrics.read_text(encoding="utf-8")
    )
    dose = json.loads(dose_metrics.read_text(encoding="utf-8"))
    return {"root": str(root), "phase": phase, "dose": dose}


def ratio(gpu: dict, topas: dict, key: str) -> float:
    return float(gpu[key] / topas[key])


def plane_rows(case: dict) -> list[dict]:
    rows = []
    for depth in case["phase"]["depths"]:
        gpu, topas = depth["gpu"], depth["topas"]
        rows.append(
            {
                "depth_mm": depth["depth_mm"],
                "survival_ratio": ratio(gpu, topas, "survival"),
                "mean_energy_ratio": ratio(gpu, topas, "mean_energy_MeV"),
                "folded_x_variance_ratio": ratio(
                    gpu, topas, "folded_var_x_mm2"
                ),
                "theta_x_variance_ratio": ratio(
                    gpu, topas, "var_theta_x_rad2"
                ),
                "folded_x_theta_x_covariance_ratio": ratio(
                    gpu, topas, "folded_x_theta_x_cov_mm_rad"
                ),
                "theta_x_q68_ratio": (
                    gpu["abs_theta_x_quantiles_mrad"]["q68"]
                    / topas["abs_theta_x_quantiles_mrad"]["q68"]
                ),
                "theta_x_q95_ratio": (
                    gpu["abs_theta_x_quantiles_mrad"]["q95"]
                    / topas["abs_theta_x_quantiles_mrad"]["q95"]
                ),
                "theta_x_q99_ratio": (
                    gpu["abs_theta_x_quantiles_mrad"]["q99"]
                    / topas["abs_theta_x_quantiles_mrad"]["q99"]
                ),
                "theta_x_q999_ratio": (
                    gpu["abs_theta_x_quantiles_mrad"]["q999"]
                    / topas["abs_theta_x_quantiles_mrad"]["q999"]
                ),
                "peak_fluence_ratio": (
                    gpu["regions"]["peak"]["fluence_per_input"]
                    / topas["regions"]["peak"]["fluence_per_input"]
                ),
                "shoulder_fluence_ratio": (
                    gpu["regions"]["shoulder"]["fluence_per_input"]
                    / topas["regions"]["shoulder"]["fluence_per_input"]
                ),
                "valley_fluence_ratio": (
                    gpu["regions"]["valley"]["fluence_per_input"]
                    / topas["regions"]["valley"]["fluence_per_input"]
                ),
            }
        )
    return rows


def dose_summary(case: dict) -> dict:
    dose = case["dose"]
    fixed = {}
    for row in dose["fixed_regions"]["selected_depths"]:
        fixed[str(row["depth_mm"])] = {
            key: row[key]
            for key in (
                "peak_gpu_over_topas",
                "shoulder_gpu_over_topas",
                "valley_gpu_over_topas",
            )
        }
    return {
        "total_ratio": dose["gpu_over_topas_dose_sum"],
        "voxel_l1_over_topas": dose["two_dimensional_l1_over_topas"],
        "idd_l1_over_topas": dose["idd_l1_over_topas"],
        "bragg_peak_depth_mm": dose["bragg_peak_depth_mm"],
        "fixed_regions": fixed,
    }


def main() -> None:
    args = parse_args()
    cases = {
        "primary FE": load_case(
            args.primary, args.primary_phase_metrics, args.primary_dose_metrics
        ),
        "secondary legacy": load_case(args.legacy),
        "secondary FE 0.10 mm": load_case(args.fe010),
        "secondary FE 0.05 mm": load_case(args.fe005),
    }
    summary = {
        "comparison": "same 1,622,795 water-entry C12 records",
        "normalization": "per replayed C12; ratios unchanged by the common original-history scale",
        "cases": {
            name: {
                "root": case["root"],
                "planes": plane_rows(case),
                "dose": dose_summary(case),
            }
            for name, case in cases.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "secondary_c12_fe_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    fields = [
        ("theta_x_variance_ratio", r"Var($\theta_x$)"),
        ("folded_x_variance_ratio", r"Var(folded $x$)"),
        ("folded_x_theta_x_covariance_ratio", r"Cov(folded $x$, $\theta_x$)"),
        ("theta_x_q99_ratio", r"$|\theta_x|$ q99"),
    ]
    styles = ["o-", "s-", "^-", "d--"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (field, title) in zip(axes.flat, fields):
        for (name, case), style in zip(cases.items(), styles):
            rows = plane_rows(case)
            axis.plot(
                [row["depth_mm"] for row in rows],
                [row[field] for row in rows],
                style,
                label=name,
                linewidth=1.5,
                markersize=4,
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("GPU / TOPAS")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Water depth (mm)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Same-source C12 water transport: primary vs secondary path")
    fig.tight_layout()
    fig.savefig(args.output_dir / "secondary_c12_fe_plane_ratios.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
