#!/usr/bin/env python3
"""Compare standardized TOPAS and carbon_mc depth-dose CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_curve(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = data.dtype.names or ()
    if "depth_mm" not in names or column not in names:
        raise ValueError(f"{path} must contain depth_mm and {column}; found {names}")
    depth = np.atleast_1d(data["depth_mm"])
    dose = np.atleast_1d(data[column])
    if len(depth) < 3 or np.any(np.diff(depth) <= 0.0):
        raise ValueError(f"{path} must contain at least three strictly increasing depths")
    if np.any(~np.isfinite(dose)) or np.max(dose) <= 0.0:
        raise ValueError(f"{path} contains invalid or nonpositive dose data")
    return depth, dose


def crossing(depth: np.ndarray, normalized: np.ndarray, level: float, distal: bool) -> float:
    peak = int(np.argmax(normalized))
    if distal:
        for index in range(peak + 1, len(depth)):
            if normalized[index] <= level < normalized[index - 1]:
                x0, x1 = depth[index - 1], depth[index]
                y0, y1 = normalized[index - 1], normalized[index]
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    else:
        for index in range(peak, 0, -1):
            if normalized[index - 1] <= level < normalized[index]:
                x0, x1 = depth[index - 1], depth[index]
                y0, y1 = normalized[index - 1], normalized[index]
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def curve_metrics(depth: np.ndarray, dose: np.ndarray) -> dict[str, float]:
    normalized = dose / np.max(dose)
    r90 = crossing(depth, normalized, 0.9, True)
    r80 = crossing(depth, normalized, 0.8, True)
    r50 = crossing(depth, normalized, 0.5, True)
    r20 = crossing(depth, normalized, 0.2, True)
    p50 = crossing(depth, normalized, 0.5, False)
    return {
        "peak_depth_mm": float(depth[int(np.argmax(dose))]),
        "peak_value": float(np.max(dose)),
        "R90_mm": r90,
        "R80_mm": r80,
        "R50_mm": r50,
        "R20_mm": r20,
        "distal_falloff_R20_minus_R80_mm": r20 - r80,
        "FWHM_mm": r50 - p50,
    }


def gamma_pass_rate(
    reference_depth: np.ndarray,
    reference_dose: np.ndarray,
    evaluation_depth: np.ndarray,
    evaluation_dose: np.ndarray,
    dose_percent: float,
    distance_mm: float,
    threshold_percent: float,
) -> float:
    reference_maximum = float(np.max(reference_dose))
    threshold = threshold_percent / 100.0 * reference_maximum
    selected = np.flatnonzero(reference_dose >= threshold)
    if len(selected) == 0:
        return float("nan")
    dose_criterion = dose_percent / 100.0 * reference_maximum
    passed = 0
    for index in selected:
        gamma_squared = (
            (evaluation_depth - reference_depth[index]) / distance_mm
        ) ** 2 + ((evaluation_dose - reference_dose[index]) / dose_criterion) ** 2
        passed += float(np.min(gamma_squared)) <= 1.0
    return 100.0 * passed / len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topas", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results/compare"))
    parser.add_argument(
        "--column", default="energy_deposition_MeV_per_primary"
    )
    parser.add_argument("--gamma-threshold-percent", type=float, default=10.0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Also write the metrics JSON to this path.",
    )
    args = parser.parse_args()

    topas_depth, topas_dose = load_curve(args.topas, args.column)
    eval_depth, eval_dose = load_curve(args.evaluation, args.column)
    interpolated_eval = np.interp(topas_depth, eval_depth, eval_dose)

    topas_metrics = curve_metrics(topas_depth, topas_dose)
    eval_metrics = curve_metrics(eval_depth, eval_dose)
    difference = interpolated_eval - topas_dose
    nrmse = float(np.sqrt(np.mean(difference * difference)) / np.max(topas_dose))
    peak_error_percent = 100.0 * (
        eval_metrics["peak_value"] - topas_metrics["peak_value"]
    ) / topas_metrics["peak_value"]
    r80 = topas_metrics["R80_mm"]
    tail_mask = topas_depth >= r80
    topas_tail = float(np.trapz(topas_dose[tail_mask], topas_depth[tail_mask]))
    eval_tail = float(np.trapz(interpolated_eval[tail_mask], topas_depth[tail_mask]))

    report = {
        "topas": topas_metrics,
        "evaluation": eval_metrics,
        "differences": {
            "R80_mm": eval_metrics["R80_mm"] - topas_metrics["R80_mm"],
            "FWHM_relative_percent": 100.0
            * (eval_metrics["FWHM_mm"] - topas_metrics["FWHM_mm"])
            / topas_metrics["FWHM_mm"],
            "peak_value_percent": peak_error_percent,
            "NRMSE": nrmse,
            "tail_integral_relative_percent": 100.0 * (eval_tail - topas_tail) / topas_tail,
        },
        "gamma": {
            "1pct_1mm_pass_rate_percent": gamma_pass_rate(
                topas_depth,
                topas_dose,
                eval_depth,
                eval_dose,
                1.0,
                1.0,
                args.gamma_threshold_percent,
            ),
            "2pct_2mm_pass_rate_percent": gamma_pass_rate(
                topas_depth,
                topas_dose,
                eval_depth,
                eval_dose,
                2.0,
                2.0,
                args.gamma_threshold_percent,
            ),
            "threshold_percent": args.gamma_threshold_percent,
            "normalization": "global TOPAS maximum",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )

    topas_normalized = topas_dose / np.max(topas_dose)
    eval_normalized = eval_dose / np.max(eval_dose)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(topas_depth, topas_normalized, label="TOPAS", linewidth=2)
    axis.plot(eval_depth, eval_normalized, label="carbon_mc", linewidth=1.5)
    axis.set(xlabel="Depth (mm)", ylabel="Normalized depth dose", ylim=(0.0, 1.08))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "depth_dose.png", dpi=180)
    plt.close(fig)

    peak_center = topas_metrics["peak_depth_mm"]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(topas_depth, topas_normalized, label="TOPAS", linewidth=2)
    axis.plot(eval_depth, eval_normalized, label="carbon_mc", linewidth=1.5)
    axis.set(
        xlabel="Depth (mm)",
        ylabel="Normalized depth dose",
        xlim=(peak_center - 15.0, peak_center + 15.0),
        ylim=(0.0, 1.08),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "bragg_peak.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(topas_depth, np.maximum(topas_normalized, 1.0e-8), label="TOPAS")
    axis.semilogy(eval_depth, np.maximum(eval_normalized, 1.0e-8), label="carbon_mc")
    axis.set(xlabel="Depth (mm)", ylabel="Normalized depth dose", ylim=(1.0e-6, 1.2))
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "tail_log.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
