#!/usr/bin/env python3
"""Analyze maximum_step_mm convergence for carbon_mc depth-dose curves.

Compares multiple GPU/CPU IDD CSVs against the finest step (or a named
reference). Absolute MeV/primary only; no global scale.
"""

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
        raise SystemExit(f"{path} must contain depth_mm and {column}; found {names}")
    depth = np.atleast_1d(np.asarray(data["depth_mm"], dtype=float))
    dose = np.atleast_1d(np.asarray(data[column], dtype=float))
    if len(depth) < 3 or np.any(np.diff(depth) <= 0.0):
        raise SystemExit(f"{path} needs strictly increasing depths")
    if np.any(~np.isfinite(dose)) or float(np.max(dose)) <= 0.0:
        raise SystemExit(f"{path} has invalid dose data")
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
        "integral_MeV_per_primary": float(np.trapz(dose, depth)),
        "peak_depth_mm": float(depth[int(np.argmax(dose))]),
        "peak_value_MeV_per_primary": float(np.max(dose)),
        "R90_mm": r90,
        "R80_mm": r80,
        "R50_mm": r50,
        "R20_mm": r20,
        "FWHM_mm": r50 - p50,
        "tail_ge_90mm_MeV_per_primary": float(
            np.trapz(dose[depth >= 90.0], depth[depth >= 90.0])
        ),
    }


def compare_to_reference(
    depth: np.ndarray,
    dose: np.ndarray,
    ref_depth: np.ndarray,
    ref_dose: np.ndarray,
) -> dict[str, float]:
    eval_on_ref = np.interp(ref_depth, depth, dose)
    diff = eval_on_ref - ref_dose
    ref_max = float(np.max(ref_dose))
    nrmse = float(np.sqrt(np.mean(diff * diff)) / ref_max)
    integral_ref = float(np.trapz(ref_dose, ref_depth))
    integral_eval = float(np.trapz(eval_on_ref, ref_depth))
    tail = ref_depth >= 90.0
    tail_ref = float(np.trapz(ref_dose[tail], ref_depth[tail]))
    tail_eval = float(np.trapz(eval_on_ref[tail], ref_depth[tail]))
    return {
        "nrmse_to_reference_max": nrmse,
        "integral_signed_percent": 100.0 * (integral_eval - integral_ref) / integral_ref,
        "tail_ge_90mm_signed_percent": (
            100.0 * (tail_eval - tail_ref) / tail_ref if tail_ref > 0.0 else float("nan")
        ),
        "peak_value_signed_percent": 100.0
        * (float(np.max(eval_on_ref)) - ref_max)
        / ref_max,
        "max_abs_bin_MeV_per_primary": float(np.max(np.abs(diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="Pairs step_mm=path, e.g. 1.0=out/step_1p0.csv 0.5=out/step_0p5.csv",
    )
    parser.add_argument(
        "--column",
        default="energy_deposition_MeV_per_primary",
    )
    parser.add_argument(
        "--reference-step-mm",
        type=float,
        default=None,
        help="Reference step size (default: finest/smallest step among cases)",
    )
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    series: list[tuple[float, Path, np.ndarray, np.ndarray]] = []
    for item in args.cases:
        if "=" not in item:
            raise SystemExit(f"Expected step_mm=path, got {item}")
        step_text, path_text = item.split("=", 1)
        step = float(step_text)
        path = Path(path_text)
        depth, dose = load_curve(path, args.column)
        series.append((step, path, depth, dose))
    series.sort(key=lambda row: row[0], reverse=True)

    steps = [row[0] for row in series]
    reference_step = (
        args.reference_step_mm if args.reference_step_mm is not None else min(steps)
    )
    reference = next((row for row in series if abs(row[0] - reference_step) < 1e-12), None)
    if reference is None:
        raise SystemExit(f"Reference step {reference_step} mm not found in cases")

    ref_depth, ref_dose = reference[2], reference[3]
    ref_metrics = curve_metrics(ref_depth, ref_dose)

    cases_out: list[dict[str, object]] = []
    for step, path, depth, dose in series:
        metrics = curve_metrics(depth, dose)
        relative = compare_to_reference(depth, dose, ref_depth, ref_dose)
        cases_out.append(
            {
                "maximum_step_mm": step,
                "path": path.as_posix(),
                "is_reference": abs(step - reference_step) < 1e-12,
                "metrics": metrics,
                "vs_reference": relative,
                "delta_R80_mm": metrics["R80_mm"] - ref_metrics["R80_mm"],
                "delta_peak_depth_mm": metrics["peak_depth_mm"] - ref_metrics["peak_depth_mm"],
                "delta_FWHM_mm": metrics["FWHM_mm"] - ref_metrics["FWHM_mm"],
            }
        )

    report = {
        "normalization": "absolute MeV/primary; no global scale",
        "reference_step_mm": reference_step,
        "column": args.column,
        "cases": cases_out,
        "convergence_notes": {
            "stable_if": (
                "finest two steps agree within ~0.5% integral and ~0.1 mm R80/FWHM "
                "for a fixed history count; re-check with higher statistics before "
                "claiming final step recommendation"
            ),
            "neutron_deferred": True,
        },
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for step, path, depth, dose in series:
            label = f"{step:g} mm"
            if abs(step - reference_step) < 1e-12:
                label += " (ref)"
            axes[0].plot(depth, dose, label=label, linewidth=1.4)
        axes[0].set_xlabel("Depth (mm)")
        axes[0].set_ylabel("MeV/primary")
        axes[0].set_title("Depth dose vs maximum_step_mm")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        nrmse = [case["vs_reference"]["nrmse_to_reference_max"] for case in cases_out]
        axes[1].semilogx(steps, nrmse, "o-", color="#1f77b4")
        axes[1].set_xlabel("maximum_step_mm")
        axes[1].set_ylabel("NRMSE to reference max")
        axes[1].set_title("Step convergence")
        axes[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=140)
        plt.close(fig)

    print(f"Wrote {args.output_metrics}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")
    for case in cases_out:
        step = case["maximum_step_mm"]
        vs = case["vs_reference"]
        print(
            f"step={step:g} mm  NRMSE={vs['nrmse_to_reference_max']:.4e}  "
            f"integralΔ={vs['integral_signed_percent']:+.3f}%  "
            f"ΔR80={case['delta_R80_mm']:+.3f} mm  "
            f"ΔFWHM={case['delta_FWHM_mm']:+.3f} mm"
        )


if __name__ == "__main__":
    main()
