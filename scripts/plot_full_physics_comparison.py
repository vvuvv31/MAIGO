#!/usr/bin/env python3
"""Plot a full-physics proton GPU profile against TOPAS.

The dose overlay uses the absolute TOPAS DoseToMedium scorer and the GPU
``dose_Gy.csv`` dose column.  TOPAS all-hadron LETd is rebuilt
from its numerator and denominator moments; the GPU value is read from the
reported LETd column.  Relative-error statistics and plots use bins where
TOPAS dose is greater than one percent of its peak.  LETd also requires a
positive TOPAS denominator and positive TOPAS LETd, so no zero denominator is
ever divided by.

The default input paths are the requested 70 MeV full-physics comparison
paths.  A missing GPU output is reported as an actionable error and does not
produce a plot or metrics file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


TOPAS_DEFAULT = Path(
    "out/proton_water_qgsp_bic_hp/compare_70MeV_100k/topas_reference_100k"
)
GPU_DEFAULT = Path(
    "out/proton_water_qgsp_bic_hp/full_physics_compare_100k/70MeV/gpu"
)
OUTPUT_DEFAULT = Path(
    "out/proton_water_qgsp_bic_hp/full_physics_compare_100k/70MeV/comparison"
)

TOPAS_DOSE_FILE = "total_dose.csv"
TOPAS_LET_NUMERATOR_FILE = "all_hadron_let.csv"
TOPAS_LET_DENOMINATOR_FILE = "all_hadron_let_denominator.csv"
GPU_DOSE_FILE = "dose_Gy.csv"
GPU_LET_FILE = "letd.csv"

TOPAS_DOSE_COLUMN = "dose_Gy"
TOPAS_LET_NUMERATOR_COLUMN = "numerator_MeV"
TOPAS_LET_DENOMINATOR_COLUMN = "denominator_MeV"
GPU_DOSE_COLUMN = "dose_Gy"
GPU_LET_COLUMN = "all_hadron_letd_MeV_per_mm_per_g_cm3"


class InputError(ValueError):
    """Raised for an unusable comparison input."""


def read_profile(path: Path, column: str, *, nonnegative: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Read a depth profile while accepting TOPAS comment preambles."""

    if not path.is_file():
        raise InputError(f"missing required input file: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(line for line in stream if not line.lstrip().startswith("#")))
    except OSError as error:
        raise InputError(f"cannot read {path}: {error}") from error
    if not rows or "depth_mm" not in rows[0] or column not in rows[0]:
        available = ", ".join(rows[0].keys()) if rows else "no header"
        raise InputError(f"{path}: requires depth_mm and {column}; columns: {available}")
    try:
        depth = np.asarray([float(row["depth_mm"]) for row in rows], dtype=np.float64)
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f"{path}: depth_mm and {column} must be numeric") from error
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(values)):
        raise InputError(f"{path}: depth_mm and {column} must be finite")
    if nonnegative and np.any(values < 0.0):
        raise InputError(f"{path}: {column} must be non-negative")
    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]
    if len(depth) < 2 or np.any(np.diff(depth) <= 0.0):
        raise InputError(f"{path}: depth_mm must contain at least two unique, increasing bins")
    return depth, values


def align_profile(
    reference_depth: np.ndarray,
    candidate_depth: np.ndarray,
    candidate: np.ndarray,
    label: str,
) -> np.ndarray:
    """Align a candidate profile to the reference grid without extrapolation."""

    if reference_depth[0] < candidate_depth[0] - 1.0e-9 or reference_depth[-1] > candidate_depth[-1] + 1.0e-9:
        raise InputError(f"{label}: candidate depth range does not cover TOPAS grid")
    if len(reference_depth) == len(candidate_depth) and np.allclose(
        reference_depth, candidate_depth, rtol=0.0, atol=1.0e-9
    ):
        return candidate.copy()
    return np.interp(reference_depth, candidate_depth, candidate)


def safe_integral(values: np.ndarray, depth: np.ndarray) -> float:
    """Use the NumPy 2 spelling while keeping compatibility with older NumPy."""

    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, depth))


def error_summary(error_percent: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error_percent[mask], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise InputError("comparison mask contains no finite relative-error values")
    absolute = np.abs(values)
    return {
        "bins": int(values.size),
        "mean_signed_percent": float(np.mean(values)),
        "median_signed_percent": float(np.median(values)),
        "mean_absolute_percent": float(np.mean(absolute)),
        "p95_absolute_percent": float(np.percentile(absolute, 95.0)),
        "max_absolute_percent": float(np.max(absolute)),
        "minimum_signed_percent": float(np.min(values)),
        "maximum_signed_percent": float(np.max(values)),
    }


def relative_error_percent(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return signed candidate/reference error, leaving invalid bins as NaN."""

    result = np.full(reference.shape, np.nan, dtype=np.float64)
    valid = mask & np.isfinite(reference) & np.isfinite(candidate) & (reference != 0.0)
    result[valid] = 100.0 * (candidate[valid] / reference[valid] - 1.0)
    return result


def load_inputs(topas_dir: Path, gpu_dir: Path) -> dict[str, Any]:
    topas_dose_path = topas_dir / TOPAS_DOSE_FILE
    topas_num_path = topas_dir / TOPAS_LET_NUMERATOR_FILE
    topas_den_path = topas_dir / TOPAS_LET_DENOMINATOR_FILE
    gpu_dose_path = gpu_dir / GPU_DOSE_FILE
    gpu_let_path = gpu_dir / GPU_LET_FILE
    required = [topas_dose_path, topas_num_path, topas_den_path, gpu_dose_path, gpu_let_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise InputError(
            "missing required comparison input file(s):\n  - " + "\n  - ".join(missing)
        )

    topas_depth, topas_dose_raw = read_profile(topas_dose_path, TOPAS_DOSE_COLUMN)
    topas_num_depth, topas_num = read_profile(topas_num_path, TOPAS_LET_NUMERATOR_COLUMN)
    topas_den_depth, topas_den = read_profile(topas_den_path, TOPAS_LET_DENOMINATOR_COLUMN)
    gpu_dose_depth, gpu_dose_raw = read_profile(gpu_dose_path, GPU_DOSE_COLUMN)
    gpu_let_depth, gpu_letd = read_profile(gpu_let_path, GPU_LET_COLUMN)

    topas_num = align_profile(topas_depth, topas_num_depth, topas_num, "TOPAS LET numerator")
    topas_den = align_profile(topas_depth, topas_den_depth, topas_den, "TOPAS LET denominator")
    gpu_dose = align_profile(topas_depth, gpu_dose_depth, gpu_dose_raw, "GPU dose")
    gpu_letd = align_profile(topas_depth, gpu_let_depth, gpu_letd, "GPU LETd")
    topas_letd = np.full(topas_depth.shape, np.nan, dtype=np.float64)
    positive_denominator = topas_den > 0.0
    topas_letd[positive_denominator] = topas_num[positive_denominator] / topas_den[positive_denominator]
    if not np.all(np.isfinite(topas_letd[positive_denominator])):
        raise InputError("TOPAS LET numerator/denominator produces non-finite LETd")
    return {
        "topas_depth": topas_depth,
        "topas_dose_raw": topas_dose_raw,
        "topas_let_numerator": topas_num,
        "topas_let_denominator": topas_den,
        "topas_letd": topas_letd,
        "gpu_dose_raw": gpu_dose,
        "gpu_letd": gpu_letd,
        "paths": {
            "topas_dose": topas_dose_path,
            "topas_let_numerator": topas_num_path,
            "topas_let_denominator": topas_den_path,
            "gpu_dose": gpu_dose_path,
            "gpu_letd": gpu_let_path,
        },
    }


def build_report(
    data: dict[str, Any],
    topas_dir: Path,
    gpu_dir: Path,
    energy_mev: float,
    histories: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    depth = data["topas_depth"]
    topas_dose = data["topas_dose_raw"]
    gpu_dose = data["gpu_dose_raw"]
    topas_peak = float(np.max(topas_dose))
    gpu_peak = float(np.max(gpu_dose))
    if topas_peak <= 0.0 or gpu_peak <= 0.0:
        raise InputError("dose profiles must have a positive peak")
    dose_mask = topas_dose > 0.01 * topas_peak
    if not np.any(dose_mask):
        raise InputError("TOPAS dose has no bins above the one-percent peak mask")

    topas_letd = data["topas_letd"]
    gpu_letd = data["gpu_letd"]
    let_mask = dose_mask & (data["topas_let_denominator"] > 0.0) & (topas_letd > 0.0)
    if not np.any(let_mask):
        raise InputError("TOPAS LETd has no positive-denominator bins within the one-percent dose mask")
    dose_error = relative_error_percent(topas_dose, gpu_dose, dose_mask)
    let_error = relative_error_percent(topas_letd, gpu_letd, let_mask)

    topas_peak_depth = float(depth[int(np.argmax(topas_dose))])
    gpu_peak_depth = float(depth[int(np.argmax(gpu_dose))])
    dose_integral_topas = safe_integral(topas_dose, depth)
    dose_integral_gpu = safe_integral(gpu_dose, depth)
    integral_relative = 100.0 * (dose_integral_gpu / dose_integral_topas - 1.0)
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "complete",
        "scenario": {
            "projectile": "proton",
            "initial_energy_MeV": energy_mev,
            "histories": histories,
            "physics_scope": "full physics",
        },
        "comparison": {
            "depth_bins": int(depth.size),
            "depth_range_mm": [float(depth[0]), float(depth[-1])],
            "dose_quantity": "absolute DoseToMedium in Gy; no normalization",
            "relative_error_definition": "100 * (GPU / TOPAS - 1)",
            "dose_mask": "absolute TOPAS dose > 1% of TOPAS peak dose",
            "letd_mask": "dose mask and TOPAS LET denominator > 0 and TOPAS LETd > 0",
            "dose_mask_bins": int(np.count_nonzero(dose_mask)),
            "letd_mask_bins": int(np.count_nonzero(let_mask)),
        },
        "metrics": {
            "dose_absolute_Gy": {
                "topas_peak_dose_Gy": topas_peak,
                "gpu_peak_dose_Gy": gpu_peak,
                "topas_peak_depth_mm": topas_peak_depth,
                "gpu_peak_depth_mm": gpu_peak_depth,
                "peak_depth_difference_mm": gpu_peak_depth - topas_peak_depth,
                "topas_depth_integral_Gy_mm": dose_integral_topas,
                "gpu_depth_integral_Gy_mm": dose_integral_gpu,
                "integral_relative_difference_percent": integral_relative,
                "signed_relative_error_percent": error_summary(dose_error, dose_mask),
            },
            "all_hadron_letd": {
                "topas_letd_peak_MeV_per_mm_per_g_cm3": float(np.nanmax(topas_letd[let_mask])),
                "gpu_letd_peak_MeV_per_mm_per_g_cm3": float(np.nanmax(gpu_letd[let_mask])),
                "signed_relative_error_percent": error_summary(let_error, let_mask),
            },
        },
        "sources": {
            "topas_reference_directory": str(topas_dir),
            "gpu_directory": str(gpu_dir),
            "topas_dose": {"file": str(data["paths"]["topas_dose"]), "column": TOPAS_DOSE_COLUMN},
            "topas_let_numerator": {
                "file": str(data["paths"]["topas_let_numerator"]),
                "column": TOPAS_LET_NUMERATOR_COLUMN,
            },
            "topas_let_denominator": {
                "file": str(data["paths"]["topas_let_denominator"]),
                "column": TOPAS_LET_DENOMINATOR_COLUMN,
            },
            "gpu_dose": {"file": str(data["paths"]["gpu_dose"]), "column": GPU_DOSE_COLUMN},
            "gpu_letd": {"file": str(data["paths"]["gpu_letd"]), "column": GPU_LET_COLUMN},
        },
    }
    curves = {
        "depth": depth,
        "topas_dose_Gy": topas_dose,
        "gpu_dose_Gy": gpu_dose,
        "topas_letd": topas_letd,
        "gpu_letd": gpu_letd,
        "dose_error": dose_error,
        "let_error": let_error,
        "dose_mask": dose_mask,
        "let_mask": let_mask,
    }
    return report, curves


def error_ylim(error: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = np.abs(error[mask])
    bound = max(2.0, float(np.percentile(values, 99.0)) * 1.15, float(np.max(values)) * 1.02)
    return -bound, bound


def plot_comparison(report: dict[str, Any], curves: dict[str, np.ndarray], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = curves["depth"]
    topas_color = "#1769aa"
    gpu_color = "#c43d3d"
    error_color = "#3b3b3b"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex="col", constrained_layout=True)
    scenario = report["scenario"]
    history_label = f'{scenario["histories"] // 1_000_000}M' if scenario["histories"] % 1_000_000 == 0 else f'{scenario["histories"] / 1_000:g}k'
    fig.suptitle(
        f'{scenario["initial_energy_MeV"]:g} MeV proton | {history_label} histories | Full physics: GPU vs TOPAS',
        fontsize=16,
        fontweight="bold",
    )

    dose_ax, dose_error_ax = axes[0]
    dose_ax.plot(depth, curves["topas_dose_Gy"], color=topas_color, lw=1.8, label="TOPAS")
    dose_ax.plot(depth, curves["gpu_dose_Gy"], color=gpu_color, lw=1.5, label="GPU")
    dose_ax.set_title("Absolute dose profile")
    dose_ax.set_ylabel("Dose to medium (Gy)")
    dose_ax.set_ylim(bottom=0.0)
    dose_ax.grid(True, alpha=0.25)
    dose_ax.legend(loc="upper left", frameon=False, ncol=2)
    dose_ax.text(
        0.99,
        0.96,
        ">1% TOPAS peak mask for errors",
        transform=dose_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    dose_error_ax.plot(depth, curves["dose_error"], color=error_color, lw=1.1)
    dose_error_ax.axhline(0.0, color="#888888", lw=0.9)
    dose_error_ax.set_title("Signed relative error: dose")
    dose_error_ax.set_ylabel("GPU / TOPAS - 1 (%)")
    dose_error_ax.set_ylim(-6.0, 6.0)
    dose_error_ax.grid(True, alpha=0.25)

    let_ax, let_error_ax = axes[1]
    let_ax.plot(depth, np.where(curves["let_mask"], curves["topas_letd"], np.nan), color=topas_color, lw=1.8, label="TOPAS")
    let_ax.plot(depth, np.where(curves["let_mask"], curves["gpu_letd"], np.nan), color=gpu_color, lw=1.5, label="GPU")
    let_ax.set_title("All-hadron LETd profile")
    let_ax.set_xlabel("Depth (mm)")
    let_ax.set_ylabel("LETd (MeV/mm per g/cm³)")
    let_ax.set_ylim(bottom=0.0)
    let_ax.grid(True, alpha=0.25)
    let_ax.legend(loc="upper left", frameon=False, ncol=2)
    let_error_ax.plot(depth, curves["let_error"], color=error_color, lw=1.1)
    let_error_ax.axhline(0.0, color="#888888", lw=0.9)
    let_error_ax.set_title("Signed relative error: LETd")
    let_error_ax.set_xlabel("Depth (mm)")
    let_error_ax.set_ylabel("GPU / TOPAS - 1 (%)")
    let_error_ax.set_ylim(-6.0, 6.0)
    let_error_ax.grid(True, alpha=0.25)

    # The phantom extends well beyond the proton range; keep a small distal
    # margin so the dose and LET structure remains legible at this scale.
    plot_end = min(float(depth[-1]), float(depth[curves["dose_mask"]][-1]) + 5.0)
    for axis in axes.flat:
        axis.set_xlim(float(depth[0]), plot_end)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas-reference-dir", type=Path, default=TOPAS_DEFAULT)
    parser.add_argument("--gpu-dir", type=Path, default=GPU_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--prefix", default="full_physics_70MeV_100k_comparison")
    parser.add_argument("--energy-mev", type=float, default=70.0)
    parser.add_argument("--histories", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.energy_mev <= 0.0 or args.histories <= 0:
        raise SystemExit("ERROR: --energy-mev and --histories must be positive")
    try:
        data = load_inputs(args.topas_reference_dir, args.gpu_dir)
        report, curves = build_report(
            data,
            args.topas_reference_dir,
            args.gpu_dir,
            args.energy_mev,
            args.histories,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = args.output_dir / f"{args.prefix}.png"
        pdf_path = args.output_dir / f"{args.prefix}.pdf"
        json_path = args.output_dir / f"{args.prefix}.json"
        plot_comparison(report, curves, png_path, pdf_path)
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except InputError as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Metrics: {json_path}")


if __name__ == "__main__":
    main()
