#!/usr/bin/env python3
"""Analyze multi-energy (100--400 MeV/u) carbon_mc IDD series.

Absolute MeV/primary; fixed model parameters (no per-energy scales).
Optional TOPAS reference comparison at 200 MeV/u only when available.
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
        raise SystemExit(f"{path} has invalid dose")
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
    # CSV column is energy deposited *per bin* (not per mm); sum over bins.
    return {
        "integral_MeV_per_primary": float(np.sum(dose)),
        "peak_depth_mm": float(depth[int(np.argmax(dose))]),
        "peak_value_MeV_per_primary": float(np.max(dose)),
        "R90_mm": r90,
        "R80_mm": r80,
        "R50_mm": r50,
        "R20_mm": r20,
        "FWHM_mm": r50 - p50,
        "tail_ge_R80_MeV_per_primary": float(
            np.sum(dose[depth >= r80]) if np.isfinite(r80) else float("nan")
        ),
    }


def compare_curves(
    eval_depth: np.ndarray,
    eval_dose: np.ndarray,
    ref_depth: np.ndarray,
    ref_dose: np.ndarray,
) -> dict[str, float]:
    on_ref = np.interp(ref_depth, eval_depth, eval_dose)
    diff = on_ref - ref_dose
    ref_max = float(np.max(ref_dose))
    ref_metrics = curve_metrics(ref_depth, ref_dose)
    eval_metrics = curve_metrics(eval_depth, eval_dose)
    return {
        "nrmse_to_reference_max": float(np.sqrt(np.mean(diff * diff)) / ref_max),
        "integral_signed_percent": 100.0
        * (float(np.sum(on_ref)) - float(np.sum(ref_dose)))
        / float(np.sum(ref_dose)),
        "peak_value_signed_percent": 100.0
        * (eval_metrics["peak_value_MeV_per_primary"] - ref_metrics["peak_value_MeV_per_primary"])
        / ref_metrics["peak_value_MeV_per_primary"],
        "delta_R80_mm": eval_metrics["R80_mm"] - ref_metrics["R80_mm"],
        "delta_FWHM_mm": eval_metrics["FWHM_mm"] - ref_metrics["FWHM_mm"],
        "delta_peak_depth_mm": eval_metrics["peak_depth_mm"] - ref_metrics["peak_depth_mm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="energy_MeVu=path pairs, e.g. 100=out/e100.csv 200=out/e200.csv",
    )
    parser.add_argument(
        "--topas-200",
        type=Path,
        default=None,
        help="Optional TOPAS absolute IDD at 200 MeV/u",
    )
    parser.add_argument("--topas-column", default="energy_deposition_MeV_per_primary")
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    series: list[tuple[float, Path, np.ndarray, np.ndarray]] = []
    for item in args.cases:
        if "=" not in item:
            raise SystemExit(f"Expected energy=path, got {item}")
        e_text, path_text = item.split("=", 1)
        energy = float(e_text)
        path = Path(path_text)
        depth, dose = load_curve(path, "energy_deposition_MeV_per_primary")
        series.append((energy, path, depth, dose))
    series.sort(key=lambda row: row[0])

    cases_out: list[dict[str, object]] = []
    for energy, path, depth, dose in series:
        metrics = curve_metrics(depth, dose)
        cases_out.append(
            {
                "energy_MeV_per_u": energy,
                "path": path.as_posix(),
                "metrics": metrics,
                "R80_over_energy_mm_per_MeVu": metrics["R80_mm"] / energy
                if energy > 0
                else float("nan"),
            }
        )

    # Empirical range scaling check: R80 should increase with energy.
    r80s = [float(c["metrics"]["R80_mm"]) for c in cases_out]  # type: ignore[index]
    energies = [float(c["energy_MeV_per_u"]) for c in cases_out]
    monotonic = all(r80s[i] < r80s[i + 1] for i in range(len(r80s) - 1))

    topas_compare = None
    if args.topas_200 is not None:
        gpu_200 = next((row for row in series if abs(row[0] - 200.0) < 1e-9), None)
        if gpu_200 is None:
            raise SystemExit("TOPAS-200 comparison requires a 200 MeV/u GPU case")
        topas_depth, topas_dose = load_curve(args.topas_200, args.topas_column)
        topas_compare = {
            "topas_path": args.topas_200.as_posix(),
            "topas_column": args.topas_column,
            "topas_metrics": curve_metrics(topas_depth, topas_dose),
            "gpu_vs_topas": compare_curves(
                gpu_200[2], gpu_200[3], topas_depth, topas_dose
            ),
        }

    report = {
        "normalization": "absolute MeV/primary; no global scale; no per-energy scale",
        "neutron_deferred": True,
        "reaction_package_note": (
            "Primary reaction final-state packages are binned 0--200 MeV/u from the "
            "200 MeV/u TOPAS campaign. Beams at E>200 MeV/u clamp high-energy reaction "
            "sampling to the top bin; stopping power and nuclear cross sections still "
            "use the full 1--400 MeV/u tables."
        ),
        "cases": cases_out,
        "range_trend": {
            "R80_mm": r80s,
            "energy_MeV_per_u": energies,
            "R80_strictly_increasing": monotonic,
        },
        "topas_200_comparison": topas_compare,
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for energy, path, depth, dose in series:
            axes[0].plot(depth, dose, label=f"{energy:g} MeV/u", linewidth=1.3)
        if args.topas_200 is not None:
            td, tdose = load_curve(args.topas_200, args.topas_column)
            axes[0].plot(td, tdose, "k--", linewidth=1.0, label="TOPAS 200 MeV/u")
        axes[0].set_xlabel("Depth (mm)")
        axes[0].set_ylabel("MeV/primary")
        axes[0].set_title("Multi-energy IDD (absolute)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(energies, r80s, "o-", color="#1f77b4")
        axes[1].set_xlabel("Energy (MeV/u)")
        axes[1].set_ylabel("R80 (mm)")
        axes[1].set_title("R80 vs energy")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=140)
        plt.close(fig)

    print(f"Wrote {args.output_metrics}")
    if args.output_plot:
        print(f"Wrote {args.output_plot}")
    for case in cases_out:
        m = case["metrics"]
        print(
            f"E={case['energy_MeV_per_u']:g}  R80={m['R80_mm']:.3f} mm  "
            f"peak_z={m['peak_depth_mm']:.2f}  FWHM={m['FWHM_mm']:.3f}  "
            f"integral={m['integral_MeV_per_primary']:.2f}"
        )
    print(f"R80 strictly increasing: {monotonic}")
    if topas_compare is not None:
        g = topas_compare["gpu_vs_topas"]
        print(
            f"GPU200 vs TOPAS: NRMSE={g['nrmse_to_reference_max']:.4e}  "
            f"integralΔ={g['integral_signed_percent']:+.3f}%  "
            f"ΔR80={g['delta_R80_mm']:+.3f} mm"
        )


if __name__ == "__main__":
    main()
