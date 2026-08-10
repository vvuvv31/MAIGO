#!/usr/bin/env python3
"""Statistical convergence and same-seed reproducibility for carbon_mc IDDs.

Absolute MeV/primary; no global scale. Neutron work remains deferred.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = data.dtype.names or ()
    if "depth_mm" not in names or "energy_deposition_MeV_per_primary" not in names:
        raise SystemExit(f"{path} missing depth/energy columns: {names}")
    depth = np.atleast_1d(np.asarray(data["depth_mm"], dtype=float))
    dose = np.atleast_1d(
        np.asarray(data["energy_deposition_MeV_per_primary"], dtype=float)
    )
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
    r80 = crossing(depth, normalized, 0.8, True)
    r50 = crossing(depth, normalized, 0.5, True)
    p50 = crossing(depth, normalized, 0.5, False)
    return {
        "integral_MeV_per_primary": float(np.trapz(dose, depth)),
        "peak_depth_mm": float(depth[int(np.argmax(dose))]),
        "peak_value_MeV_per_primary": float(np.max(dose)),
        "R80_mm": r80,
        "FWHM_mm": r50 - p50,
        "tail_ge_90mm_MeV_per_primary": float(
            np.trapz(dose[depth >= 90.0], depth[depth >= 90.0])
        ),
    }


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    ref_max = float(np.max(b))
    return {
        "max_abs_MeV_per_primary": float(np.max(np.abs(diff))),
        "mean_abs_MeV_per_primary": float(np.mean(np.abs(diff))),
        "nrmse_to_b_max": float(np.sqrt(np.mean(diff * diff)) / ref_max),
        "integral_signed_percent": 100.0
        * (float(np.sum(a)) - float(np.sum(b)))
        / float(np.sum(b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="histories=path pairs, e.g. 10000=out/n1e4.csv 100000=out/n1e5.csv",
    )
    parser.add_argument(
        "--repro",
        nargs=2,
        metavar=("PATH_A", "PATH_B"),
        default=None,
        help="Two same-seed runs for reproducibility",
    )
    parser.add_argument(
        "--reference-histories",
        type=int,
        default=None,
        help="Reference N (default: largest N)",
    )
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    series: list[tuple[int, Path, np.ndarray, np.ndarray]] = []
    for item in args.cases:
        if "=" not in item:
            raise SystemExit(f"Expected histories=path, got {item}")
        n_text, path_text = item.split("=", 1)
        n = int(float(n_text))
        path = Path(path_text)
        depth, dose = load_idd(path)
        series.append((n, path, depth, dose))
    series.sort(key=lambda row: row[0])

    ref_n = args.reference_histories if args.reference_histories is not None else series[-1][0]
    reference = next(row for row in series if row[0] == ref_n)
    ref_depth, ref_dose = reference[2], reference[3]
    ref_metrics = curve_metrics(ref_depth, ref_dose)

    cases_out: list[dict[str, object]] = []
    for n, path, depth, dose in series:
        metrics = curve_metrics(depth, dose)
        on_ref = np.interp(ref_depth, depth, dose)
        vs = compare(on_ref, ref_dose)
        cases_out.append(
            {
                "histories": n,
                "path": path.as_posix(),
                "is_reference": n == ref_n,
                "metrics": metrics,
                "vs_reference": vs,
                "delta_R80_mm": metrics["R80_mm"] - ref_metrics["R80_mm"],
                "delta_FWHM_mm": metrics["FWHM_mm"] - ref_metrics["FWHM_mm"],
                "delta_peak_depth_mm": metrics["peak_depth_mm"]
                - ref_metrics["peak_depth_mm"],
                "one_over_sqrt_N": 1.0 / math.sqrt(n),
            }
        )

    # Rough scaling check: NRMSE should decrease ~ 1/sqrt(N) when ref is largest N.
    scaling = None
    if len(cases_out) >= 2 and ref_n == series[-1][0]:
        pairs = []
        for case in cases_out:
            if case["is_reference"]:
                continue
            n = int(case["histories"])
            nrmse = float(case["vs_reference"]["nrmse_to_b_max"])  # type: ignore[index]
            pairs.append(
                {
                    "histories": n,
                    "nrmse_to_reference": nrmse,
                    "nrmse_times_sqrt_N": nrmse * math.sqrt(n),
                }
            )
        if pairs:
            values = [p["nrmse_times_sqrt_N"] for p in pairs if p["nrmse_to_reference"] > 0]
            scaling = {
                "pairs": pairs,
                "nrmse_times_sqrt_N_mean": float(np.mean(values)) if values else None,
                "nrmse_times_sqrt_N_std": float(np.std(values)) if values else None,
                "comment": (
                    "If Monte Carlo noise dominates vs the largest-N reference, "
                    "NRMSE * sqrt(N) should be roughly constant across smaller N."
                ),
            }

    repro = None
    if args.repro is not None:
        path_a, path_b = Path(args.repro[0]), Path(args.repro[1])
        depth_a, dose_a = load_idd(path_a)
        depth_b, dose_b = load_idd(path_b)
        if depth_a.shape != depth_b.shape or not np.allclose(depth_a, depth_b):
            raise SystemExit("Repro runs must share the same depth grid")
        bit_identical = bool(np.array_equal(dose_a, dose_b))
        repro = {
            "path_a": path_a.as_posix(),
            "path_b": path_b.as_posix(),
            "bit_identical": bit_identical,
            "compare": compare(dose_a, dose_b),
            "metrics_a": curve_metrics(depth_a, dose_a),
            "metrics_b": curve_metrics(depth_b, dose_b),
        }

    report = {
        "normalization": "absolute MeV/primary; no global scale",
        "neutron_deferred": True,
        "reference_histories": ref_n,
        "cases": cases_out,
        "noise_scaling": scaling,
        "reproducibility": repro,
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for n, path, depth, dose in series:
            axes[0].plot(depth, dose, label=f"N={n:g}", linewidth=1.2)
        axes[0].set_xlabel("Depth (mm)")
        axes[0].set_ylabel("MeV/primary")
        axes[0].set_title("IDD vs history count")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        ns = [int(c["histories"]) for c in cases_out if not c["is_reference"]]
        nrmse = [
            float(c["vs_reference"]["nrmse_to_b_max"])  # type: ignore[index]
            for c in cases_out
            if not c["is_reference"]
        ]
        if ns:
            axes[1].loglog(ns, nrmse, "o-", label="NRMSE to largest N")
            # guide line ~ 1/sqrt(N) anchored at first point
            guide = nrmse[0] * math.sqrt(ns[0]) / np.sqrt(np.asarray(ns, dtype=float))
            axes[1].loglog(ns, guide, "--", color="0.5", label=r"$\propto 1/\sqrt{N}$")
            axes[1].set_xlabel("histories")
            axes[1].set_ylabel("NRMSE")
            axes[1].set_title("Statistical convergence")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=140)
        plt.close(fig)

    print(f"Wrote {args.output_metrics}")
    if args.output_plot:
        print(f"Wrote {args.output_plot}")
    for case in cases_out:
        vs = case["vs_reference"]
        print(
            f"N={case['histories']}  NRMSE={vs['nrmse_to_b_max']:.4e}  "
            f"integralΔ={vs['integral_signed_percent']:+.4f}%  "
            f"ΔR80={case['delta_R80_mm']:+.4f} mm"
        )
    if repro is not None:
        print(
            f"Repro bit_identical={repro['bit_identical']}  "
            f"max|Δ|={repro['compare']['max_abs_MeV_per_primary']:.3e}  "
            f"NRMSE={repro['compare']['nrmse_to_b_max']:.3e}"
        )


if __name__ == "__main__":
    main()
