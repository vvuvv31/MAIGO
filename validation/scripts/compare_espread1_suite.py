#!/usr/bin/env python3
"""Compare GPU vs TOPAS multi-energy suite with 1% RMS beam energy spread.

Absolute MeV/primary; no global dose scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = data.dtype.names or ()
    col = "energy_deposition_MeV_per_primary"
    if col not in names:
        raise SystemExit(f"{path} missing {col}; found {names}")
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data[col], dtype=float)),
    )


def r80(depth: np.ndarray, dose: np.ndarray) -> float:
    n = dose / np.max(dose)
    pk = int(np.argmax(n))
    for i in range(pk + 1, len(depth)):
        if n[i] <= 0.8 < n[i - 1]:
            return float(
                depth[i - 1]
                + (0.8 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
    return float("nan")


def fwhm(depth: np.ndarray, dose: np.ndarray) -> float:
    n = dose / np.max(dose)
    pk = int(np.argmax(n))
    r50 = float("nan")
    p50 = float("nan")
    for i in range(pk + 1, len(depth)):
        if n[i] <= 0.5 < n[i - 1]:
            r50 = float(
                depth[i - 1]
                + (0.5 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
            break
    for i in range(pk, 0, -1):
        if n[i - 1] <= 0.5 < n[i]:
            p50 = float(
                depth[i - 1]
                + (0.5 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
            break
    return r50 - p50


def metrics_pair(gpu_path: Path, topas_path: Path) -> dict[str, float]:
    gd, gv = load_idd(gpu_path)
    td, tv = load_idd(topas_path)
    on = np.interp(td, gd, gv)
    r_t = r80(td, tv)
    r_g = r80(gd, gv)
    nrmse = float(np.sqrt(np.mean((on - tv) ** 2)) / np.max(tv))
    # Dose-thresholded metrics avoid far-tail relative noise dominating judgment.
    thr = 0.05 * float(np.max(tv))
    mask = tv >= thr
    nrmse_5pct = (
        float(np.sqrt(np.mean((on[mask] - tv[mask]) ** 2)) / np.max(tv))
        if mask.any()
        else float("nan")
    )
    mid = (td >= 0.2 * r_t) & (td <= 0.7 * r_t) if np.isfinite(r_t) else mask
    mid_rel = (
        float(np.mean(100.0 * (on[mid] - tv[mid]) / tv[mid])) if mid.any() else float("nan")
    )
    peak_pct = 100.0 * (float(np.max(gv)) - float(np.max(tv))) / float(np.max(tv))
    int_pct = 100.0 * (float(np.sum(gv)) - float(np.sum(tv))) / float(np.sum(tv))
    fwhm_t = fwhm(td, tv)
    fwhm_g = fwhm(gd, gv)
    fwhm_pct = 100.0 * (fwhm_g - fwhm_t) / fwhm_t if fwhm_t > 0 else float("nan")
    ent_pct = 100.0 * (float(on[0]) - float(tv[0])) / float(tv[0]) if tv[0] > 0 else float("nan")
    return {
        "R80_topas_mm": r_t,
        "R80_gpu_mm": r_g,
        "delta_R80_mm": r_g - r_t,
        "FWHM_topas_mm": fwhm_t,
        "FWHM_gpu_mm": fwhm_g,
        "delta_FWHM_percent": fwhm_pct,
        "integral_diff_percent": int_pct,
        "peak_diff_percent": peak_pct,
        "entrance_diff_percent": ent_pct,
        "midplateau_mean_rel_percent": mid_rel,
        "NRMSE": nrmse,
        "NRMSE_dose_gt_5pct_peak": nrmse_5pct,
        "gpu_integral": float(np.sum(gv)),
        "topas_integral": float(np.sum(tv)),
        "gpu_peak": float(np.max(gv)),
        "topas_peak": float(np.max(tv)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-dir",
        type=Path,
        default=Path("validation/results/espread1"),
        help="Directory with gpu_e{E}_idd_100k.csv (name kept for compatibility)",
    )
    parser.add_argument(
        "--topas-dir",
        type=Path,
        default=Path("validation/results/espread1"),
        help="Directory with topas_e{E}_idd.csv",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 150, 200, 250, 300, 350, 400],
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("validation/results/espread1/summary.metrics.json"),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("validation/results/espread1/overview.png"),
    )
    args = parser.parse_args()

    cases: dict[int, dict] = {}
    available: dict[int, tuple[Path, Path]] = {}
    for e in args.energies:
        gp = args.gpu_dir / f"gpu_e{e}_idd_100k.csv"
        tp = args.topas_dir / f"topas_e{e}_idd.csv"
        # Also accept out/ paths
        if not gp.exists():
            alt = Path(f"out/multi_energy_espread1/e{e}_idd.csv")
            if alt.exists():
                gp = alt
        if not gp.exists() or not tp.exists():
            print(f"skip e={e}: gpu={gp.exists()} topas={tp.exists()}")
            continue
        available[e] = (gp, tp)
        cases[e] = metrics_pair(gp, tp)

    if not cases:
        raise SystemExit("No complete GPU+TOPAS pairs found for espread1 suite")

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "beam_energy_spread": 0.01,
        "beam_energy_spread_note": "1% RMS; TOPAS BeamEnergySpread=1.0 (percent)",
        "normalization": "absolute MeV/primary; no global scale",
        "histories": 100000,
        "cases": {str(e): m for e, m in sorted(cases.items())},
    }
    args.output_metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Overview IDD panel
    n = len(available)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for ax, (e, (gp, tp)) in zip(axes.flat, sorted(available.items())):
        gd, gv = load_idd(gp)
        td, tv = load_idd(tp)
        m = cases[e]
        ax.plot(td, tv, lw=2, label="TOPAS")
        ax.plot(gd, gv, lw=1.3, label="GPU")
        ax.set_title(
            f"{e} MeV/u  ΔR80={m['delta_R80_mm']:+.2f} mm\n"
            f"int {m['integral_diff_percent']:+.2f}%  NRMSE {m['NRMSE']:.3f}"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("MeV/primary/bin")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("GPU vs TOPAS — 1% RMS beam energy spread (absolute MeV/primary)")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    plt.close(fig)

    # Residual panel
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
    for ax, (e, (gp, tp)) in zip(axes.flat, sorted(available.items())):
        gd, gv = load_idd(gp)
        td, tv = load_idd(tp)
        on = np.interp(td, gd, gv)
        rel = 100.0 * (on - tv) / np.maximum(tv, 1e-12)
        thr = 0.05 * float(np.max(tv))
        rel_plot = np.where(tv >= thr, rel, np.nan)
        ax.plot(td, rel_plot, lw=1.2)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylim(-8, 8)
        ax.set_title(f"{e} MeV/u residual % (dose≥5% peak)")
        ax.set_xlabel("Depth (mm)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("(GPU-TOPAS)/TOPAS %")
    fig.suptitle("1% energy spread residuals")
    fig.tight_layout()
    res_path = args.output_plot.with_name(args.output_plot.stem + "_residual.png")
    fig.savefig(res_path, dpi=150)
    plt.close(fig)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output_metrics}")
    print(f"Wrote {args.output_plot}")
    print(f"Wrote {res_path}")


if __name__ == "__main__":
    main()
