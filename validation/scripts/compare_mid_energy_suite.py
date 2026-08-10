#!/usr/bin/env python3
"""Compare GPU vs TOPAS for mid energies 150/250/350 and full 100--400 suite."""

from __future__ import annotations

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
    pk = int(np.argmax(n))
    for i in range(pk + 1, len(depth)):
        if n[i] <= 0.8 < n[i - 1]:
            return float(
                depth[i - 1]
                + (0.8 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
    return float("nan")


def metrics_pair(gpu_path: Path, topas_path: Path) -> dict[str, float]:
    gd, gv = load_idd(gpu_path)
    td, tv = load_idd(topas_path)
    on = np.interp(td, gd, gv)
    r = r80(td, tv)
    nrmse = float(np.sqrt(np.mean((on - tv) ** 2)) / np.max(tv))
    peak_pct = 100.0 * (float(np.max(gv)) - float(np.max(tv))) / float(np.max(tv))
    int_pct = 100.0 * (float(np.sum(gv)) - float(np.sum(tv))) / float(np.sum(tv))
    mask = td >= r
    tail_t = float(np.trapz(tv[mask], td[mask])) if np.isfinite(r) else float("nan")
    tail_g = float(np.trapz(on[mask], td[mask])) if np.isfinite(r) else float("nan")
    mid = (td >= 0.2 * r) & (td <= 0.7 * r)
    mid_rel = float(np.mean(100.0 * (on[mid] - tv[mid]) / tv[mid])) if mid.any() else float("nan")
    return {
        "R80_topas_mm": r,
        "R80_gpu_mm": r80(gd, gv),
        "delta_R80_mm": r80(gd, gv) - r,
        "integral_diff_percent": int_pct,
        "peak_diff_percent": peak_pct,
        "NRMSE": nrmse,
        "tail_integral_diff_percent": 100.0 * (tail_g - tail_t) / tail_t if tail_t else float("nan"),
        "midplateau_mean_rel_percent": mid_rel,
        "gpu_integral": float(np.sum(gv)),
        "topas_integral": float(np.sum(tv)),
    }


def main() -> None:
    mid = {
        150: (
            Path("validation/results/windows_b580_multi_energy_150MeVu_idd_100k.csv"),
            Path("validation/results/topas_150MeVu_development.csv"),
        ),
        250: (
            Path("validation/results/windows_b580_multi_energy_250MeVu_idd_100k.csv"),
            Path("validation/results/topas_250MeVu_development.csv"),
        ),
        350: (
            Path("validation/results/windows_b580_multi_energy_350MeVu_idd_100k.csv"),
            Path("validation/results/topas_350MeVu_development.csv"),
        ),
    }
    full = {
        100: (
            Path("validation/results/windows_b580_multi_energy_100MeVu_idd_100k.csv"),
            Path("validation/results/topas_100MeVu_development.csv"),
        ),
        150: mid[150],
        200: (
            Path("validation/results/windows_b580_multi_energy_200MeVu_idd_100k.csv"),
            Path("validation/results/topas_200MeVu_development.csv"),
        ),
        250: mid[250],
        300: (
            Path("validation/results/windows_b580_multi_energy_300MeVu_idd_100k.csv"),
            Path("validation/results/topas_300MeVu_development.csv"),
        ),
        350: mid[350],
        400: (
            Path("validation/results/windows_b580_multi_energy_400MeVu_idd_100k.csv"),
            Path("validation/results/topas_400MeVu_development.csv"),
        ),
    }

    cases = {}
    for e, (gp, tp) in mid.items():
        if not gp.exists() or not tp.exists():
            raise SystemExit(f"Missing inputs for {e}: {gp.exists()=} {tp.exists()=}")
        cases[e] = metrics_pair(gp, tp)

    # mid-energy comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (e, (gp, tp)) in zip(axes, mid.items()):
        gd, gv = load_idd(gp)
        td, tv = load_idd(tp)
        m = cases[e]
        ax.plot(td, tv, lw=2, label="TOPAS")
        ax.plot(gd, gv, lw=1.4, label="GPU")
        ax.set_title(
            f"{e} MeV/u\nint {m['integral_diff_percent']:+.2f}%  "
            f"mid {m['midplateau_mean_rel_percent']:+.1f}%  NRMSE {m['NRMSE']:.3f}"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("MeV/primary/bin")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("GPU vs TOPAS mid-energy suite (150 / 250 / 350 MeV/u)")
    fig.tight_layout()
    out_mid = Path("validation/results/windows_b580_vs_topas_mid_energy_overview.png")
    fig.savefig(out_mid, dpi=160)
    plt.close(fig)

    # residual panel
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (e, (gp, tp)) in zip(axes, mid.items()):
        gd, gv = load_idd(gp)
        td, tv = load_idd(tp)
        on = np.interp(td, gd, gv)
        rel = 100.0 * (on - tv) / np.maximum(tv, 1e-12)
        ax.plot(td, rel, lw=1.3)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylim(-12, 12)
        ax.set_title(f"{e} MeV/u residual %")
        ax.set_xlabel("Depth (mm)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("(GPU-TOPAS)/TOPAS %")
    fig.tight_layout()
    out_res = Path("validation/results/windows_b580_vs_topas_mid_energy_residual.png")
    fig.savefig(out_res, dpi=160)
    plt.close(fig)

    # full suite curves if all available
    available = {e: p for e, p in full.items() if p[0].exists() and p[1].exists()}
    full_metrics = {e: metrics_pair(*p) for e, p in available.items()}
    if len(available) >= 5:
        fig, ax = plt.subplots(figsize=(11, 6))
        for e, (gp, tp) in sorted(available.items()):
            gd, gv = load_idd(gp)
            td, tv = load_idd(tp)
            ax.plot(td, tv, lw=2, label=f"TOPAS {e}")
            ax.plot(gd, gv, lw=1.1, ls="--", label=f"GPU {e}")
        ax.set(
            xlabel="Depth (mm)",
            ylabel="MeV/primary/bin",
            title="Multi-energy IDD suite: TOPAS solid, GPU dashed",
        )
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        out_full = Path("validation/results/windows_b580_vs_topas_energy_suite_curves.png")
        fig.savefig(out_full, dpi=160)
        plt.close(fig)
    else:
        out_full = None

    report = {
        "histories": 100000,
        "neutral_local_kerma_fraction": 0.298,
        "mid_energy_cases": {str(k): v for k, v in cases.items()},
        "full_suite_cases": {str(k): v for k, v in full_metrics.items()},
        "plots": {
            "mid_overview": out_mid.as_posix(),
            "mid_residual": out_res.as_posix(),
            "full_curves": out_full.as_posix() if out_full else None,
        },
    }
    out_json = Path("validation/results/windows_b580_vs_topas_mid_energy_summary.metrics.json")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_mid}")
    print(f"Wrote {out_res}")
    if out_full:
        print(f"Wrote {out_full}")


if __name__ == "__main__":
    main()
