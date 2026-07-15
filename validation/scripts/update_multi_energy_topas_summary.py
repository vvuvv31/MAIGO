#!/usr/bin/env python3
"""Refresh multi-energy GPU-vs-TOPAS summary metrics and overview plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data["energy_deposition_MeV_per_primary"], dtype=float)),
    )


def r80_mm(depth: np.ndarray, dose: np.ndarray) -> float:
    normalized = dose / np.max(dose)
    peak = int(np.argmax(normalized))
    for index in range(peak + 1, len(depth)):
        if normalized[index] <= 0.8 < normalized[index - 1]:
            x0, x1 = depth[index - 1], depth[index]
            y0, y1 = normalized[index - 1], normalized[index]
            return float(x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def main() -> None:
    cases = {
        100: (
            Path("validation/results/windows_b580_multi_energy_100MeVu_idd_100k.csv"),
            Path("validation/results/topas_100MeVu_development.csv"),
            Path("validation/results/windows_b580_vs_topas_100MeVu.metrics.json"),
        ),
        200: (
            Path("validation/results/windows_b580_multi_energy_200MeVu_idd_100k.csv"),
            Path("validation/results/topas_200MeVu_development.csv"),
            None,
        ),
        300: (
            Path("validation/results/windows_b580_multi_energy_300MeVu_idd_100k.csv"),
            Path("validation/results/topas_300MeVu_development.csv"),
            Path("validation/results/windows_b580_vs_topas_300MeVu.metrics.json"),
        ),
        400: (
            Path("validation/results/windows_b580_multi_energy_400MeVu_idd_100k.csv"),
            Path("validation/results/topas_400MeVu_development.csv"),
            Path("validation/results/windows_b580_vs_topas_400MeVu.metrics.json"),
        ),
    }

    summary_cases: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (energy, (gpu_path, topas_path, metrics_path)) in zip(axes.ravel(), cases.items()):
        gpu_depth, gpu_dose = load(gpu_path)
        topas_depth, topas_dose = load(topas_path)
        on_topas = np.interp(topas_depth, gpu_depth, gpu_dose)
        gpu_int = float(np.sum(gpu_dose))
        topas_int = float(np.sum(topas_dose))
        int_pct = 100.0 * (gpu_int - topas_int) / topas_int

        if metrics_path is not None and metrics_path.exists():
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            differences = report["differences"]
            gamma = report.get("gamma", {})
            entry = {
                "E": energy,
                "gpu_int": gpu_int,
                "topas_int": topas_int,
                "int_pct": int_pct,
                "R80": differences["R80_mm"],
                "NRMSE": differences["NRMSE"],
                "FWHM": differences["FWHM_relative_percent"],
                "peak_pct": differences["peak_value_percent"],
                "tail": differences["tail_integral_relative_percent"],
                "g2": gamma.get("2pct_2mm_pass_rate_percent"),
            }
        else:
            topas_max = float(np.max(topas_dose))
            nrmse = float(np.sqrt(np.mean((on_topas - topas_dose) ** 2)) / topas_max)
            peak_pct = 100.0 * (float(np.max(gpu_dose)) - topas_max) / topas_max
            topas_r80 = r80_mm(topas_depth, topas_dose)
            gpu_r80 = r80_mm(gpu_depth, gpu_dose)
            mask = topas_depth >= topas_r80
            tail_topas = float(np.trapz(topas_dose[mask], topas_depth[mask]))
            tail_gpu = float(np.trapz(on_topas[mask], topas_depth[mask]))
            entry = {
                "E": energy,
                "gpu_int": gpu_int,
                "topas_int": topas_int,
                "int_pct": int_pct,
                "R80": gpu_r80 - topas_r80,
                "NRMSE": nrmse,
                "peak_pct": peak_pct,
                "tail": 100.0 * (tail_gpu - tail_topas) / tail_topas,
            }

        summary_cases.append(entry)
        axis.plot(topas_depth, topas_dose, label="TOPAS", linewidth=2)
        axis.plot(gpu_depth, gpu_dose, label="GPU", linewidth=1.4)
        tail = entry.get("tail", float("nan"))
        title = f"{energy} MeV/u  tail={tail:.1f}%  NRMSE={entry['NRMSE']:.3f}"
        axis.set_title(title)
        axis.set_xlabel("Depth (mm)")
        axis.set_ylabel("MeV/primary/bin")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("GPU vs TOPAS multi-energy after 400 MeV/u reaction packages", fontsize=12)
    fig.tight_layout()
    overview = Path("validation/results/windows_b580_vs_topas_multi_energy_overview.png")
    fig.savefig(overview, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for energy, (gpu_path, topas_path, _) in cases.items():
        gpu_depth, gpu_dose = load(gpu_path)
        topas_depth, topas_dose = load(topas_path)
        axis.plot(topas_depth, topas_dose, linewidth=2, label=f"TOPAS {energy}")
        axis.plot(gpu_depth, gpu_dose, linewidth=1.2, linestyle="--", label=f"GPU {energy}")
    axis.set(
        xlabel="Depth (mm)",
        ylabel="MeV/primary/bin",
        title="Multi-energy IDD: TOPAS solid, GPU dashed",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    curves = Path("validation/results/windows_b580_vs_topas_multi_energy_curves.png")
    fig.savefig(curves, dpi=160)
    plt.close(fig)

    summary = {
        "normalization": "absolute MeV/primary; no global scale",
        "histories_topas": 100000,
        "histories_gpu": 100000,
        "topas_version_remote": "4.1.p1",
        "geant4_version_remote": "geant4-11-01-patch-03",
        "reaction_packages": {
            "100_200_MeVu": (
                "validation/results/topas_200MeVu_cascade_aligned_primary_3d.bin "
                "(0-200 MeV/u, 201 bins)"
            ),
            "300_400_MeVu": (
                "validation/results/topas_400MeVu_cascade_aligned_primary_3d.bin "
                "(0-400 MeV/u, 401 bins)"
            ),
            "cascade_300_400": "validation/results/topas_400MeVu_cascade_100k_3d.bin",
            "primary_400_stats": {
                "reactions": 73739,
                "secondaries": 715976,
                "energy_bins": 401,
                "min_packages_per_bin": 4,
                "max_packages_per_bin": 521,
            },
        },
        "cases": summary_cases,
        "improvement_vs_200package_ceiling": {
            "300_tail_pct": {"before": -20.82, "after": summary_cases[2].get("tail")},
            "400_tail_pct": {"before": -38.36, "after": summary_cases[3].get("tail")},
            "400_g2_pct": {"before": 65.38, "after": summary_cases[3].get("g2")},
            "400_NRMSE": {"before": 0.03528, "after": summary_cases[3].get("NRMSE")},
        },
        "neutral_local_kerma_fraction": 0.298,
        "neutral_kerma_calibration": (
            "TOPAS 200 MeV/u neutral-origin dose 16.35 MeV/primary / "
            "GPU untransported n/γ birth KE ~54.9 MeV/primary"
        ),
        "summary": (
            "400 MeV/u cascade packages fix high-E final-state ceiling; interim "
            "neutral_local_kerma_fraction=0.298 (calibrated at 200 MeV/u) restores "
            "plateau/integral for 300/400. Remaining peak-height bias is charged-path "
            "physics, not the previous package energy ceiling. Full neutral transport "
            "should eventually replace local kerma."
        ),
    }
    out = Path("validation/results/windows_b580_vs_topas_multi_energy_summary.metrics.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {overview}")
    print(f"Wrote {curves}")


if __name__ == "__main__":
    main()
