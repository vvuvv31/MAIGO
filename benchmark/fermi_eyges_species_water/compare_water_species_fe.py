#!/usr/bin/env python3
"""Compare calibrated GPU FE slab phase space with the TOPAS references."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path("/mnt/sda/wuwei/fe_species_water_em10gev_20260919")
QUANTILES = np.asarray([0.68, 0.95, 0.99, 0.999])


def gpu_metrics(path: Path, mass_number: int) -> dict:
    # source_history, energy, x, y, dx, dy, dz
    data = np.loadtxt(path, delimiter=",", skiprows=1,
                      usecols=(0, 3, 4, 5, 6, 7, 8))
    histories = data[:, 0].astype(np.int64)
    x0 = ((histories % 500) - 249.5) * 0.2
    y0 = ((histories // 500) - 249.5) * 0.2
    displacement = np.concatenate((data[:, 2] - x0, data[:, 3] - y0))
    theta_x = np.arctan2(data[:, 4], data[:, 6])
    theta_y = np.arctan2(data[:, 5], data[:, 6])
    projected = np.concatenate((theta_x, theta_y))
    energy_mevu = data[:, 1] / mass_number
    return {
        "count": int(len(data)),
        "energy_out_mean_MeVu": float(np.mean(energy_mevu)),
        "energy_out_std_MeVu": float(np.std(energy_mevu)),
        "abs_theta_quantiles_rad": np.quantile(np.abs(projected), QUANTILES).tolist(),
        "theta_projected_variance_rad2": float(np.var(projected)),
        "transverse_variance_mm2": float(np.var(displacement)),
    }


def ratio(gpu: float, topas: float) -> float:
    return gpu / topas if topas else float("nan")


def main() -> None:
    manifest = json.loads((RUN_ROOT / "manifest.json").read_text())
    fit = json.loads((RUN_ROOT / "water_species_fe_fit.json").read_text())
    topas = {record["name"]: record for record in fit["records"]}
    rows = []
    details = []
    for case in manifest["cases"]:
        name = case["name"]
        gpu = gpu_metrics(RUN_ROOT / "gpu" / name / "plane.csv",
                          case["mass_number"])
        reference = topas[name]
        quantile_ratios = [ratio(g, t) for g, t in zip(
            gpu["abs_theta_quantiles_rad"],
            reference["abs_theta_quantiles_rad"])]
        row = {
            "case": name,
            "species": name.split("_", 1)[0],
            "energy_MeVu": case["energy_MeVu"],
            "thickness_mm": case["thickness_mm"],
            "gpu_count": gpu["count"],
            "topas_count": reference["count"],
            "energy_mean_ratio": ratio(
                gpu["energy_out_mean_MeVu"], reference["energy_out_mean_MeVu"]),
            "theta_variance_ratio": ratio(
                gpu["theta_projected_variance_rad2"],
                reference["theta_projected_variance_rad2"]),
            "displacement_variance_ratio": ratio(
                gpu["transverse_variance_mm2"],
                reference["transverse_variance_mm2"]),
            "q68_ratio": quantile_ratios[0],
            "q95_ratio": quantile_ratios[1],
            "q99_ratio": quantile_ratios[2],
            "q999_ratio": quantile_ratios[3],
        }
        rows.append(row)
        details.append({"case": case, "topas": reference, "gpu": gpu,
                        "ratios": row})
    csv_path = RUN_ROOT / "water_species_fe_comparison.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (RUN_ROOT / "water_species_fe_comparison.json").write_text(
        json.dumps(details, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = (("q68_ratio", "q68"), ("q99_ratio", "q99"),
               ("q999_ratio", "q99.9"),
               ("displacement_variance_ratio", "Var(displacement)"))
    markers = {1: "o", 10: "s"}
    for axis, (key, label) in zip(axes.ravel(), metrics):
        for species in ("p", "d", "t", "he4"):
            for thickness in (1, 10):
                selected = [row for row in rows if row["species"] == species and
                            row["thickness_mm"] == thickness]
                axis.plot([row["energy_MeVu"] for row in selected],
                          [row[key] for row in selected],
                          marker=markers[thickness], label=f"{species} {thickness}mm")
        axis.axhline(1.0, color="black", linewidth=0.7)
        axis.set_title(f"GPU/TOPAS {label}")
        axis.set_xlabel("Energy (MeV/u)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_ROOT / "water_species_fe_ratios.png", dpi=170)
    plt.close(fig)
    print(csv_path)


if __name__ == "__main__":
    main()
