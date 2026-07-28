#!/usr/bin/env python3
"""Decompose minibeam TOPAS/GPU integral-dose differences by origin."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENERGIES_MEVU = (100, 200, 300, 400)
CHARGED = (
    ("primary_c12", "primary_c12_Gy"),
    ("secondary_carbon", "secondary_carbon_Gy"),
    ("boron", "boron_Gy"),
    ("beryllium", "beryllium_Gy"),
    ("lithium", "lithium_Gy"),
    ("helium", "helium_Gy"),
    ("proton", "proton_Gy"),
    ("other_charged", "other_Gy"),
)
NEUTRAL = ("neutron", "gamma", "neutral_other", "unresolved")


def load_topas(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=10_000)
    parser.add_argument(
        "--root", type=Path, default=Path("out/minibeam/energy_sweep")
    )
    parser.add_argument(
        "--energies",
        type=int,
        nargs="+",
        choices=ENERGIES_MEVU,
        default=list(ENERGIES_MEVU),
        help="Subset of prepared energies to analyze.",
    )
    args = parser.parse_args()
    records: list[dict[str, float | str]] = []
    contribution_rows: list[list[float]] = []
    labels = [name for name, _ in CHARGED] + ["neutral_origin"]

    for energy in args.energies:
        stem = f"minibeam_energy_e{energy}_{args.histories}"
        topas_dir = Path("ct/minibeam/output")
        gpu = np.genfromtxt(
            args.root / "gpu" / f"e{energy}" / "species_dose.csv",
            delimiter=",", names=True,
        )
        topas_total = float(
            np.sum(load_topas(topas_dir / f"{stem}_total.bin"))
        )
        gpu_total = float(np.sum(gpu["total_Gy"]))
        gpu_charged_sum = 0.0
        topas_charged_sum = 0.0
        contributions: list[float] = []
        for category, gpu_column in CHARGED:
            topas_value = float(
                np.sum(load_topas(topas_dir / f"{stem}_{category}.bin"))
            )
            gpu_value = float(np.sum(gpu[gpu_column]))
            gpu_charged_sum += gpu_value
            topas_charged_sum += topas_value
            contribution = 100.0 * (
                gpu_value - topas_value
            ) / topas_total
            contributions.append(contribution)
            records.append(
                {
                    "energy_MeVu": float(energy),
                    "category": category,
                    "topas_integral_Gy": topas_value,
                    "gpu_integral_Gy": gpu_value,
                    "category_relative_difference_percent": (
                        100.0 * (gpu_value / topas_value - 1.0)
                        if topas_value > 0.0 else float("nan")
                    ),
                    "contribution_to_total_difference_percent": contribution,
                }
            )
        topas_neutral = sum(
            float(np.sum(load_topas(topas_dir / f"{stem}_{category}.bin")))
            for category in NEUTRAL
        )
        gpu_neutral = gpu_total - gpu_charged_sum
        neutral_contribution = 100.0 * (
            gpu_neutral - topas_neutral
        ) / topas_total
        contributions.append(neutral_contribution)
        records.append(
            {
                "energy_MeVu": float(energy),
                "category": "neutral_origin",
                "topas_integral_Gy": topas_neutral,
                "gpu_integral_Gy": gpu_neutral,
                "category_relative_difference_percent": (
                    100.0 * (gpu_neutral / topas_neutral - 1.0)
                    if topas_neutral > 0.0 else float("nan")
                ),
                "contribution_to_total_difference_percent":
                    neutral_contribution,
            }
        )
        closure = sum(contributions)
        direct_total = 100.0 * (gpu_total / topas_total - 1.0)
        if abs(closure - direct_total) > 1.0e-6:
            raise RuntimeError(
                f"origin closure failed at {energy} MeV/u: "
                f"{closure} versus {direct_total}"
            )
        contribution_rows.append(contributions)

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / "origin_integrals.csv"
    json_path = args.root / "origin_integrals.json"
    plot_path = args.root / "origin_difference_contributions.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    matrix = np.asarray(contribution_rows, dtype=np.float64).T
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-12.0, vmax=12.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, f"{matrix[row, column]:+.1f}",
                ha="center", va="center", fontsize=8,
            )
    axis.set_xticks(range(len(args.energies)), args.energies)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Incident energy (MeV/u)")
    axis.set_title("Origin contribution to GPU−TOPAS total-dose error (%)")
    figure.colorbar(image, ax=axis, label="Contribution to total difference (%)")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(json.dumps(records, indent=2))
    print(csv_path)
    print(json_path)
    print(plot_path)


if __name__ == "__main__":
    main()
