#!/usr/bin/env python3
"""Per-case water-exit EM and phase-space audit with approximate 95% CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


QUANTILES = (0.68, 0.95, 0.99, 0.999)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def variance_ci(values: np.ndarray) -> tuple[float, float, float]:
    centered = values - np.mean(values)
    variance = float(np.mean(centered * centered))
    fourth = float(np.mean(centered ** 4))
    sem = math.sqrt(max(0.0, fourth - variance * variance) / len(values))
    return variance, max(0.0, variance - 1.96 * sem), variance + 1.96 * sem


def covariance_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    products = (x - np.mean(x)) * (y - np.mean(y))
    return mean_ci(products)


def quantile_ci(values: np.ndarray, probability: float) -> tuple[float, float, float]:
    ordered = np.sort(values)
    count = len(ordered)
    center = probability * (count - 1)
    rank_sigma = math.sqrt(count * probability * (1.0 - probability))
    ranks = [int(np.clip(round(center + offset * 1.96 * rank_sigma), 0, count - 1))
             for offset in (0.0, -1.0, 1.0)]
    return tuple(float(ordered[index]) for index in ranks)


def summarize(energy_mevu: np.ndarray, displacement: np.ndarray,
              theta: np.ndarray, incident_mevu: float) -> dict:
    loss = incident_mevu - energy_mevu
    result = {
        "count": int(len(energy_mevu)),
        "energy_out_mean_MeVu": mean_ci(energy_mevu),
        "energy_loss_mean_MeVu": mean_ci(loss),
        "energy_out_std_MeVu": float(np.std(energy_mevu)),
        "theta_variance_rad2": variance_ci(theta),
        "displacement_variance_mm2": variance_ci(displacement),
        "displacement_theta_covariance_mm_rad": covariance_ci(
            displacement, theta),
    }
    absolute_theta = np.abs(theta)
    for probability in QUANTILES:
        result[f"abs_theta_q{probability:g}_rad"] = quantile_ci(
            absolute_theta, probability)
    return result


def read_topas(root: Path, case: dict) -> dict:
    rows = np.loadtxt(root / case["name"] / "output/slab_exit.phsp")
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    rows = rows[(rows[:, 7].astype(np.int64) == case["pdg"]) &
                (rows[:, 13].astype(np.int64) == 0)]
    dz = np.sqrt(np.maximum(0.0, 1.0 - rows[:, 3] ** 2 - rows[:, 4] ** 2))
    dz = np.where(rows[:, 8] != 0, -dz, dz)
    # The scorer surface is 0.010 mm downstream of the water. Back-project to
    # the water exit rather than comparing a vacuum drift with the GPU surface.
    vacuum_path_mm = 0.010 / np.maximum(np.abs(dz), 1.0e-12)
    x_exit = 10.0 * rows[:, 0] - rows[:, 3] * vacuum_path_mm
    y_exit = 10.0 * rows[:, 1] - rows[:, 4] * vacuum_path_mm
    theta_x = np.arctan2(rows[:, 3], -dz)
    theta_y = np.arctan2(rows[:, 4], -dz)
    return summarize(rows[:, 5] / case["mass_number"],
                     np.concatenate((x_exit, y_exit)),
                     np.concatenate((theta_x, theta_y)),
                     case["energy_MeVu"])


def read_gpu(gpu_root: Path, case: dict) -> dict:
    data = np.loadtxt(gpu_root / case["name"] / "plane.csv", delimiter=",",
                      skiprows=1, usecols=(0, 3, 4, 5, 6, 7, 8))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    histories = data[:, 0].astype(np.int64)
    x0 = ((histories % 500) - 249.5) * 0.2
    y0 = ((histories // 500) - 249.5) * 0.2
    displacement = np.concatenate((data[:, 2] - x0, data[:, 3] - y0))
    theta = np.concatenate((np.arctan2(data[:, 4], data[:, 6]),
                            np.arctan2(data[:, 5], data[:, 6])))
    return summarize(data[:, 1] / case["mass_number"], displacement, theta,
                     case["energy_MeVu"])


def ratio_interval(gpu: tuple[float, float, float],
                   topas: tuple[float, float, float]) -> tuple[float, float, float]:
    if topas[0] == 0.0:
        return float("nan"), float("nan"), float("nan")
    bounds = (gpu[1] / topas[2], gpu[2] / topas[1]) \
        if topas[1] > 0.0 else (float("nan"), float("nan"))
    return gpu[0] / topas[0], bounds[0], bounds[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    records = []
    scalar_keys = (
        "energy_out_mean_MeVu", "energy_loss_mean_MeVu",
        "theta_variance_rad2", "displacement_variance_mm2",
        "displacement_theta_covariance_mm_rad",
        "abs_theta_q0.68_rad", "abs_theta_q0.95_rad",
        "abs_theta_q0.99_rad", "abs_theta_q0.999_rad")
    for case in manifest["cases"]:
        topas = read_topas(args.run_root, case)
        gpu = read_gpu(args.gpu_root, case)
        ratios = {key: ratio_interval(gpu[key], topas[key])
                  for key in scalar_keys}
        ratios["energy_out_std_MeVu"] = (
            gpu["energy_out_std_MeVu"] / topas["energy_out_std_MeVu"])
        records.append({"case": case, "topas": topas, "gpu": gpu,
                        "ratios_gpu_topas": ratios})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n")
    rows = []
    for record in records:
        row = {"case": record["case"]["name"],
               "species": record["case"]["name"].split("_", 1)[0],
               "energy_MeVu": record["case"]["energy_MeVu"],
               "thickness_mm": record["case"]["thickness_mm"],
               "topas_count": record["topas"]["count"],
               "gpu_count": record["gpu"]["count"]}
        for key, value in record["ratios_gpu_topas"].items():
            if isinstance(value, tuple):
                row[key + "_ratio"] = value[0]
                row[key + "_ratio_ci_low"] = value[1]
                row[key + "_ratio_ci_high"] = value[2]
            else:
                row[key + "_ratio"] = value
        rows.append(row)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
