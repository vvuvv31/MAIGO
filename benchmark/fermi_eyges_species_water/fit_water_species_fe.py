#!/usr/bin/env python3
"""Fit the shared FE core+Poisson-tail form to pure-water TOPAS slabs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, gammaln


NUCLEON_REST_MEV = 931.49410242
WATER_X0_MM = 360.8297746402233
QUANTILES = np.asarray([0.68, 0.95, 0.99, 0.999])


def beta_momentum(energy_mevu: float, mass_number: int) -> tuple[float, float]:
    total_u = energy_mevu + NUCLEON_REST_MEV
    momentum_u = math.sqrt(energy_mevu * (energy_mevu + 2.0 * NUCLEON_REST_MEV))
    return momentum_u / total_u, mass_number * momentum_u


def projected_abs_cdf(value: float, core_energy: float, tail_rate: float,
                      tail_energy: float, q2: float, length_mm: float) -> float:
    mean = tail_rate * length_mm
    total = 0.0
    # The fitted means are small; retaining through n=12 is far beyond machine
    # relevance while keeping the expression deterministic.
    for count in range(13):
        log_probability = -mean + count * math.log(max(mean, 1.0e-300)) - gammaln(count + 1)
        variance = q2 * (core_energy * core_energy * length_mm / WATER_X0_MM +
                         count * tail_energy * tail_energy)
        if variance > 0.0:
            total += math.exp(log_probability) * erf(value / math.sqrt(2.0 * variance))
    return min(1.0, max(0.0, total))


def read_case(run_root: Path, case: dict) -> dict:
    phase = run_root / case["name"] / "output/slab_exit.phsp"
    rows = np.loadtxt(phase, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    selected = rows[(rows[:, 7].astype(np.int64) == case["pdg"]) &
                    (rows[:, 13].astype(np.int64) == 0)]
    if not len(selected):
        raise RuntimeError(f"No primary rows in {phase}")
    direction_z = np.sqrt(np.maximum(0.0, 1.0 - selected[:, 3] ** 2 - selected[:, 4] ** 2))
    direction_z = np.where(selected[:, 8] != 0, -direction_z, direction_z)
    theta_x = np.arctan2(selected[:, 3], -direction_z)
    theta_y = np.arctan2(selected[:, 4], -direction_z)
    projected_abs = np.abs(np.concatenate((theta_x, theta_y)))
    transverse_mm = 10.0 * np.concatenate((selected[:, 0], selected[:, 1]))
    energy_out_mevu = selected[:, 5] / case["mass_number"]
    beta_in, momentum_in = beta_momentum(case["energy_MeVu"], case["mass_number"])
    beta_out, momentum_out = beta_momentum(float(np.mean(energy_out_mevu)), case["mass_number"])
    q2_in = (case["atomic_number"] / (beta_in * momentum_in)) ** 2
    q2_out = (case["atomic_number"] / (beta_out * momentum_out)) ** 2
    return {
        "name": case["name"],
        "count": int(len(selected)),
        "survival": float(len(selected) / case["histories"]),
        "energy_out_mean_MeVu": float(np.mean(energy_out_mevu)),
        "energy_out_std_MeVu": float(np.std(energy_out_mevu)),
        "q2_effective": 0.5 * (q2_in + q2_out),
        "abs_theta_quantiles_rad": np.quantile(projected_abs, QUANTILES).tolist(),
        "theta_projected_variance_rad2": float(np.var(np.concatenate((theta_x, theta_y)))),
        "transverse_variance_mm2": float(np.var(transverse_mm)),
    }


def fit_species(records: list[dict], cases: dict[str, dict]) -> dict:
    def residual(log_parameters: np.ndarray) -> np.ndarray:
        core, rate, tail = np.exp(log_parameters)
        values = []
        for record in records:
            length = cases[record["name"]]["thickness_mm"]
            for target, observed in zip(QUANTILES, record["abs_theta_quantiles_rad"]):
                model = projected_abs_cdf(
                    observed, core, rate, tail,
                    record["q2_effective"], length)
                # Tail quantiles receive comparable leverage to the core in
                # probability space without fitting a depth-dose observable.
                scale = max(1.0e-4, 1.0 - target)
                values.append((model - target) / scale)
        return np.asarray(values)

    starts = (
        (9.9, 0.0025, 2.4),
        (13.6, 0.001, 5.0),
        (8.0, 0.01, 2.0),
        (12.0, 0.0005, 12.0),
    )
    solutions = []
    for start in starts:
        solutions.append(least_squares(
            residual, np.log(start),
            bounds=(np.log((2.0, 1.0e-6, 0.1)),
                    np.log((30.0, 0.2, 100.0))),
            max_nfev=3000))
    best = min(solutions, key=lambda item: float(np.dot(item.fun, item.fun)))
    core, rate, tail = np.exp(best.x)
    return {
        "core_scattering_energy_MeV": float(core),
        "tail_rate_per_water_mm": float(rate),
        "tail_scattering_energy_MeV": float(tail),
        "cost": float(best.cost),
        "residual_rms": float(np.sqrt(np.mean(best.fun ** 2))),
        "fit_success": bool(best.success),
        "fit_message": best.message,
        "datasets": [record["name"] for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=Path(
                            "/mnt/sda/wuwei/fe_species_water_em10gev_20260919"))
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    cases = {case["name"]: case for case in manifest["cases"]}
    records = [read_case(args.run_root, case) for case in manifest["cases"]]
    by_species: dict[str, list[dict]] = {}
    for record in records:
        tag = record["name"].split("_", 1)[0]
        by_species.setdefault(tag, []).append(record)
    fits = {tag: fit_species(group, cases) for tag, group in by_species.items()}
    fits_by_energy = {}
    for tag, group in by_species.items():
        fits_by_energy[tag] = {}
        for energy in sorted({cases[record["name"]]["energy_MeVu"]
                              for record in group}):
            selected = [record for record in group
                        if cases[record["name"]]["energy_MeVu"] == energy]
            fits_by_energy[tag][str(energy)] = fit_species(selected, cases)
    result = {
        "model": "projected Gaussian FE core plus Poisson Gaussian tail",
        "water_radiation_length_mm": WATER_X0_MM,
        "quantiles": QUANTILES.tolist(),
        "fits": fits,
        "fits_by_energy_MeVu": fits_by_energy,
        "records": records,
    }
    output = args.run_root / "water_species_fe_fit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
