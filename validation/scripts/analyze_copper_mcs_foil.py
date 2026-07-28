#!/usr/bin/env python3
"""Compare TOPAS C-12 Copper-foil scattering with the GPU Highland model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CARBON12_PDG = 1000060120
CARBON12_A = 12.0
CARBON12_Z = 6.0
CARBON12_REST_MEV = CARBON12_A * 931.49410242
COPPER_DENSITY_G_CM3 = 8.96
COPPER_MASS_RADIATION_LENGTH_G_CM2 = 12.8628


def stopping_power_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", comments="#", skiprows=3)
    return values[:, 0], values[:, 1]


def propagate_energy(
    initial_energy_mev: float,
    thickness_mm: float,
    energy_grid_mevu: np.ndarray,
    stopping_grid_mev_mm: np.ndarray,
    step_mm: float = 0.001,
) -> float:
    energy = initial_energy_mev
    remaining = thickness_mm
    while remaining > 0.0 and energy > 0.0:
        step = min(step_mm, remaining)
        stopping = float(
            np.interp(
                energy / CARBON12_A,
                energy_grid_mevu,
                stopping_grid_mev_mm,
                left=stopping_grid_mev_mm[0],
                right=stopping_grid_mev_mm[-1],
            )
        )
        energy = max(0.0, energy - stopping * step)
        remaining -= step
    return energy


def highland_sigma_rad(energy_mev: float, thickness_mm: float) -> float:
    total_energy = energy_mev + CARBON12_REST_MEV
    momentum = math.sqrt(max(0.0, total_energy**2 - CARBON12_REST_MEV**2))
    beta = momentum / total_energy
    radiation_lengths = (
        COPPER_DENSITY_G_CM3
        * (thickness_mm / 10.0)
        / COPPER_MASS_RADIATION_LENGTH_G_CM2
    )
    log_argument = radiation_lengths * CARBON12_Z**2 / beta**2
    correction = max(0.0, 1.0 + 0.038 * math.log(log_argument))
    return (
        13.6
        * CARBON12_Z
        / (beta * momentum)
        * math.sqrt(radiation_lengths)
        * correction
    )


def analyze_case(
    thickness_mm: float,
    phase_space_path: Path,
    energy_grid_mevu: np.ndarray,
    stopping_grid_mev_mm: np.ndarray,
) -> dict[str, float | int | str]:
    values = np.loadtxt(phase_space_path)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    primary = values[
        (values[:, 7].astype(np.int64) == CARBON12_PDG)
        & (values[:, 12].astype(np.int64) == 1)
        & (values[:, 13].astype(np.int64) == 0)
    ]
    if primary.size == 0:
        raise RuntimeError(f"no primary C-12 records in {phase_space_path}")

    direction_x = primary[:, 3]
    direction_y = primary[:, 4]
    sigma_x = float(np.std(direction_x, ddof=1))
    sigma_y = float(np.std(direction_y, ddof=1))
    rms_sigma = math.sqrt(0.5 * (sigma_x**2 + sigma_y**2))
    # For a Gaussian, 68.2689% of |x - median| lies within one sigma.
    core_x = float(np.quantile(np.abs(direction_x - np.median(direction_x)), 0.682689492))
    core_y = float(np.quantile(np.abs(direction_y - np.median(direction_y)), 0.682689492))
    core_sigma = math.sqrt(0.5 * (core_x**2 + core_y**2))

    midpoint_energy = propagate_energy(
        2150.0, 0.5 * thickness_mm, energy_grid_mevu, stopping_grid_mev_mm
    )
    predicted = highland_sigma_rad(midpoint_energy, thickness_mm)
    energies = primary[:, 5]
    return {
        "thickness_mm": thickness_mm,
        "phase_space": str(phase_space_path),
        "primary_c12_count": int(primary.shape[0]),
        "exit_energy_mean_MeV": float(np.mean(energies)),
        "exit_energy_std_MeV": float(np.std(energies, ddof=1)),
        "model_midpoint_energy_MeV": midpoint_energy,
        "topas_sigma_x_rad": sigma_x,
        "topas_sigma_y_rad": sigma_y,
        "topas_rms_sigma_rad": rms_sigma,
        "topas_core68_sigma_rad": core_sigma,
        "highland_sigma_rad": predicted,
        "topas_rms_over_highland": rms_sigma / predicted,
        "topas_core68_over_highland": core_sigma / predicted,
    }


def parse_case(value: str) -> tuple[float, Path]:
    try:
        thickness, path = value.split(":", 1)
        return float(thickness), Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be THICKNESS_MM:PHSP") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument(
        "--stopping-power",
        type=Path,
        default=Path("data/stopping_power_copper_geant4_11_3_2.csv"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    energy_grid, stopping_grid = stopping_power_table(args.stopping_power)
    cases = [
        analyze_case(thickness, path, energy_grid, stopping_grid)
        for thickness, path in args.case
    ]
    rms_ratios = np.asarray([case["topas_rms_over_highland"] for case in cases])
    core_ratios = np.asarray([case["topas_core68_over_highland"] for case in cases])
    result = {
        "description": (
            "Independent all-EM TOPAS 4.2.p3 / Geant4 11.3.2 Copper-foil "
            "benchmark for the GPU projected Highland scattering model."
        ),
        "incident_particle": "primary C-12",
        "incident_energy_MeV": 2150.0,
        "cases": cases,
        "summary": {
            "rms_scale_median": float(np.median(rms_ratios)),
            "rms_scale_min": float(np.min(rms_ratios)),
            "rms_scale_max": float(np.max(rms_ratios)),
            "core68_scale_median": float(np.median(core_ratios)),
            "core68_scale_min": float(np.min(core_ratios)),
            "core68_scale_max": float(np.max(core_ratios)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    print("thickness  E_exit(TOPAS)  sigma_TOPAS  sigma_Highland  RMS/H  core68/H")
    for case in cases:
        print(
            f"{case['thickness_mm']:8.3f}  "
            f"{case['exit_energy_mean_MeV']:13.3f}  "
            f"{case['topas_rms_sigma_rad']:11.7f}  "
            f"{case['highland_sigma_rad']:14.7f}  "
            f"{case['topas_rms_over_highland']:5.3f}  "
            f"{case['topas_core68_over_highland']:8.3f}"
        )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
