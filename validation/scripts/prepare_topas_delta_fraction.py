#!/usr/bin/env python3
"""Convert TOPAS step-born electron energy into an energy-dependent LET correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_topas(path: Path) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("#") or not line.strip():
                continue
            values.append(float(line.split(",")[3]))
    return np.asarray(values)


def read_stopping_power(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    return (
        np.asarray([float(row["energy_MeVu"]) for row in rows]),
        np.asarray([float(row["stopping_power_MeV_per_mm"]) for row in rows]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--electron",
        type=Path,
        default=Path(
            "validation/topas/output/delta_reference_primary_c12_electron.csv"
        ),
    )
    parser.add_argument(
        "--energy-deposit",
        type=Path,
        default=Path(
            "validation/topas/output/delta_reference_primary_c12_energy_deposit.csv"
        ),
    )
    parser.add_argument(
        "--stopping-power",
        type=Path,
        default=Path("data/stopping_power_water_geant4_11_3_2.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/let_delta_electron_fraction_water_geant4_11_3_2.csv"
        ),
    )
    parser.add_argument("--initial-energy-mevu", type=float, default=200.0)
    parser.add_argument("--depth-bin-mm", type=float, default=0.5)
    parser.add_argument("--mass-number", type=float, default=12.0)
    parser.add_argument("--histories", type=int, default=10_000)
    args = parser.parse_args()

    electron = read_topas(args.electron)
    deposited = read_topas(args.energy_deposit)
    sp_energy, sp_value = read_stopping_power(args.stopping_power)
    if electron.shape != deposited.shape:
        raise SystemExit("TOPAS electron and deposited-energy grids differ")

    # Reconstruct mean primary energy at each depth-bin center using the same
    # unrestricted electronic stopping table used by the GPU transport.
    energy = args.initial_energy_mevu
    bin_energy = np.zeros_like(deposited)
    for index in range(len(bin_energy)):
        stopping = np.interp(energy, sp_energy, sp_value)
        half_loss_mevu = 0.5 * stopping * args.depth_bin_mm / args.mass_number
        bin_energy[index] = max(1.0, energy - half_loss_mevu)
        energy = max(1.0, energy - 2.0 * half_loss_mevu)

    # Smooth the two extensive quantities before division. This is less biased
    # than smoothing the noisy ratio, especially near the Bragg peak.
    kernel = np.ones(9)
    smooth_electron = np.convolve(electron, kernel, mode="same")
    smooth_total = np.convolve(electron + deposited, kernel, mode="same")
    fraction = np.divide(
        smooth_electron,
        smooth_total,
        out=np.zeros_like(smooth_electron),
        where=smooth_total > 0,
    )
    valid = (
        (deposited > 0.01 * deposited.max())
        & (np.arange(len(deposited)) <= int(np.argmax(deposited)))
        & (bin_energy > 1.0)
    )
    sampled_energy = bin_energy[valid][::-1]
    sampled_fraction = fraction[valid][::-1]
    # The delta-ray share should be nondecreasing with projectile velocity.
    sampled_fraction = np.maximum.accumulate(sampled_fraction)

    # Keep the LET correction on exactly the same grid as the runtime stopping
    # table. This includes the sub-1 MeV/u Bragg-tail extension.
    output_energy = sp_energy
    output_fraction = np.interp(
        output_energy,
        sampled_energy,
        sampled_fraction,
        left=max(1.0e-6, sampled_fraction[0]),
        right=sampled_fraction[-1],
    )
    # Extrapolate 200--400 MeV/u from the high-energy slope, conservatively
    # capped. The current 200 MeV/u validation uses only the measured interval.
    high = sampled_energy >= max(120.0, sampled_energy.max() - 60.0)
    if high.sum() >= 2:
        slope, intercept = np.polyfit(
            sampled_energy[high], sampled_fraction[high], 1
        )
        above = output_energy > sampled_energy.max()
        output_fraction[above] = np.clip(
            slope * output_energy[above] + intercept,
            sampled_fraction[-1],
            0.25,
        )
    output_fraction = np.clip(output_fraction, 1.0e-6, 0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        stream.write(
            "# Fraction E_delta/(Edep+E_delta), measured with TOPAS "
            "CarbonDeltaElectronEnergy in Water_75eV at 0.05 mm production cut.\n"
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["energy_MeVu", "delta_electron_fraction"])
        for energy_mevu, value in zip(output_energy, output_fraction):
            writer.writerow([f"{energy_mevu:.12g}", f"{value:.12g}"])

    metadata = {
        "quantity": "step-born electron kinetic-energy fraction",
        "definition": "E_delta / (Edep + E_delta)",
        "topas": "4.2.p3",
        "geant4": "11.3.2",
        "material": "Water_75eV",
        "production_cut_mm": 0.05,
        "histories": args.histories,
        "measured_energy_interval_MeVu": [
            float(sampled_energy.min()),
            float(sampled_energy.max()),
        ],
        "fraction_at_200_MeVu": float(np.interp(200.0, output_energy, output_fraction)),
        "above_measured_interval": "linear high-energy extrapolation capped at 0.25",
        "energy_grid_MeVu": {
            "minimum": float(output_energy.min()),
            "maximum": float(output_energy.max()),
            "count": int(output_energy.size),
        },
        "inputs": {
            "electron": str(args.electron),
            "energy_deposit": str(args.energy_deposit),
            "stopping_power": str(args.stopping_power),
        },
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
