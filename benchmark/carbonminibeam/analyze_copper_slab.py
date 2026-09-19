#!/usr/bin/env python3
"""Summarize homogeneous TOPAS Copper-slab exit phase space."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


C12_PDG = 1_000_060_120
CASE_PATTERN = re.compile(r"e(?P<energy>[0-9.]+)_t(?P<thickness>[0-9.]+)mm$")


def moments(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


def particle_z_a(pdg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = np.abs(pdg.astype(np.int64))
    ion = code > 1_000_000_000
    z = np.where(ion, (code // 10_000) % 1000, np.where(code == 2212, 1, 0))
    a = np.where(ion, (code // 10) % 1000, np.where(code == 2212, 1, 0))
    return z, a


def track_summary(rows: np.ndarray) -> dict:
    if rows.size == 0:
        return {"count": 0}
    ux = rows[:, 3]
    uy = rows[:, 4]
    uz_abs = np.sqrt(np.maximum(0.0, 1.0 - ux * ux - uy * uy))
    uz = np.where(rows[:, 8] != 0, -uz_abs, uz_abs)
    forward = np.maximum(1.0e-12, -uz)
    theta_x = np.arctan2(ux, forward)
    theta_y = np.arctan2(uy, forward)
    theta_r = np.hypot(theta_x, theta_y)
    x_mm = rows[:, 0] * 10.0
    y_mm = rows[:, 1] * 10.0
    centered_x = x_mm - np.mean(x_mm)
    centered_theta_x = theta_x - np.mean(theta_x)
    return {
        "count": int(rows.shape[0]),
        "energy_MeV": moments(rows[:, 5]),
        "x_mm": moments(x_mm),
        "y_mm": moments(y_mm),
        "theta_x_mrad": moments(theta_x * 1.0e3),
        "theta_y_mrad": moments(theta_y * 1.0e3),
        "theta_r_mrad_quantiles": {
            str(q): float(np.quantile(theta_r, q) * 1.0e3)
            for q in (0.68, 0.95, 0.99, 0.999)
        },
        "fermi_eyges_x": {
            "A0_x2_mm2": float(np.mean(centered_x * centered_x)),
            "A1_x_theta_mm_rad": float(
                np.mean(centered_x * centered_theta_x)),
            "A2_theta2_rad2": float(
                np.mean(centered_theta_x * centered_theta_x)),
        },
    }


def summarize_case(path: Path, histories: int) -> dict:
    phase_path = path / "output" / "slab_exit.phsp"
    rows = np.loadtxt(phase_path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    primary = rows[(rows[:, 7] == C12_PDG) & (rows[:, 13] == 0)]
    z, a = particle_z_a(rows[:, 7])
    charged_fragment = (z > 0) & ~((z == 6) & (a == 12) & (rows[:, 13] == 0))
    fragments = rows[charged_fragment]
    fragment_z = z[charged_fragment]
    fragment_a = a[charged_fragment]
    species = {}
    for species_z, species_a in sorted(set(zip(fragment_z, fragment_a))):
        selected = fragments[(fragment_z == species_z) & (fragment_a == species_a)]
        species[f"Z{species_z}A{species_a}"] = {
            "multiplicity_per_incident": float(selected.shape[0] / histories),
            **track_summary(selected),
        }
    return {
        "histories": histories,
        "primary_survival": float(primary.shape[0] / histories),
        "primary": track_summary(primary),
        "charged_fragment_multiplicity_per_incident": float(
            fragments.shape[0] / histories),
        "charged_fragments": species,
    }


def csda_exit_energy(incident_MeV: float, thickness_mm: float,
                     energies_MeVu: np.ndarray,
                     stopping_MeV_per_mm: np.ndarray) -> float:
    energy = incident_MeV
    steps = max(1, int(np.ceil(thickness_mm / 0.001)))
    step_mm = thickness_mm / steps
    for _ in range(steps):
        if energy <= 0.0:
            return 0.0
        stopping = np.interp(
            energy / 12.0, energies_MeVu, stopping_MeV_per_mm)
        energy = max(0.0, energy - stopping * step_mm)
    return energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument("--histories", type=int, default=256_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stopping-power-file", type=Path)
    args = parser.parse_args()

    stopping = None
    if args.stopping_power_file is not None:
        table = np.genfromtxt(
            args.stopping_power_file, delimiter=",", names=True)
        stopping = (
            np.asarray(table["energy_MeVu"], dtype=np.float64),
            np.asarray(table["stopping_power_MeV_per_mm"], dtype=np.float64),
        )

    result = {}
    for run_root in args.run_roots:
        for path in sorted(run_root.iterdir()):
            match = CASE_PATTERN.fullmatch(path.name)
            phase_path = path / "output" / "slab_exit.phsp"
            if match is None or not phase_path.exists() or not phase_path.stat().st_size:
                continue
            key = f"{match.group('energy')}MeVu_{match.group('thickness')}mm"
            summary = {
                "run_dir": str(path),
                "energy_MeV_per_u": float(match.group("energy")),
                "thickness_mm": float(match.group("thickness")),
                **summarize_case(path, args.histories),
            }
            if stopping is not None:
                csda = csda_exit_energy(
                    12.0 * summary["energy_MeV_per_u"],
                    summary["thickness_mm"], *stopping)
                summary["csda_exit_energy_MeV"] = csda
                primary_energy = summary["primary"].get("energy_MeV", {})
                if primary_energy.get("count", 0):
                    summary["primary_mean_minus_csda_MeV"] = (
                        primary_energy["mean"] - csda)
            result[key] = summary
    if not result:
        raise RuntimeError(f"No completed slab cases found under {args.run_roots}")
    output = args.output or args.run_roots[0] / "copper_slab_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
