#!/usr/bin/env python3
"""Prepare matched primary-C12 plane replay inputs for GPU and TOPAS.

TOPAS uses +Y as depth in this benchmark and MAIGO uses +Z. Transverse
coordinates/directions are mapped accordingly. The GPU axial source position
is explicit: water-entry replay defaults just inside water, while an upstream
plane supplies --gpu-source-z-mm. The script rejects non-unit weights because
the current GPU primary batch has no per-history statistical-weight field.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


C12_PDG = 1000060120


def original_histories(header: Path) -> int:
    match = re.search(
        r"^Number of Original Histories:\s*(\d+)\s*$",
        header.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing original-history count in {header}")
    return int(match.group(1))


def write_topas_header(path: Path, histories: int, rows: np.ndarray) -> None:
    energies = rows[:, 5]
    text = f"""TOPAS ASCII Phase Space

Number of Original Histories: {histories}
Number of Original Histories that Reached Phase Space: {rows.shape[0]}
Number of Scored Particles: {rows.shape[0]}

Columns of data are as follows:
 1: Position X [cm]
 2: Position Y [cm]
 3: Position Z [cm]
 4: Direction Cosine X
 5: Direction Cosine Y
 6: Energy [MeV]
 7: Weight
 8: Particle Type (in PDG Format)
 9: Flag to tell if Third Direction Cosine is Negative (1 means true)
10: Flag to tell if this is the First Scored Particle from this History (1 means true)
11: Run ID
12: Event ID
13: Track ID
14: Parent ID

Number of C12: {rows.shape[0]}
Minimum Kinetic Energy of C12: {energies.min():.12g} MeV
Maximum Kinetic Energy of C12: {energies.max():.12g} MeV
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_space", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--water-entrance-mm", type=float, default=60.0)
    parser.add_argument("--gpu-entry-epsilon-mm", type=float, default=1.0e-4)
    parser.add_argument("--gpu-source-z-mm", type=float)
    parser.add_argument("--plane-name", default="water_entrance")
    args = parser.parse_args()

    header = args.header or args.phase_space.with_suffix(".header")
    histories = original_histories(header)
    data = np.loadtxt(args.phase_space, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 14:
        raise ValueError("TOPAS phase space must contain the 14 diagnostic columns")
    rows = data[(data[:, 7] == C12_PDG) & (data[:, 13] == 0)].copy()
    if rows.shape[0] == 0:
        raise ValueError("No parent-0 C12 particles found")
    if not np.allclose(rows[:, 6], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("GPU replay currently requires unit phase-space weights")

    # A filtered primary is one independent replay history even when another
    # particle happened to be written first in the original mixed-species file.
    rows[:, 9] = 1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topas_base = args.output_dir / f"{args.plane_name}_primary_c12"
    np.savetxt(topas_base.with_suffix(".phsp"), rows, fmt="%.12g")
    write_topas_header(
        topas_base.with_suffix(".header"), histories, rows)

    dx = rows[:, 3]
    dy_topas = rows[:, 4]
    dz_topas = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy_topas * dy_topas))
    dz_topas = np.where(rows[:, 8] != 0, -dz_topas, dz_topas)
    gpu_csv = args.output_dir / f"{args.plane_name}_primary_c12_gpu.csv"
    gpu_source_z_mm = (args.gpu_entry_epsilon_mm if args.gpu_source_z_mm is None
                       else args.gpu_source_z_mm)
    with gpu_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "spot_id", "energy_MeV", "x_mm", "y_mm", "weight",
            "source_x_mm", "source_y_mm", "source_z_mm",
            "direction_x", "direction_y", "direction_z",
            "energy_spread_percent", "sigma_x_mm", "sigma_y_mm",
            "sigma_x_prime", "sigma_y_prime", "correlation_x",
            "correlation_y"])
        for index, row in enumerate(rows):
            writer.writerow([
                index + 1, f"{row[5]:.12g}", 0, 0, 1,
                f"{row[0] * 10.0:.12g}", f"{row[2] * 10.0:.12g}",
                f"{gpu_source_z_mm:.12g}",
                f"{dx[index]:.12g}", f"{dz_topas[index]:.12g}",
                f"{dy_topas[index]:.12g}", 0, 0, 0, 0, 0, 0, 0])

    metadata = {
        "source_phase_space": str(args.phase_space),
        "source_header": str(header),
        "original_histories": histories,
        "replayed_primary_c12": int(rows.shape[0]),
        "dose_scale_to_original_histories": rows.shape[0] / histories,
        "water_entrance_mm": args.water_entrance_mm,
        "gpu_entry_epsilon_mm": args.gpu_entry_epsilon_mm,
        "gpu_source_z_mm": gpu_source_z_mm,
        "plane_name": args.plane_name,
        "gpu_csv": str(gpu_csv),
        "topas_phase_space_base": str(topas_base),
    }
    (args.output_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
