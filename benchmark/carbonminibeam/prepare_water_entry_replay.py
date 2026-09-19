#!/usr/bin/env python3
"""Prepare matched water-entry plane replay inputs for GPU and TOPAS.

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


def ion_za(pdg: int) -> tuple[int, int] | None:
    """Return (Z, A) for supported positive ions in a TOPAS PDG column."""
    if pdg == 2212:
        return 1, 1
    if pdg < 1_000_000_000:
        return None
    z = (pdg // 10_000) % 1_000
    a = (pdg // 10) % 1_000
    if z <= 0 or a <= 0 or z > a:
        return None
    return z, a


def original_histories(header: Path) -> int:
    match = re.search(
        r"^Number of Original Histories:\s*(\d+)\s*$",
        header.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing original-history count in {header}")
    return int(match.group(1))


def write_topas_header(path: Path, histories: int, rows: np.ndarray,
                       particle_label: str) -> None:
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

Number of {particle_label}: {rows.shape[0]}
Minimum Kinetic Energy of {particle_label}: {energies.min():.12g} MeV
Maximum Kinetic Energy of {particle_label}: {energies.max():.12g} MeV
"""
    path.write_text(text)


def write_gpu_spots(rows: np.ndarray, output: Path,
                    gpu_source_z_mm: float) -> None:
    dx = rows[:, 3]
    dy_topas = rows[:, 4]
    dz_topas = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy_topas * dy_topas))
    dz_topas = np.where(rows[:, 8] != 0, -dz_topas, dz_topas)
    if np.any(dy_topas <= 0.0):
        raise ValueError("GPU replay requires forward-going entrance particles")
    # The canonical rows are on z=0. Move the GPU start a small distance into
    # water along the particle direction, including the transverse drift.
    path_to_gpu_start = gpu_source_z_mm / dy_topas
    gpu_x_mm = rows[:, 0] * 10.0 + path_to_gpu_start * dx
    gpu_y_mm = rows[:, 2] * 10.0 + path_to_gpu_start * dz_topas
    with output.open("w", newline="") as stream:
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
                f"{gpu_x_mm[index]:.12g}", f"{gpu_y_mm[index]:.12g}",
                f"{gpu_source_z_mm:.12g}",
                f"{dx[index]:.12g}", f"{dz_topas[index]:.12g}",
                f"{dy_topas[index]:.12g}", 0, 0, 0, 0, 0, 0, 0])


def advance_to_water_entrance(data: np.ndarray,
                              water_entrance_mm: float) -> np.ndarray:
    """Propagate forward tracks to the common TOPAS +Y water boundary."""
    dx = data[:, 3]
    dy = data[:, 4]
    dz = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy * dy))
    dz = np.where(data[:, 8] != 0, -dz, dz)
    forward = dy > 1.0e-12
    rows = data[forward].copy()
    path_mm = (water_entrance_mm - rows[:, 1] * 10.0) / rows[:, 4]
    if np.any(path_mm < -1.0e-6):
        raise ValueError("Forward phase-space record is downstream of water entrance")
    rows[:, 0] += path_mm * rows[:, 3] / 10.0
    rows[:, 1] = water_entrance_mm / 10.0
    rows[:, 2] += path_mm * dz[forward] / 10.0
    return rows


def write_identity_replay(rows: np.ndarray, output: Path,
                          gpu_source_z_mm: float,
                          origin_filter: str | None = None) -> int:
    """Write the lossless charged-ion replay contract in GPU coordinates."""
    written = 0
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "origin", "run_id", "event_id", "track_id", "parent_id",
            "pdg", "atomic_number", "mass_number", "kinetic_energy_MeV",
            "weight", "x_mm", "y_mm", "z_mm", "direction_x",
            "direction_y", "direction_z"])
        for row in rows:
            pdg = int(round(row[7]))
            za = ion_za(pdg)
            if za is None:
                continue
            z, a = za
            dx = row[3]
            depth_direction = row[4]
            transverse_y_direction = np.sqrt(max(
                0.0, 1.0 - dx * dx - depth_direction * depth_direction))
            if row[8] != 0:
                transverse_y_direction = -transverse_y_direction
            # Match write_gpu_spots exactly: start the identity replay at the
            # same small in-water plane and advance both transverse coordinates
            # along the recorded direction.  Leaving identity tracks at z=0
            # makes the two nominally same-source paths differ at the boundary.
            path_to_gpu_start = gpu_source_z_mm / depth_direction
            gpu_x_mm = row[0] * 10.0 + path_to_gpu_start * dx
            gpu_y_mm = (
                row[2] * 10.0 +
                path_to_gpu_start * transverse_y_direction
            )
            origin = ("primary" if pdg == C12_PDG and
                      int(round(row[13])) == 0 else "fragment")
            if origin_filter is not None and origin != origin_filter:
                continue
            writer.writerow([
                origin, int(round(row[10])), int(round(row[11])),
                int(round(row[12])), int(round(row[13])), pdg, z, a,
                f"{row[5]:.12g}", f"{row[6]:.12g}",
                f"{gpu_x_mm:.12g}", f"{gpu_y_mm:.12g}",
                f"{gpu_source_z_mm:.12g}",
                f"{dx:.12g}", f"{transverse_y_direction:.12g}",
                f"{depth_direction:.12g}"])
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_space", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--water-entrance-mm", type=float, default=60.0)
    parser.add_argument("--gpu-entry-epsilon-mm", type=float, default=1.0e-4)
    parser.add_argument("--gpu-source-z-mm", type=float)
    parser.add_argument("--plane-name", default="water_entrance")
    parser.add_argument(
        "--selection", choices=("primary-c12", "charged-ions"),
        default="primary-c12",
        help="primary-c12 preserves the historical replay; charged-ions emits "
             "one independently runnable source per isotope")
    args = parser.parse_args()

    header = args.header or args.phase_space.with_suffix(".header")
    histories = original_histories(header)
    raw_data = np.loadtxt(args.phase_space, dtype=np.float64)
    raw_data = np.atleast_2d(raw_data)
    if raw_data.shape[1] < 14:
        raise ValueError("TOPAS phase space must contain the 14 diagnostic columns")
    data = advance_to_water_entrance(raw_data, args.water_entrance_mm)
    primary_selection = args.selection == "primary-c12"
    if primary_selection:
        rows = data[(data[:, 7] == C12_PDG) & (data[:, 13] == 0)].copy()
    else:
        supported = np.array(
            [ion_za(int(round(value))) is not None for value in data[:, 7]],
            dtype=bool)
        rows = data[supported].copy()
    if rows.shape[0] == 0:
        raise ValueError(f"No particles found for selection {args.selection}")
    if not np.allclose(rows[:, 6], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("GPU replay currently requires unit phase-space weights")

    # A filtered primary is one independent replay history even when another
    # particle happened to be written first in the original mixed-species file.
    rows[:, 9] = 1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_source_z_mm = (args.gpu_entry_epsilon_mm if args.gpu_source_z_mm is None
                       else args.gpu_source_z_mm)
    groups: list[tuple[int, int, np.ndarray]] = []
    if primary_selection:
        groups.append((6, 12, rows))
    else:
        za = np.array([ion_za(int(round(value))) for value in rows[:, 7]])
        for z, a in sorted({tuple(value) for value in za.tolist()}):
            groups.append((z, a, rows[np.all(za == (z, a), axis=1)]))

    sources = []
    for z, a, group_rows in groups:
        suffix = "primary_c12" if primary_selection else f"ion_z{z}_a{a}"
        topas_base = args.output_dir / f"{args.plane_name}_{suffix}"
        gpu_csv = args.output_dir / f"{args.plane_name}_{suffix}_gpu.csv"
        np.savetxt(topas_base.with_suffix(".phsp"), group_rows, fmt="%.12g")
        write_topas_header(topas_base.with_suffix(".header"), histories,
                           group_rows, f"ion_Z{z}_A{a}")
        write_gpu_spots(group_rows, gpu_csv, gpu_source_z_mm)
        sources.append({
            "atomic_number": z,
            "mass_number": a,
            "particles": int(group_rows.shape[0]),
            "dose_scale_to_original_histories": group_rows.shape[0] / histories,
            "gpu_csv": str(gpu_csv),
            "topas_phase_space_base": str(topas_base),
        })
    identity_replay = args.output_dir / f"{args.plane_name}_{args.selection}_identity.csv"
    write_identity_replay(rows, identity_replay, gpu_source_z_mm)
    fragment_identity_replay = None
    fragment_count = 0
    if not primary_selection:
        fragment_identity_replay = (
            args.output_dir / f"{args.plane_name}_fragments_identity.csv")
        fragment_count = write_identity_replay(
            rows, fragment_identity_replay, gpu_source_z_mm, "fragment")

    metadata = {
        "source_phase_space": str(args.phase_space),
        "source_header": str(header),
        "original_histories": histories,
        "selection": args.selection,
        "replayed_particles": int(rows.shape[0]),
        "dose_scale_to_original_histories": rows.shape[0] / histories,
        "water_entrance_mm": args.water_entrance_mm,
        "gpu_entry_epsilon_mm": args.gpu_entry_epsilon_mm,
        "gpu_source_z_mm": gpu_source_z_mm,
        "plane_name": args.plane_name,
        "sources": sources,
        "identity_replay_csv": str(identity_replay),
        "fragment_identity_replay_csv": (
            str(fragment_identity_replay)
            if fragment_identity_replay is not None else None),
        "fragment_replay_particles": fragment_count,
        "source_plane_world_y_mm_minmax": [
            float(np.min(raw_data[:, 1]) * 10.0),
            float(np.max(raw_data[:, 1]) * 10.0)],
        "canonical_plane_world_y_mm": args.water_entrance_mm,
        "gpu_transport_semantics": (
            "primary C12 replay" if primary_selection else
            "isotope-split primary-source diagnostic; a production-equivalent "
            "mixed-ion replay still requires injection into the GPU secondary "
            "transport queue"),
    }
    if primary_selection:
        # Preserve the historical metadata keys for existing replay drivers.
        metadata["replayed_primary_c12"] = int(rows.shape[0])
        metadata["gpu_csv"] = sources[0]["gpu_csv"]
        metadata["topas_phase_space_base"] = sources[0][
            "topas_phase_space_base"]
    (args.output_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
