#!/usr/bin/env python3
"""Summarize TOPAS minibeam phase-space planes and primary C-12 transmission."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


C12_PDG = 1_000_060_120
MINIBEAM_SLIT_COUNT = 15
MINIBEAM_SLIT_PITCH_MM = 3.6
HEADER_COUNT_RE = re.compile(r"^Number of (.+):\s+(\d+)\s*$")


def parse_header(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    species: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADER_COUNT_RE.match(line.strip())
        if not match:
            continue
        label, value = match.groups()
        if label == "Original Histories":
            metadata["original_histories"] = int(value)
        elif label == "Original Histories that Reached Phase Space":
            metadata["histories_reaching_plane"] = int(value)
        elif label == "Scored Particles":
            metadata["scored_particles"] = int(value)
        else:
            species[label] = int(value)
    metadata["header_species_counts"] = species
    return metadata


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_primary_c12(path: Path) -> dict[str, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != 14:
        raise ValueError(f"{path}: expected 14 columns, found {values.shape[1]}")
    pdg = values[:, 7].astype(np.int64)
    track_id = values[:, 12].astype(np.int64)
    parent_id = values[:, 13].astype(np.int64)
    primary = (pdg == C12_PDG) & (track_id == 1) & (parent_id == 0)
    selected = values[primary]
    direction_x = selected[:, 3]
    direction_y = selected[:, 4]
    direction_z = np.sqrt(
        np.maximum(
            0.0, 1.0 - direction_x * direction_x -
            direction_y * direction_y
        )
    )
    direction_z[selected[:, 8].astype(np.int64) == 1] *= -1.0
    return {
        "x_mm": 10.0 * selected[:, 0],
        "plane_mm": 10.0 * selected[:, 1],
        "z_mm": 10.0 * selected[:, 2],
        "direction_x": direction_x,
        "direction_y": direction_y,
        "direction_z": direction_z,
        "energy_MeV": selected[:, 5],
        "event_id": selected[:, 11].astype(np.int64),
    }


def direct_air_event_ids(entrance_phsp: Path) -> set[int]:
    primary = load_primary_c12(entrance_phsp)
    x_mm = primary["x_mm"]
    z_mm = primary["z_mm"]
    direction_y = primary["direction_y"]
    distance_mm = -primary["plane_mm"] / direction_y
    exit_x_mm = x_mm + distance_mm * primary["direction_x"]
    exit_z_mm = z_mm + distance_mm * primary["direction_z"]
    nearest_slit = np.where(
        x_mm >= 0.0,
        np.floor(x_mm / MINIBEAM_SLIT_PITCH_MM + 0.5),
        np.ceil(x_mm / MINIBEAM_SLIT_PITCH_MM - 0.5),
    ).astype(np.int64)
    exit_slit = np.where(
        exit_x_mm >= 0.0,
        np.floor(exit_x_mm / MINIBEAM_SLIT_PITCH_MM + 0.5),
        np.ceil(exit_x_mm / MINIBEAM_SLIT_PITCH_MM - 0.5),
    ).astype(np.int64)
    direct = (
        (nearest_slit == exit_slit)
        & (np.abs(x_mm - nearest_slit * MINIBEAM_SLIT_PITCH_MM) < 0.25)
        & (
            np.abs(
                exit_x_mm - exit_slit * MINIBEAM_SLIT_PITCH_MM
            ) < 0.25
        )
        & (np.abs(z_mm) < 25.0)
        & (np.abs(exit_z_mm) < 25.0)
    )
    return set(primary["event_id"][direct].tolist())


def energy_statistics(energy_MeV: np.ndarray) -> dict[str, float | int]:
    count = int(energy_MeV.size)
    if count == 0:
        return {"count": 0}
    standard_deviation = float(np.std(energy_MeV))
    return {
        "count": count,
        "mean_MeV": float(np.mean(energy_MeV)),
        "std_MeV": standard_deviation,
        "standard_error_MeV": standard_deviation / math.sqrt(count),
    }


def summarize_plane(phsp: Path, header: Path) -> dict[str, object]:
    values = np.loadtxt(phsp, dtype=np.float64, ndmin=2)
    if values.shape[1] != 14:
        raise ValueError(f"{phsp}: expected 14 columns, found {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError(f"{phsp}: non-finite phase-space values")

    x_mm = 10.0 * values[:, 0]
    plane_coordinate_mm = 10.0 * values[:, 1]
    z_mm = 10.0 * values[:, 2]
    direction_x = values[:, 3]
    direction_y = values[:, 4]
    direction_z_squared = np.maximum(
        0.0, 1.0 - direction_x * direction_x - direction_y * direction_y
    )
    direction_z = np.sqrt(direction_z_squared)
    direction_z[values[:, 8].astype(np.int64) == 1] *= -1.0
    energy_MeV = values[:, 5]
    pdg = values[:, 7].astype(np.int64)
    event_id = values[:, 11].astype(np.int64)
    track_id = values[:, 12].astype(np.int64)
    parent_id = values[:, 13].astype(np.int64)

    metadata = parse_header(header)
    scored = int(metadata.get("scored_particles", -1))
    if scored != values.shape[0]:
        raise ValueError(
            f"{phsp}: header reports {scored} particles, file has {values.shape[0]}"
        )
    original_histories = int(metadata.get("original_histories", 0))
    primary = (pdg == C12_PDG) & (track_id == 1) & (parent_id == 0)
    primary_count = int(np.count_nonzero(primary))

    unique_pdg, counts = np.unique(pdg, return_counts=True)
    result: dict[str, object] = {
        **metadata,
        "phsp": phsp.as_posix(),
        "header": header.as_posix(),
        "rows": int(values.shape[0]),
        "mean_plane_world_y_mm": float(np.mean(plane_coordinate_mm)),
        "pdg_counts": {
            str(int(code)): int(count)
            for code, count in zip(unique_pdg, counts, strict=True)
        },
        "primary_c12": {
            "count": primary_count,
            "fraction_of_incident_histories": (
                primary_count / original_histories if original_histories else None
            ),
        },
    }
    if primary_count:
        primary_result = result["primary_c12"]
        assert isinstance(primary_result, dict)
        primary_result.update(
            {
                "unique_events": int(np.unique(event_id[primary]).size),
                "energy_MeV": {
                    "mean": float(np.mean(energy_MeV[primary])),
                    "std": float(np.std(energy_MeV[primary])),
                    "min": float(np.min(energy_MeV[primary])),
                    "max": float(np.max(energy_MeV[primary])),
                },
                "position_x_mm": {
                    "mean": float(np.mean(x_mm[primary])),
                    "std": float(np.std(x_mm[primary])),
                },
                "position_z_mm": {
                    "mean": float(np.mean(z_mm[primary])),
                    "std": float(np.std(z_mm[primary])),
                },
                "direction_x": {
                    "mean": float(np.mean(direction_x[primary])),
                    "std": float(np.std(direction_x[primary])),
                },
                "direction_y": {
                    "mean": float(np.mean(direction_y[primary])),
                    "std": float(np.std(direction_y[primary])),
                },
                "direction_z": {
                    "mean": float(np.mean(direction_z[primary])),
                    "std": float(np.std(direction_z[primary])),
                },
                "correlation_x_direction_x": safe_correlation(
                    x_mm[primary], direction_x[primary]
                ),
                "correlation_z_direction_z": safe_correlation(
                    z_mm[primary], direction_z[primary]
                ),
            }
        )
        half_count = MINIBEAM_SLIT_COUNT // 2
        nearest_slit = np.where(
            x_mm[primary] >= 0.0,
            np.floor(x_mm[primary] / MINIBEAM_SLIT_PITCH_MM + 0.5),
            np.ceil(x_mm[primary] / MINIBEAM_SLIT_PITCH_MM - 0.5),
        ).astype(np.int64)
        primary_result["nearest_slit_primary_counts"] = {
            "pitch_mm": MINIBEAM_SLIT_PITCH_MM,
            "slit_indices": list(range(-half_count, half_count + 1)),
            "counts": [
                int(np.count_nonzero(nearest_slit == slit))
                for slit in range(-half_count, half_count + 1)
            ],
        }
        if abs(float(np.mean(plane_coordinate_mm)) + 60.0) < 0.1:
            entrance_x = x_mm[primary]
            entrance_z = z_mm[primary]
            dy = direction_y[primary]
            distance = -float(np.mean(plane_coordinate_mm)) / dy
            exit_x = entrance_x + distance * direction_x[primary]
            exit_z = entrance_z + distance * direction_z[primary]
            entrance_slit = nearest_slit
            exit_slit = np.where(
                exit_x >= 0.0,
                np.floor(exit_x / MINIBEAM_SLIT_PITCH_MM + 0.5),
                np.ceil(exit_x / MINIBEAM_SLIT_PITCH_MM - 0.5),
            ).astype(np.int64)
            entrance_delta = (
                entrance_x - entrance_slit * MINIBEAM_SLIT_PITCH_MM
            )
            exit_delta = exit_x - exit_slit * MINIBEAM_SLIT_PITCH_MM
            direct = (
                (entrance_slit == exit_slit)
                & (np.abs(entrance_delta) < 0.25)
                & (np.abs(exit_delta) < 0.25)
                & (np.abs(entrance_z) < 25.0)
                & (np.abs(exit_z) < 25.0)
            )
            primary_result["direct_air_primary_counts"] = {
                "counts": [
                    int(
                        np.count_nonzero(
                            direct & (entrance_slit == slit)
                        )
                    )
                    for slit in range(-half_count, half_count + 1)
                ],
                "total": int(np.count_nonzero(direct)),
            }
    return result


def analyze(planes: list[tuple[str, Path, Path]]) -> dict[str, object]:
    summaries = {
        name: summarize_plane(phsp, header)
        for name, phsp, header in planes
    }
    names = list(summaries)
    transmission: dict[str, object] = {}
    if names:
        first = summaries[names[0]]
        first_primary = int(first["primary_c12"]["count"])  # type: ignore[index]
        for name in names[1:]:
            count = int(summaries[name]["primary_c12"]["count"])  # type: ignore[index]
            transmission[f"{names[0]}_to_{name}"] = {
                "primary_c12_count_ratio": (
                    count / first_primary if first_primary else None
                )
            }
    conditioned_primary: dict[str, object] = {}
    if len(planes) >= 2:
        entrance_name, entrance_phsp, _ = planes[0]
        downstream_name, downstream_phsp, _ = planes[-1]
        direct_events = direct_air_event_ids(entrance_phsp)
        downstream = load_primary_c12(downstream_phsp)
        direct = np.fromiter(
            (
                int(event_id) in direct_events
                for event_id in downstream["event_id"]
            ),
            dtype=bool,
            count=downstream["event_id"].size,
        )
        conditioned_primary = {
            "entrance_plane": entrance_name,
            "downstream_plane": downstream_name,
            "classification": (
                "direct if the entrance ray remains inside the same "
                "0.5 mm x 50 mm slit through the 60 mm collimator"
            ),
            "direct_air": energy_statistics(
                downstream["energy_MeV"][direct]
            ),
            "copper_touched": energy_statistics(
                downstream["energy_MeV"][~direct]
            ),
        }
    return {
        "direction_cosine_reconstruction": (
            "TOPAS columns give world X/Y; world Z is reconstructed from "
            "sqrt(1-X^2-Y^2) and the negative-third-cosine flag"
        ),
        "planes": summaries,
        "transmission": transmission,
        "downstream_primary_c12_by_collimator_path": conditioned_primary,
    }


def plane_argument(text: str) -> tuple[str, Path, Path]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "plane must be NAME:PHSP:HEADER"
        )
    name, phsp, header = parts
    if not name:
        raise argparse.ArgumentTypeError("plane name must not be empty")
    return name, Path(phsp), Path(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plane",
        action="append",
        type=plane_argument,
        required=True,
        help="NAME:PHSP:HEADER; repeat in beamline order",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.plane)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
