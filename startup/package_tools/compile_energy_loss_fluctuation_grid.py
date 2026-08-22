#!/usr/bin/env python3
"""Compile validated thin-slab fluctuation points into the runtime CSV grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


POINT_SCHEMA = "maigo-energy-loss-fluctuation-point-v1"
GRID_SCHEMA = "maigo-energy-loss-fluctuation-grid-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_numeric_list(value: Any, label: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def trapezoid_integral(probabilities: list[float], quantiles: list[float]) -> float:
    return math.fsum(
        0.5 * (left_q + right_q) * (right_p - left_p)
        for left_p, right_p, left_q, right_q in zip(
            probabilities, probabilities[1:], quantiles, quantiles[1:]
        )
    )


def load_point(path: Path) -> dict[str, Any]:
    try:
        point = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    point = require_mapping(point, str(path))
    if point.get("schema") != POINT_SCHEMA:
        raise ValueError(f"{path}: expected schema {POINT_SCHEMA!r}")

    projectile = require_mapping(point.get("projectile"), f"{path}: projectile")
    z = positive_integer(projectile.get("Z"), f"{path}: projectile Z")
    a = positive_integer(projectile.get("A"), f"{path}: projectile A")
    if a < z:
        raise ValueError(f"{path}: projectile A must be at least Z")
    material = point.get("material")
    if (
        not isinstance(material, str)
        or not material
        or any(character in material for character in ",#\r\n")
        or material.strip() != material
    ):
        raise ValueError(f"{path}: material is empty or unsafe for the runtime CSV")

    energy = finite_number(
        point.get("energy_MeV_per_u"), f"{path}: energy", positive=True
    )
    density = finite_number(
        point.get("areal_density_g_per_cm2"),
        f"{path}: areal density",
        positive=True,
    )
    histories = positive_integer(point.get("histories"), f"{path}: histories")
    sample_loss = require_mapping(
        point.get("sample_energy_loss_MeV"), f"{path}: sample energy loss"
    )
    sample_mean = finite_number(
        sample_loss.get("mean"), f"{path}: sample mean loss", positive=True
    )
    sample_minimum = finite_number(
        sample_loss.get("minimum"), f"{path}: sample minimum loss"
    )
    sample_maximum = finite_number(
        sample_loss.get("maximum"), f"{path}: sample maximum loss"
    )
    if sample_minimum < 0.0 or sample_maximum < sample_minimum:
        raise ValueError(f"{path}: invalid sample energy-loss range")

    inverse_cdf = require_mapping(point.get("inverse_cdf"), f"{path}: inverse_cdf")
    probabilities = require_numeric_list(
        inverse_cdf.get("probabilities"), f"{path}: probabilities"
    )
    quantiles = require_numeric_list(
        inverse_cdf.get("loss_over_mean_quantiles"), f"{path}: quantiles"
    )
    if len(probabilities) < 3 or len(quantiles) != len(probabilities):
        raise ValueError(f"{path}: inverse CDF needs matching arrays with at least 3 points")
    if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
        raise ValueError(f"{path}: probabilities require exact 0 and 1 endpoints")
    if any(right <= left for left, right in zip(probabilities, probabilities[1:])):
        raise ValueError(f"{path}: probabilities must be strictly increasing")
    if any(value < 0.0 for value in quantiles) or any(
        right < left for left, right in zip(quantiles, quantiles[1:])
    ):
        raise ValueError(f"{path}: quantiles must be nonnegative and nondecreasing")
    integral = trapezoid_integral(probabilities, quantiles)
    if abs(integral - 1.0) > 1.0e-9:
        raise ValueError(
            f"{path}: piecewise-linear inverse-CDF mean is {integral:.17g}, not 1"
        )

    sources = require_mapping(point.get("sources"), f"{path}: sources")
    for kind in ("header", "phsp"):
        source = require_mapping(sources.get(kind), f"{path}: source {kind}")
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise ValueError(f"{path}: source {kind} requires a path")
        digest = source.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{path}: source {kind} has an invalid SHA-256")

    return {
        "path": path,
        "raw": point,
        "z": z,
        "a": a,
        "material": material,
        "energy": energy,
        "density": density,
        "histories": histories,
        "sample_mean": sample_mean,
        "probabilities": probabilities,
        "quantiles": quantiles,
    }


def probability_label(value: float) -> str:
    return format(value, ".17g")


def compile_grid(
    point_paths: list[Path], output: Path, output_metadata: Path
) -> dict[str, Any]:
    if not point_paths:
        raise ValueError("at least one point JSON is required")
    points = [load_point(path) for path in point_paths]
    identity = (points[0]["z"], points[0]["a"], points[0]["material"])
    probabilities = points[0]["probabilities"]
    coordinates: set[tuple[float, float]] = set()
    for point in points:
        point_identity = (point["z"], point["a"], point["material"])
        if point_identity != identity:
            raise ValueError(
                f"{point['path']}: projectile/material identity {point_identity} "
                f"does not match {identity}"
            )
        if point["probabilities"] != probabilities:
            raise ValueError(f"{point['path']}: probability grid does not match other points")
        coordinate = (point["energy"], point["density"])
        if coordinate in coordinates:
            raise ValueError(
                f"duplicate fluctuation grid point energy={coordinate[0]:g} "
                f"density={coordinate[1]:g}"
            )
        coordinates.add(coordinate)

    energies = sorted({point["energy"] for point in points})
    densities = sorted({point["density"] for point in points})
    if len(energies) < 2 or len(densities) < 2:
        raise ValueError("runtime fluctuation grid requires at least two energies and densities")
    densities_by_energy = {
        energy: sorted(
            point["density"] for point in points if point["energy"] == energy
        )
        for energy in energies
    }
    insufficient = [
        energy
        for energy, local_densities in densities_by_energy.items()
        if len(local_densities) < 2
    ]
    if insufficient:
        formatted = ", ".join(f"{energy:g}" for energy in insufficient)
        raise ValueError(
            "each fluctuation energy requires at least two density points; "
            f"insufficient energies: {formatted}"
        )

    points.sort(key=lambda point: (point["energy"], point["density"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "projectile_Z", "projectile_A", "material",
                "energy_MeV_per_u", "areal_density_g_per_cm2",
                *(f"q_{probability_label(value)}" for value in probabilities),
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point["z"], point["a"], point["material"],
                    format(point["energy"], ".17g"),
                    format(point["density"], ".17g"),
                    *(format(value, ".17g") for value in point["quantiles"]),
                ]
            )

    metadata: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "runtime_format": "EnergyLossFluctuationTable CSV v1",
        "projectile": {"Z": identity[0], "A": identity[1]},
        "material": identity[2],
        "grid": {
            "energies_MeV_per_u": energies,
            "areal_densities_g_per_cm2": densities,
            "areal_densities_by_energy_g_per_cm2": [
                {
                    "energy_MeV_per_u": energy,
                    "values": densities_by_energy[energy],
                }
                for energy in energies
            ],
            "layout": (
                "ragged input; runtime loader expands local density grids "
                "to their union"
            ),
            "probabilities": probabilities,
            "point_count": len(points),
        },
        "normalization": {
            "sample": "primary_kinetic_energy_loss / per-point_TOPAS_sample_mean",
            "inverse_cdf": "piecewise-linear integral normalized to exactly one",
            "empirical_energy_or_projectile_scale": False,
        },
        "histories": {
            "total": sum(point["histories"] for point in points),
            "per_point": [
                {
                    "energy_MeV_per_u": point["energy"],
                    "areal_density_g_per_cm2": point["density"],
                    "count": point["histories"],
                }
                for point in points
            ],
        },
        "points": [
            {
                "path": str(point["path"]),
                "sha256": sha256(point["path"]),
                "energy_MeV_per_u": point["energy"],
                "areal_density_g_per_cm2": point["density"],
                "sample_mean_energy_loss_MeV": point["sample_mean"],
                "sources": point["raw"]["sources"],
            }
            for point in points
        ],
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metadata = compile_grid(args.points, args.output, args.output_metadata)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"compiled fluctuation grid: {args.output} "
        f"({metadata['grid']['point_count']} points)"
    )


if __name__ == "__main__":
    main()
