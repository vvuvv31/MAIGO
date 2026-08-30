#!/usr/bin/env python3
"""Compile fluctuation-point JSON files into a runtime inverse-CDF CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


POINT_SCHEMA = "maigo-energy-loss-fluctuation-point-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_point(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != POINT_SCHEMA:
        raise ValueError(f"invalid fluctuation point: {path}")
    return value


def trapezoid(probabilities: list[float], quantiles: list[float]) -> float:
    total = 0.0
    for index in range(1, len(probabilities)):
        total += 0.5 * (quantiles[index - 1] + quantiles[index]) * (
            probabilities[index] - probabilities[index - 1]
        )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        loaded = [load_point(path) for path in args.points]
        identity = None
        probabilities = None
        seen: set[tuple[float, float]] = set()
        rows: list[dict[str, Any]] = []
        for point in loaded:
            projectile = point.get("projectile")
            material = point.get("material")
            inverse = point.get("inverse_cdf")
            if not isinstance(projectile, dict) or not isinstance(inverse, dict):
                raise ValueError("point identity or inverse CDF is missing")
            coord = (
                float(point["energy_MeV_per_u"]),
                float(point["areal_density_g_per_cm2"]),
            )
            if coord in seen:
                raise ValueError(f"duplicate fluctuation grid point {coord}")
            seen.add(coord)
            probs = inverse["probabilities"]
            quants = inverse["loss_over_mean_quantiles"]
            if identity is None:
                identity = (int(projectile["Z"]), int(projectile["A"]), str(material))
                probabilities = list(probs)
            else:
                if (int(projectile["Z"]), int(projectile["A"]), str(material)) != identity:
                    raise ValueError("does not match")
                if list(probs) != probabilities:
                    raise ValueError("probability grid does not match")
            if any(q < 0.0 for q in quants) or any(
                quants[i] < quants[i - 1] for i in range(1, len(quants))
            ):
                raise ValueError("nonnegative and nondecreasing")
            mean = trapezoid(list(probs), list(quants))
            if abs(mean - 1.0) > 0.001:
                raise ValueError("not 1")
            rows.append(
                {
                    "energy": coord[0],
                    "density": coord[1],
                    "quantiles": list(quants),
                    "histories": int(point.get("histories", 0)),
                }
            )
        assert identity is not None and probabilities is not None
        rows.sort(key=lambda row: (row["energy"], row["density"]))
        by_energy: dict[float, list[float]] = {}
        for row in rows:
            by_energy.setdefault(row["energy"], []).append(row["density"])
        if len(by_energy) < 2:
            raise ValueError("at least two energies")
        for energy, densities in by_energy.items():
            if len(densities) < 2:
                raise ValueError(
                    f"energy {energy} needs at least two density points"
                )
        union = sorted({row["density"] for row in rows})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "projectile_Z",
            "projectile_A",
            "material",
            "energy_MeV_per_u",
            "areal_density_g_per_cm2",
            *[f"q_{format(probability, 'g')}" for probability in probabilities],
        ]
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in rows:
                writer.writerow(
                    [
                        identity[0],
                        identity[1],
                        identity[2],
                        row["energy"],
                        row["density"],
                        *row["quantiles"],
                    ]
                )
        metadata = {
            "schema": "maigo-energy-loss-fluctuation-runtime-v1",
            "projectile": {"Z": identity[0], "A": identity[1]},
            "material": identity[2],
            "grid": {
                "point_count": len(rows),
                "energies_MeV_per_u": sorted(by_energy),
                "areal_densities_g_per_cm2": union,
                "areal_densities_by_energy_g_per_cm2": [
                    {"energy_MeV_per_u": energy, "values": by_energy[energy]}
                    for energy in sorted(by_energy)
                ],
                "probability_count": len(probabilities),
            },
            "histories": {
                "total": sum(row["histories"] for row in rows),
                "per_point": [row["histories"] for row in rows],
            },
            "normalization": {
                "empirical_energy_or_projectile_scale": False,
            },
            "output": {
                "path": str(args.output.resolve()),
                "sha256": sha256(args.output),
            },
        }
        args.output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
