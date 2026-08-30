#!/usr/bin/env python3
"""Build one inverse-CDF fluctuation grid point from a TOPAS ASCII n-tuple."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


POINT_SCHEMA = "maigo-energy-loss-fluctuation-point-v1"
REQUIRED_COLUMNS = (
    "Run ID",
    "Event ID",
    "Thread ID",
    "Primary Track ID",
    "Atomic Number Z",
    "Mass Number A",
    "Material Name",
    "Entry Kinetic Energy (MeV)",
    "Exit Kinetic Energy (MeV)",
    "Primary Kinetic Energy Loss (MeV)",
    "Primary Local Deposit (MeV)",
    "Primary Path Length (mm)",
    "Primary Step Count",
    "Material Consistent",
    "Completed",
    "Completion Status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_probabilities() -> list[float]:
    values = [i / 300.0 for i in range(0, 298)]
    values.extend(0.99 + i / 2000.0 for i in range(1, 19))
    values.extend(0.999 + i / 10000.0 for i in range(1, 10))
    values.extend(0.9999 + i / 100000.0 for i in range(1, 10))
    values.append(1.0)
    return values


def empirical_quantile(samples: list[float], probability: float) -> float:
    if not samples:
        raise ValueError("cannot compute a quantile of an empty sample")
    ordered = sorted(samples)
    if probability <= 0.0:
        return ordered[0]
    if probability >= 1.0:
        return ordered[-1]
    count = len(ordered)
    h = (count - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return ordered[lower]
    fraction = h - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def trapezoid_integral(probabilities: list[float], quantiles: list[float]) -> float:
    total = 0.0
    for index in range(1, len(probabilities)):
        total += 0.5 * (quantiles[index - 1] + quantiles[index]) * (
            probabilities[index] - probabilities[index - 1]
        )
    return total


def exact_type7_quantile_integral(samples: list[float]) -> float:
    ordered = sorted(samples)
    count = len(ordered)
    if count == 1:
        return ordered[0]
    total = 0.0
    for index in range(1, count):
        total += 0.5 * (ordered[index - 1] + ordered[index]) / (count - 1)
    return total


def parse_header(path: Path) -> tuple[int, int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    histories = None
    entries = None
    columns: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("Number of Original Histories:"):
            histories = int(line.split(":", 1)[1].strip())
        elif line.startswith("Number of Scored Entries:"):
            entries = int(line.split(":", 1)[1].strip())
        elif line[:1].isdigit() and ":" in line:
            columns.append(line.split(":", 1)[1].strip())
    if histories is None or entries is None or columns != list(REQUIRED_COLUMNS):
        raise ValueError(f"unexpected fluctuation header: {path}")
    return histories, entries, columns


def parse_bool(token: str) -> bool:
    return token in {"1", "true", "True", "TRUE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projectile-z", type=int, required=True)
    parser.add_argument("--projectile-a", type=int, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--energy-mev-per-u", type=float, required=True)
    parser.add_argument("--areal-density-g-per-cm2", type=float, required=True)
    parser.add_argument("--expected-histories", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem = args.input
    header_path = Path(f"{stem}.header") if stem.suffix != ".header" else stem
    if header_path.suffix != ".header":
        header_path = Path(str(stem) + ".header")
        phsp_path = Path(str(stem) + ".phsp")
    else:
        phsp_path = header_path.with_suffix(".phsp")
        header_path = Path(str(stem) + ".header") if not str(stem).endswith(".header") else header_path
    if not str(args.input).endswith(".header") and not str(args.input).endswith(".phsp"):
        header_path = Path(str(args.input) + ".header")
        phsp_path = Path(str(args.input) + ".phsp")
    try:
        histories, entries, _ = parse_header(header_path)
        if entries != args.expected_histories or histories != args.expected_histories:
            raise ValueError(
                f"wrote {entries} rows for {args.expected_histories} histories"
            )
        losses: list[float] = []
        with phsp_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 16:
                    raise ValueError(f"short fluctuation row {line_number}")
                atomic_number = int(fields[4])
                mass_number = int(fields[5])
                material = fields[6]
                if atomic_number != args.projectile_z or mass_number != args.projectile_a:
                    raise ValueError(
                        f"expected projectile Z{args.projectile_z}A{args.projectile_a}"
                    )
                if material != args.material:
                    raise ValueError(f"expected material {args.material}")
                entry = float(fields[7])
                exit_energy = float(fields[8])
                loss = float(fields[9])
                if abs((entry - exit_energy) - loss) > 1.0e-4 * max(1.0, abs(entry)):
                    raise ValueError("does not match recorded loss")
                if not parse_bool(fields[13]):
                    raise ValueError("material was not consistent")
                if not parse_bool(fields[14]) or fields[15] != "exited":
                    raise ValueError("did not exit")
                if not math.isfinite(loss) or loss < 0.0:
                    raise ValueError("invalid energy loss")
                losses.append(loss)
        if len(losses) != args.expected_histories:
            raise ValueError(
                f"wrote {len(losses)} rows for {args.expected_histories} histories"
            )
        mean = math.fsum(losses) / len(losses)
        if not (mean > 0.0):
            raise ValueError("mean energy loss must be positive")
        ratios = [loss / mean for loss in losses]
        probabilities = default_probabilities()
        raw_quantiles = [
            empirical_quantile(ratios, probability) for probability in probabilities
        ]
        exact_mean = exact_type7_quantile_integral(ratios)
        integral_before = trapezoid_integral(probabilities, raw_quantiles)
        if exact_mean <= 0.0:
            raise ValueError("empirical type-7 mean is not positive")
        compression_error = integral_before / exact_mean - 1.0
        factor = 1.0 / integral_before
        quantiles = [value * factor for value in raw_quantiles]
        point = {
            "schema": POINT_SCHEMA,
            "projectile": {"Z": args.projectile_z, "A": args.projectile_a},
            "material": args.material,
            "energy_MeV_per_u": args.energy_mev_per_u,
            "areal_density_g_per_cm2": args.areal_density_g_per_cm2,
            "histories": args.expected_histories,
            "sample_energy_loss_MeV": {
                "mean": mean,
                "minimum": min(losses),
                "maximum": max(losses),
            },
            "inverse_cdf": {
                "probabilities": probabilities,
                "loss_over_mean_quantiles": quantiles,
                "piecewise_linear_mean_before_normalization": integral_before,
                "empirical_type7_mean_exact": exact_mean,
                "piecewise_linear_compression_mean_relative_error": compression_error,
                "unit_mean_normalization_factor": factor,
            },
            "sources": {
                "header": {"path": str(header_path.resolve()), "sha256": sha256(header_path)},
                "phsp": {"path": str(phsp_path.resolve()), "sha256": sha256(phsp_path)},
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(point) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
