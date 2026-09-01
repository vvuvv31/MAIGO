#!/usr/bin/env python3
"""Convert a TOPAS rate-only capture and compare it with GPU depth counters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_gpu(path: Path) -> tuple[list[float], list[int]]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    return [float(row["depth_mm"]) for row in rows], [int(row["count"]) for row in rows]


def read_crossing(path: Path) -> list[int]:
    counts: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or fields[0:2] != ["0", "0"]:
            raise ValueError(f"{path}:{line_number}: expected one-voxel X/Y crossing row")
        z, count = int(fields[2]), int(fields[3])
        if z in counts:
            raise ValueError(f"{path}:{line_number}: duplicate Z bin {z}")
        counts[z] = count
    if sorted(counts) != list(range(800)):
        raise ValueError(f"{path}: expected Z bins 0..799")
    result = [counts[index] for index in range(800)]
    if any(current > previous for previous, current in zip(result, result[1:])):
        raise ValueError(f"{path}: primary survival is not non-increasing")
    return result


def write_profile(path: Path, counts: list[int]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("depth_mm", "count"))
        for index, count in enumerate(counts):
            writer.writerow((f"{index * 0.5 + 0.25:g}", count))


def max_cdf_difference(left: list[int], right: list[int]) -> float:
    left_total, right_total = sum(left), sum(right)
    left_sum = right_sum = 0
    maximum = 0.0
    for left_value, right_value in zip(left, right):
        left_sum += left_value
        right_sum += right_value
        maximum = max(maximum, abs(left_sum / left_total - right_sum / right_total))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossing", type=Path, required=True)
    parser.add_argument("--first-depths", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    survival = read_crossing(args.crossing)
    first = [0] * 800
    for line_number, line in enumerate(args.first_depths.read_text(encoding="ascii").splitlines(), 1):
        depth = float(line)
        if not math.isfinite(depth) or not 0.0 <= depth <= 400.0:
            raise ValueError(f"{args.first_depths}:{line_number}: invalid depth {depth}")
        first[min(799, int(depth / 0.5))] += 1

    write_profile(args.output_dir / "primary_survival.csv", survival)
    write_profile(args.output_dir / "first_reactions.csv", first)
    write_profile(args.output_dir / "reaction_rate.csv", first)

    gpu_depth, gpu_first = read_gpu(args.gpu_dir / "inelastic_reactions.csv")
    survival_depth, gpu_survival = read_gpu(args.gpu_dir / "primary_survival.csv")
    if gpu_depth != survival_depth or len(gpu_depth) != 800:
        raise ValueError("GPU rate-only outputs must share the expected 800-bin grid")
    gpu_first = gpu_first[:500]
    gpu_survival = gpu_survival[:500]
    topas_first = first[:500]
    topas_survival = survival[:500]
    histories = 100_000
    first_fraction_gpu = sum(gpu_first) / histories
    first_fraction_topas = sum(topas_first) / histories
    survival_max_abs = max(abs(a - b) / histories for a, b in zip(gpu_survival, topas_survival))
    closure = [survival[i] - survival[i + 1] - first[i] for i in range(499)]
    report = {
        "schema": "MAIGO_TOPAS_RATE_ONLY_COMPARISON_V1",
        "histories": histories,
        "comparison_depth_mm": 250.0,
        "gpu": {"first_reactions": sum(gpu_first), "first_reaction_fraction": first_fraction_gpu},
        "topas": {
            "first_reactions_0_250mm": sum(topas_first),
            "first_reactions_full_400mm": sum(first),
            "first_reaction_fraction_0_250mm": first_fraction_topas,
        },
        "metrics": {
            "first_reaction_fraction_absolute_difference": abs(first_fraction_gpu - first_fraction_topas),
            "first_reaction_fraction_relative_difference": abs(first_fraction_gpu - first_fraction_topas) / first_fraction_topas,
            "first_reaction_cdf_max_absolute_difference": max_cdf_difference(gpu_first, topas_first),
            "primary_survival_max_absolute_fraction_difference": survival_max_abs,
        },
        "topas_internal": {
            "survival_minus_next_survival_minus_first_max_abs_count": max(abs(value) for value in closure),
            "survival_at_0_25_mm": survival[0],
            "survival_at_249_75_mm": survival[499],
        },
        "target_fraction": {
            "status": "unavailable",
            "reason": "job 267 produced no CINEL02 raw files; CarbonReactionNtuple does not record target Z",
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
