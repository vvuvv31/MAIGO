#!/usr/bin/env python3
"""Validate a TOPAS spot plan and export exact matRad optimizer weights.

The matRad dose is ``Dij * x``.  The TOPAS spot files contain the number of
histories used to score each Dij column (L4), so the corresponding absolute
Monte Carlo population is ``L4[i] * x[i]``.  This script keeps the c_01/c_02
file order explicit and records all totals needed for a reduced-history run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np


def vector_from_topas(path: Path, key: str) -> list[float]:
    prefix = f"iv:Tf/Scatterer1/{key}/Values"
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            matches.append(line)
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {prefix!r} line, got {len(matches)}")
    _, rhs = matches[0].split("=", 1)
    tokens = rhs.split()
    declared = int(tokens[0])
    values = [float(value) for value in tokens[1:]]
    if len(values) != declared:
        raise ValueError(
            f"{path}: {key} declares {declared} values but contains {len(values)}"
        )
    return values


def read_csv_vector(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        token = line.split(",", 1)[0].strip()
        if token:
            values.append(float(token))
    return np.asarray(values, dtype=np.float64)


def hamilton_allocation(weights: np.ndarray, total: int) -> np.ndarray:
    """Mirror TopasSpotPlan::apply_weights_from_csv exactly."""
    positive = weights > 0.0
    positive_count = int(np.count_nonzero(positive))
    if total < positive_count:
        raise ValueError(f"total {total} is smaller than {positive_count} active spots")
    result = np.zeros(weights.size, dtype=np.int64)
    distributable = total - positive_count
    exact = distributable * weights[positive] / math.fsum(float(v) for v in weights)
    base = np.floor(exact).astype(np.int64)
    result[positive] = 1 + base
    remaining = total - int(np.sum(result))
    positive_indices = np.flatnonzero(positive)
    # Python's stable sort preserves spot order for equal remainders, as does C++.
    order = sorted(
        range(positive_count), key=lambda i: float(exact[i] - base[i]), reverse=True
    )
    for i in order[:remaining]:
        result[positive_indices[i]] += 1
    if int(np.sum(result)) != total:
        raise AssertionError("Hamilton allocation did not preserve the requested total")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--spot-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-allocation", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--legacy-weights", type=Path)
    parser.add_argument(
        "--pilot-histories",
        type=int,
        default=9_170_000,
        help="actual histories proposed for the reduced GPU run",
    )
    args = parser.parse_args()

    with h5py.File(args.mat, "r") as mat:
        if "x" not in mat:
            raise ValueError(f"{args.mat}: dataset 'x' not found")
        weights = np.asarray(mat["x"], dtype=np.float64).reshape(-1)

    ids: list[int] = []
    base_histories: list[int] = []
    file_counts: list[dict[str, object]] = []
    for path in args.spot_files:
        part_ids_float = vector_from_topas(path, "L0")
        part_hist_float = vector_from_topas(path, "L4")
        if len(part_ids_float) != len(part_hist_float):
            raise ValueError(f"{path}: L0/L4 lengths differ")
        part_ids = [int(value) for value in part_ids_float]
        part_hist = [int(value) for value in part_hist_float]
        if any(float(i) != value for i, value in zip(part_ids, part_ids_float)):
            raise ValueError(f"{path}: non-integral spot ID")
        if any(float(n) != value or n <= 0 for n, value in zip(part_hist, part_hist_float)):
            raise ValueError(f"{path}: L4 histories must be positive integers")
        ids.extend(part_ids)
        base_histories.extend(part_hist)
        file_counts.append(
            {
                "path": str(path),
                "spots": len(part_ids),
                "first_id": part_ids[0],
                "last_id": part_ids[-1],
            }
        )

    if len(ids) != len(weights):
        raise ValueError(f"spot count {len(ids)} != MAT weight count {len(weights)}")
    expected_ids = list(range(1, len(ids) + 1))
    if ids != expected_ids:
        raise ValueError("concatenated L0 spot IDs are not exactly 1..N")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("MAT weights must be finite and nonnegative")

    base = np.asarray(base_histories, dtype=np.float64)
    exact_per_spot = base * weights
    exact_total = float(math.fsum(float(value) for value in exact_per_spot))
    rounded_total = int(round(exact_total))
    pilot = args.pilot_histories
    if pilot <= 0:
        raise ValueError("--pilot-histories must be positive")

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    args.output_weights.write_text(
        "".join(f"{float(value):.17g}\n" for value in weights), encoding="ascii"
    )
    full_allocation = hamilton_allocation(weights, rounded_total)
    pilot_allocation = hamilton_allocation(weights, pilot)
    args.output_allocation.parent.mkdir(parents=True, exist_ok=True)
    with args.output_allocation.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "spot_id",
                "mat_weight",
                "topas_L4_histories",
                "exact_weighted_histories",
                "full_integer_histories",
                "reduced_run_histories",
            ]
        )
        for spot_id, weight, l4, exact, full, reduced in zip(
            ids, weights, base_histories, exact_per_spot, full_allocation, pilot_allocation
        ):
            writer.writerow(
                [
                    spot_id,
                    f"{float(weight):.17g}",
                    l4,
                    f"{float(exact):.17g}",
                    int(full),
                    int(reduced),
                ]
            )

    legacy: dict[str, object] | None = None
    if args.legacy_weights:
        old = read_csv_vector(args.legacy_weights)
        legacy = {"path": str(args.legacy_weights), "count": int(old.size)}
        if old.size == weights.size:
            delta = old - weights
            legacy.update(
                {
                    "sum": float(math.fsum(float(value) for value in old)),
                    "max_abs_difference": float(np.max(np.abs(delta))),
                    "different_values_at_1e-12": int(
                        np.count_nonzero(~np.isclose(old, weights, rtol=0.0, atol=1e-12))
                    ),
                }
            )

    summary = {
        "mat_file": str(args.mat),
        "mat_weight_dataset": "x",
        "spot_files_in_concatenation_order": file_counts,
        "spot_count": len(ids),
        "active_spot_count": int(np.count_nonzero(weights > 0.0)),
        "zero_weight_spot_count": int(np.count_nonzero(weights == 0.0)),
        "spot_ids_contiguous_1_to_n": True,
        "base_histories": {
            "total_unweighted": int(sum(base_histories)),
            "minimum_per_spot": min(base_histories),
            "maximum_per_spot": max(base_histories),
            "all_equal": len(set(base_histories)) == 1,
        },
        "weights": {
            "output_file": str(args.output_weights),
            "sum": float(math.fsum(float(value) for value in weights)),
            "minimum": float(np.min(weights)),
            "maximum": float(np.max(weights)),
            "mean": float(np.mean(weights)),
        },
        "weighted_histories": {
            "definition": "sum_i(TOPAS_L4_i * MAT_x_i)",
            "exact": exact_total,
            "rounded_full_plan": rounded_total,
            "rounded_one_tenth": int(round(exact_total / 10.0)),
            "rounded_one_hundredth": int(round(exact_total / 100.0)),
        },
        "reduced_gpu_run": {
            "actual_histories": pilot,
            "fraction_of_weighted_full_plan": pilot / exact_total,
            "equivalent_dose_scale": exact_total / pilot,
        },
        "integer_allocation": {
            "output_file": str(args.output_allocation),
            "full_minimum_active": int(np.min(full_allocation[full_allocation > 0])),
            "full_maximum_active": int(np.max(full_allocation)),
            "reduced_minimum_active": int(np.min(pilot_allocation[pilot_allocation > 0])),
            "reduced_maximum_active": int(np.max(pilot_allocation)),
        },
        "legacy_weight_comparison": legacy,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Validated {len(ids)} spots ({summary['active_spot_count']} active)")
    print(f"Unweighted TOPAS histories: {sum(base_histories):,}")
    print(f"Weight sum: {summary['weights']['sum']:.15g}")
    print(f"Weighted histories (exact): {exact_total:.9f}")
    print(f"Weighted histories (rounded): {rounded_total:,}")
    print(f"One tenth: {summary['weighted_histories']['rounded_one_tenth']:,}")
    print(f"Pilot: {pilot:,}; dose scale to full = {exact_total / pilot:.12g}")
    print(f"Wrote {args.output_weights}")
    print(f"Wrote {args.output_allocation}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
