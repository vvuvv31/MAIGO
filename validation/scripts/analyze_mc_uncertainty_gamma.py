#!/usr/bin/env python3
"""Estimate batch uncertainty and calculate uncertainty-aware 3D gamma."""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import gamma_3d, read_mhd


def load(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, values = read_mhd(path)
    return metadata, np.asarray(values, dtype=np.float64)


def same_grid(first: dict[str, str], second: dict[str, str], label: str) -> None:
    for key in ("DimSize", "ElementSpacing", "Offset"):
        if first.get(key) != second.get(key):
            raise ValueError(
                f"{label}: {key} mismatch: "
                f"{first.get(key)!r} != {second.get(key)!r}"
            )


def map_beam_y(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    return values.reshape(nz, ny, nx).transpose(1, 0, 2).reshape(-1)


def batch_mean_and_sem(
    first: np.ndarray,
    second: np.ndarray,
    histories_per_batch: float,
) -> tuple[np.ndarray, np.ndarray]:
    first_per_history = first / histories_per_batch
    second_per_history = second / histories_per_batch
    mean = 0.5 * (first_per_history + second_per_history)
    # For two independent batches, sample-SD/sqrt(2) = |x1-x2|/2.
    standard_error_of_mean = 0.5 * np.abs(
        first_per_history - second_per_history
    )
    return mean, standard_error_of_mean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-batch", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-batch", type=Path, action="append", required=True)
    parser.add_argument("--reference-histories", type=float, required=True)
    parser.add_argument("--evaluation-histories", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dose-percent", type=float, default=3.0)
    parser.add_argument("--distance-mm", type=float, default=0.3)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.1)
    parser.add_argument("--gamma-points", type=int, default=300000)
    args = parser.parse_args()

    if len(args.reference_batch) != 2 or len(args.evaluation_batch) != 2:
        raise ValueError("exactly two independent batches are required per engine")

    ref_meta_1, ref_1 = load(args.reference_batch[0])
    ref_meta_2, ref_2 = load(args.reference_batch[1])
    same_grid(ref_meta_1, ref_meta_2, "reference batches")
    eval_meta_1, eval_1 = load(args.evaluation_batch[0])
    eval_meta_2, eval_2 = load(args.evaluation_batch[1])
    same_grid(eval_meta_1, eval_meta_2, "evaluation batches")

    reference_shape = tuple(int(v) for v in ref_meta_1["DimSize"].split())
    reference_spacing = tuple(
        float(v) for v in ref_meta_1["ElementSpacing"].split()
    )
    evaluation_shape = tuple(int(v) for v in eval_meta_1["DimSize"].split())
    expected_evaluation_shape = (
        reference_shape[0],
        reference_shape[2],
        reference_shape[1],
    )
    if evaluation_shape != expected_evaluation_shape:
        raise ValueError(
            f"beam_y evaluation shape {evaluation_shape} != "
            f"expected {expected_evaluation_shape}"
        )

    reference, reference_sem = batch_mean_and_sem(
        ref_1, ref_2, args.reference_histories
    )
    evaluation, evaluation_sem = batch_mean_and_sem(
        eval_1, eval_2, args.evaluation_histories
    )
    evaluation = map_beam_y(evaluation, evaluation_shape)
    evaluation_sem = map_beam_y(evaluation_sem, evaluation_shape)

    reference_array = array.array("f", reference.astype(np.float32, copy=False))
    evaluation_list = evaluation.tolist()
    report: dict[str, object] = {
        "reference_batches": [str(path) for path in args.reference_batch],
        "evaluation_batches": [str(path) for path in args.evaluation_batch],
        "histories_per_reference_batch": args.reference_histories,
        "histories_per_evaluation_batch": args.evaluation_histories,
        "dose_normalization": "per incident history before two-batch mean",
        "batch_sem_estimator": "abs(batch1 - batch2) / 2",
        "criterion": {
            "dose_percent": args.dose_percent,
            "distance_mm": args.distance_mm,
            "threshold_percent": args.threshold_percent,
            "interpolation_step_mm": args.gamma_resolution_mm,
        },
        "relative_sem_percentiles_thr10": {},
        "gamma": {},
    }

    selected = reference >= args.threshold_percent * 0.01 * float(reference.max())
    for label, dose, sem in (
        ("reference", reference, reference_sem),
        ("evaluation", evaluation, evaluation_sem),
    ):
        relative = 100.0 * sem[selected] / np.maximum(dose[selected], 1.0e-30)
        report["relative_sem_percentiles_thr10"][label] = {
            "p50": float(np.percentile(relative, 50)),
            "p90": float(np.percentile(relative, 90)),
            "p95": float(np.percentile(relative, 95)),
        }

    for coverage in (0.0, 1.0, 2.0):
        label = "raw" if coverage == 0.0 else f"{coverage:g}sigma"
        report["gamma"][label] = gamma_3d(
            reference_array,
            evaluation_list,
            reference_shape,
            reference_spacing,
            args.dose_percent,
            args.distance_mm,
            args.threshold_percent,
            args.gamma_points,
            0,
            local_dose=True,
            interpolation_step_mm=args.gamma_resolution_mm,
            reference_uncertainty=reference_sem,
            evaluation_uncertainty=evaluation_sem,
            uncertainty_coverage=coverage,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
