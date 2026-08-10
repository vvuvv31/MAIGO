#!/usr/bin/env python3
"""Quantify the dose-loss bounds introduced by sparse-Dij thresholding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from match_gpu_to_physical_dose import read_mhd


def mhd_values(path: Path) -> np.ndarray:
    _metadata, values = read_mhd(path)
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dij", type=Path, required=True)
    parser.add_argument("--weights-mat", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--body-mask", type=Path)
    parser.add_argument("--gamma-dose-percent", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.dij, "r") as handle:
        bio = handle["bioDij"]
        cutoff = float(bio["thresholds/dose"][0, 0])
        kept_nnz = np.asarray(bio["totalDoseNnzBySpot"], dtype=np.float64).reshape(-1)
        dropped_nnz = np.asarray(
            bio["totalDoseDroppedBySpot"], dtype=np.float64
        ).reshape(-1)
        kept_sum = np.asarray(
            bio["totalDoseSumBySpot"], dtype=np.float64
        ).reshape(-1)
    with h5py.File(args.weights_mat, "r") as handle:
        weights = np.asarray(handle["resultGUI/w"], dtype=np.float64).reshape(-1)
    if not (weights.size == kept_nnz.size == dropped_nnz.size == kept_sum.size):
        raise ValueError("Dij columns and resultGUI.w have different lengths")

    reference = mhd_values(args.reference)
    reference_sum = float(reference.sum())
    reference_peak = float(reference.max())
    body_mask = None
    gamma_normalization_peak = reference_peak
    if args.body_mask is not None:
        body_mask = mhd_values(args.body_mask) > 0.5
        if body_mask.size != reference.size or not np.any(body_mask):
            raise ValueError("body mask is empty or has a different volume size")
        gamma_normalization_peak = float(reference[body_mask].max())
    weighted_dropped_count = float(np.dot(weights, dropped_nnz))
    omitted_integral_upper = cutoff * weighted_dropped_count
    per_voxel_upper = cutoff * float(weights.sum())
    report: dict[str, object] = {
        "dij": str(args.dij),
        "weights_mat": str(args.weights_mat),
        "reference": str(args.reference),
        "dose_threshold_Gy_per_spot_voxel": cutoff,
        "spots": int(weights.size),
        "positive_weight_spots": int(np.count_nonzero(weights)),
        "weight_sum": float(weights.sum()),
        "kept_nnz": int(kept_nnz.sum()),
        "dropped_nnz": int(dropped_nnz.sum()),
        "dropped_nnz_fraction": float(
            dropped_nnz.sum() / (kept_nnz.sum() + dropped_nnz.sum())
        ),
        "weighted_kept_integral_Gy_voxel": float(np.dot(weights, kept_sum)),
        "reference_integral_Gy_voxel": reference_sum,
        "reference_peak_Gy": reference_peak,
        "body_mask": str(args.body_mask) if args.body_mask is not None else None,
        "gamma_normalization_peak_Gy": gamma_normalization_peak,
        "weighted_dropped_count": weighted_dropped_count,
        "omitted_integral_upper_bound_Gy_voxel": omitted_integral_upper,
        "omitted_integral_upper_bound_percent_reference": float(
            100.0 * omitted_integral_upper / reference_sum
        ),
        "omitted_per_voxel_upper_bound_Gy": per_voxel_upper,
        "omitted_per_voxel_upper_bound_percent_peak": float(
            100.0 * per_voxel_upper / gamma_normalization_peak
        ),
        "gamma_global_dose_tolerance_Gy": float(
            args.gamma_dose_percent * 0.01 * gamma_normalization_peak
        ),
    }
    if args.gpu is not None:
        gpu = mhd_values(args.gpu)
        if gpu.size != reference.size:
            raise ValueError("GPU and reference volume sizes differ")
        gpu_sum = float(gpu.sum())
        gap = gpu_sum - reference_sum
        report.update(
            {
                "gpu": str(args.gpu),
                "gpu_integral_Gy_voxel": gpu_sum,
                "gpu_minus_reference_percent": 100.0 * gap / reference_sum,
                "mean_dropped_value_needed_to_explain_gpu_gap_Gy": (
                    gap / weighted_dropped_count
                ),
                "fraction_of_threshold_needed_to_explain_gpu_gap": (
                    gap / weighted_dropped_count / cutoff
                ),
            }
        )
        if body_mask is not None:
            body_reference_sum = float(reference[body_mask].sum())
            body_gpu_sum = float(gpu[body_mask].sum())
            report.update(
                {
                    "body_reference_integral_Gy_voxel": body_reference_sum,
                    "body_gpu_integral_Gy_voxel": body_gpu_sum,
                    "body_gpu_minus_reference_percent": float(
                        100.0
                        * (body_gpu_sum - body_reference_sum)
                        / body_reference_sum
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
