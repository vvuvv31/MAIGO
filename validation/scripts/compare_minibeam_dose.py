#!/usr/bin/env python3
"""Compare two large minibeam MHD dose grids without loading them into RAM.

The comparison is deliberately absolute: GPU and TOPAS must represent the same
number of incident histories.  No fitted dose scale or peak normalization is
applied.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MhdGrid:
    path: Path
    raw_path: Path
    shape_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    offset_xyz_mm: tuple[float, float, float]
    data_zyx: np.memmap


def _triplet(fields: dict[str, str], key: str, cast: type) -> tuple:
    values = tuple(cast(item) for item in fields[key].split())
    if len(values) != 3:
        raise ValueError(f"{key}: expected three values, found {values}")
    return values


def read_mhd(path: Path) -> MhdGrid:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    if fields.get("ElementType") != "MET_FLOAT":
        raise ValueError(f"{path}: only little-endian MET_FLOAT is supported")
    if fields.get("BinaryDataByteOrderMSB", "False").lower() == "true":
        raise ValueError(f"{path}: big-endian MHD is not supported")

    shape_xyz = _triplet(fields, "DimSize", int)
    spacing = _triplet(fields, "ElementSpacing", float)
    offset = tuple(
        float(value) for value in fields.get("Offset", "0 0 0").split()
    )
    if len(offset) != 3:
        raise ValueError(f"{path}: Offset must contain three values")
    raw_path = path.parent / fields["ElementDataFile"]
    expected_bytes = math.prod(shape_xyz) * np.dtype("<f4").itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{raw_path}: expected {expected_bytes} bytes, "
            f"found {raw_path.stat().st_size}"
        )
    nx, ny, nz = shape_xyz
    data = np.memmap(raw_path, dtype="<f4", mode="r", shape=(nz, ny, nx))
    return MhdGrid(path, raw_path, shape_xyz, spacing, offset, data)


def compare(
    reference: MhdGrid,
    evaluation: MhdGrid,
    threshold_fraction: float,
    chunk_depth: int,
    evaluation_scale: float = 1.0,
) -> dict[str, object]:
    for field in ("shape_xyz", "spacing_xyz_mm", "offset_xyz_mm"):
        left = getattr(reference, field)
        right = getattr(evaluation, field)
        if left != right:
            raise ValueError(f"Grid mismatch for {field}: {left} != {right}")
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold fraction must be in [0, 1]")
    if chunk_depth <= 0:
        raise ValueError("chunk depth must be positive")
    if not math.isfinite(evaluation_scale) or evaluation_scale <= 0.0:
        raise ValueError("evaluation scale must be finite and positive")

    nz, ny, nx = reference.data_zyx.shape
    reference_max = 0.0
    reference_sum = 0.0
    evaluation_sum = 0.0
    nonzero_reference = 0
    depth_reference = np.zeros(nz, dtype=np.float64)
    depth_evaluation = np.zeros(nz, dtype=np.float64)
    lateral_x_reference = np.zeros(nx, dtype=np.float64)
    lateral_x_evaluation = np.zeros(nx, dtype=np.float64)
    lateral_y_reference = np.zeros(ny, dtype=np.float64)
    lateral_y_evaluation = np.zeros(ny, dtype=np.float64)

    for start in range(0, nz, chunk_depth):
        stop = min(nz, start + chunk_depth)
        ref = np.asarray(reference.data_zyx[start:stop], dtype=np.float64)
        eva = (
            np.asarray(evaluation.data_zyx[start:stop], dtype=np.float64)
            * evaluation_scale
        )
        if not np.isfinite(ref).all() or not np.isfinite(eva).all():
            raise ValueError("Dose grids contain non-finite values")
        if np.min(ref) < 0.0 or np.min(eva) < 0.0:
            raise ValueError("Dose grids contain negative values")
        reference_max = max(reference_max, float(np.max(ref)))
        reference_sum += float(np.sum(ref, dtype=np.float64))
        evaluation_sum += float(np.sum(eva, dtype=np.float64))
        nonzero_reference += int(np.count_nonzero(ref))
        depth_reference[start:stop] = np.sum(ref, axis=(1, 2))
        depth_evaluation[start:stop] = np.sum(eva, axis=(1, 2))
        lateral_x_reference += np.sum(ref, axis=(0, 1))
        lateral_x_evaluation += np.sum(eva, axis=(0, 1))
        lateral_y_reference += np.sum(ref, axis=(0, 2))
        lateral_y_evaluation += np.sum(eva, axis=(0, 2))

    if reference_max <= 0.0 or reference_sum <= 0.0:
        raise ValueError("Reference dose must be positive")
    threshold = threshold_fraction * reference_max
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    selected_count = 0
    selected_abs_relative_sum = 0.0
    sum_ref = sum_eva = sum_ref2 = sum_eva2 = sum_cross = 0.0

    for start in range(0, nz, chunk_depth):
        stop = min(nz, start + chunk_depth)
        ref = np.asarray(reference.data_zyx[start:stop], dtype=np.float64)
        eva = (
            np.asarray(evaluation.data_zyx[start:stop], dtype=np.float64)
            * evaluation_scale
        )
        delta = eva - ref
        absolute_error_sum += float(np.sum(np.abs(delta), dtype=np.float64))
        squared_error_sum += float(np.sum(delta * delta, dtype=np.float64))
        selected = ref >= threshold
        selected_count += int(np.count_nonzero(selected))
        if np.any(selected):
            r = ref[selected]
            e = eva[selected]
            selected_abs_relative_sum += float(
                np.sum(np.abs(e - r) / r, dtype=np.float64)
            )
            sum_ref += float(np.sum(r, dtype=np.float64))
            sum_eva += float(np.sum(e, dtype=np.float64))
            sum_ref2 += float(np.sum(r * r, dtype=np.float64))
            sum_eva2 += float(np.sum(e * e, dtype=np.float64))
            sum_cross += float(np.sum(r * e, dtype=np.float64))

    if selected_count == 0:
        raise ValueError(
            "No reference voxels passed the requested dose threshold"
        )
    covariance = sum_cross - sum_ref * sum_eva / selected_count
    variance_ref = sum_ref2 - sum_ref * sum_ref / selected_count
    variance_eva = sum_eva2 - sum_eva * sum_eva / selected_count
    pearson_denominator = math.sqrt(max(0.0, variance_ref * variance_eva))
    pearson = (
        covariance / pearson_denominator
        if pearson_denominator > 0.0
        else 0.0
    )

    def profile_metrics(
        reference_values: np.ndarray,
        evaluation_values: np.ndarray,
    ) -> dict[str, float]:
        reference_total = float(np.sum(reference_values, dtype=np.float64))
        reference_centered = reference_values - float(
            np.mean(reference_values)
        )
        evaluation_centered = evaluation_values - float(
            np.mean(evaluation_values)
        )
        denominator = math.sqrt(
            float(np.sum(reference_centered * reference_centered))
            * float(np.sum(evaluation_centered * evaluation_centered))
        )
        return {
            "normalized_L1_percent": 100.0
            * float(
                np.sum(
                    np.abs(evaluation_values - reference_values),
                    dtype=np.float64,
                )
            )
            / reference_total,
            "pearson_r": (
                float(
                    np.sum(
                        reference_centered * evaluation_centered,
                        dtype=np.float64,
                    )
                )
                / denominator
                if denominator > 0.0
                else 0.0
            ),
        }

    def distal_crossing(values: np.ndarray, fraction: float) -> float:
        peak = int(np.argmax(values))
        target = fraction * float(values[peak])
        for index in range(peak + 1, values.size):
            if values[index] <= target < values[index - 1]:
                left = offset_z + (index - 1) * spacing_z
                right = offset_z + index * spacing_z
                denominator = float(values[index] - values[index - 1])
                if denominator == 0.0:
                    return left
                weight = (target - float(values[index - 1])) / denominator
                return left + weight * (right - left)
        return math.nan

    spacing_x, spacing_y, spacing_z = reference.spacing_xyz_mm
    offset_x, offset_y, offset_z = reference.offset_xyz_mm
    return {
        "normalization": (
            "absolute Gy; evaluation scale is prescribed, not fitted"
        ),
        "evaluation_scale": evaluation_scale,
        "reference_mhd": reference.path.as_posix(),
        "evaluation_mhd": evaluation.path.as_posix(),
        "grid": {
            "shape_xyz": list(reference.shape_xyz),
            "spacing_xyz_mm": list(reference.spacing_xyz_mm),
            "offset_first_voxel_center_xyz_mm": list(reference.offset_xyz_mm),
        },
        "integral_summed_voxel_dose_Gy": {
            "reference": reference_sum,
            "evaluation": evaluation_sum,
            "evaluation_over_reference": evaluation_sum / reference_sum,
            "signed_difference_percent": 100.0
            * (evaluation_sum / reference_sum - 1.0),
        },
        "voxel": {
            "reference_maximum_Gy": reference_max,
            "reference_nonzero_count": nonzero_reference,
            "normalized_L1_percent": 100.0
            * absolute_error_sum
            / reference_sum,
            "normalized_RMSE_to_reference_max_percent": 100.0
            * math.sqrt(squared_error_sum / (nx * ny * nz))
            / reference_max,
            "threshold_percent_of_reference_max": 100.0
            * threshold_fraction,
            "selected_count": selected_count,
            "selected_mean_absolute_relative_difference_percent": (
                100.0 * selected_abs_relative_sum / selected_count
            ),
            "selected_pearson_r": pearson,
        },
        "profiles": {
            "depth_integrated": profile_metrics(
                depth_reference, depth_evaluation
            ),
            "range": {
                "reference_R80_mm": distal_crossing(
                    depth_reference, 0.8
                ),
                "evaluation_R80_mm": distal_crossing(
                    depth_evaluation, 0.8
                ),
                "delta_R80_mm": (
                    distal_crossing(depth_evaluation, 0.8)
                    - distal_crossing(depth_reference, 0.8)
                ),
            },
            "lateral_x_integrated": profile_metrics(
                lateral_x_reference, lateral_x_evaluation
            ),
            "lateral_y_integrated": profile_metrics(
                lateral_y_reference, lateral_y_evaluation
            ),
            "reference_depth_max_center_mm": offset_z
            + float(np.argmax(depth_reference)) * spacing_z,
            "evaluation_depth_max_center_mm": offset_z
            + float(np.argmax(depth_evaluation)) * spacing_z,
            "reference_lateral_x_max_center_mm": offset_x
            + float(np.argmax(lateral_x_reference)) * spacing_x,
            "evaluation_lateral_x_max_center_mm": offset_x
            + float(np.argmax(lateral_x_evaluation)) * spacing_x,
            "reference_lateral_y_max_center_mm": offset_y
            + float(np.argmax(lateral_y_reference)) * spacing_y,
            "evaluation_lateral_y_max_center_mm": offset_y
            + float(np.argmax(lateral_y_evaluation)) * spacing_y,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=1.0)
    parser.add_argument("--chunk-depth", type=int, default=10)
    parser.add_argument(
        "--evaluation-scale",
        type=float,
        default=1.0,
        help="Prescribed scale applied to evaluation dose (for history-count normalization)",
    )
    args = parser.parse_args()
    metrics = compare(
        read_mhd(args.reference),
        read_mhd(args.evaluation),
        args.threshold_percent / 100.0,
        args.chunk_depth,
        args.evaluation_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
