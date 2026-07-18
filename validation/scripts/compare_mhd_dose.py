#!/usr/bin/env python3
"""Compare two float MetaImage dose grids and write benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_mhd(path: Path) -> tuple[list[float], dict[str, object]]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    if fields.get("ElementType") != "MET_FLOAT":
        raise ValueError(f"{path}: only MET_FLOAT is supported")
    shape = tuple(int(value) for value in fields["DimSize"].split())
    spacing = tuple(float(value) for value in fields["ElementSpacing"].split())
    origin = tuple(float(value) for value in fields.get("Offset", "0 0 0").split())
    if len(shape) != 3 or len(spacing) != 3 or len(origin) != 3:
        raise ValueError(f"{path}: expected 3D shape, spacing, and offset")
    raw_path = path.parent / fields["ElementDataFile"]
    count = math.prod(shape)
    data = raw_path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{raw_path}: expected {count * 4} bytes, found {len(data)}")
    values = list(struct.unpack(f"<{count}f", data))
    return values, {
        "path": path.as_posix(),
        "raw_path": raw_path.as_posix(),
        "shape_xyz": list(shape),
        "spacing_mm": list(spacing),
        "origin_mm": list(origin),
        "units": fields.get("DoseUnits", "unspecified"),
    }


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    variance_left = sum((value - mean_left) ** 2 for value in left)
    variance_right = sum((value - mean_right) ** 2 for value in right)
    denominator = math.sqrt(variance_left * variance_right)
    return covariance / denominator if denominator > 0.0 else 0.0


def sobp_metrics(
    values: list[float], metadata: dict[str, object], start_mm: float, end_mm: float
) -> dict[str, object]:
    nx, ny, nz = metadata["shape_xyz"]
    _, _, spacing_z = metadata["spacing_mm"]
    _, _, origin_z = metadata["origin_mm"]
    ix, iy = nx // 2, ny // 2
    samples: list[tuple[float, float]] = []
    for iz in range(nz):
        z_mm = origin_z + iz * spacing_z
        if start_mm <= z_mm <= end_mm:
            linear = iz * nx * ny + iy * nx + ix
            samples.append((z_mm, values[linear]))
    dose = [value for _, value in samples]
    mean = sum(dose) / len(dose) if dose else 0.0
    minimum = min(dose) if dose else 0.0
    maximum = max(dose) if dose else 0.0
    standard_deviation = (
        math.sqrt(sum((value - mean) ** 2 for value in dose) / len(dose)) if dose else 0.0
    )
    return {
        "depth_range_mm": [start_mm, end_mm],
        "central_voxel_xy": [ix, iy],
        "sample_count": len(samples),
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "flatness_percent": (
            100.0 * (maximum - minimum) / (maximum + minimum)
            if maximum + minimum > 0.0
            else 0.0
        ),
        "coefficient_of_variation_percent": 100.0 * standard_deviation / mean if mean > 0.0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-dose-threshold", type=float, default=0.01)
    parser.add_argument("--sobp-start-mm", type=float, default=50.0)
    parser.add_argument("--sobp-end-mm", type=float, default=100.0)
    args = parser.parse_args()

    topas, topas_metadata = read_mhd(args.topas)
    gpu, gpu_metadata = read_mhd(args.gpu)
    for key in ("shape_xyz", "spacing_mm", "origin_mm"):
        if topas_metadata[key] != gpu_metadata[key]:
            raise ValueError(f"Grid mismatch for {key}: {topas_metadata[key]} != {gpu_metadata[key]}")

    differences = [gpu_value - topas_value for gpu_value, topas_value in zip(gpu, topas)]
    topas_integral = sum(topas)
    gpu_integral = sum(gpu)
    topas_maximum = max(topas)
    if topas_integral <= 0.0 or topas_maximum <= 0.0:
        raise ValueError("TOPAS reference dose must have a positive integral and maximum")
    threshold = topas_maximum * args.high_dose_threshold
    selected = [(gpu_value, topas_value) for gpu_value, topas_value in zip(gpu, topas) if topas_value >= threshold]
    if not selected:
        raise ValueError("No TOPAS voxels passed the high-dose threshold")
    selected_gpu = [pair[0] for pair in selected]
    selected_topas = [pair[1] for pair in selected]

    metrics = {
        "normalization": "Gy/primary",
        "topas": topas_metadata,
        "gpu": gpu_metadata,
        "integral": {
            "topas": topas_integral,
            "gpu": gpu_integral,
            "signed_percent": 100.0 * (gpu_integral / topas_integral - 1.0),
        },
        "voxel": {
            "normalized_L1_percent": 100.0 * sum(abs(value) for value in differences) / topas_integral,
            "normalized_RMSE_to_topas_max_percent": 100.0
            * math.sqrt(sum(value * value for value in differences) / len(differences))
            / topas_maximum,
            "high_dose_threshold_percent_of_topas_max": 100.0 * args.high_dose_threshold,
            "high_dose_voxel_count": len(selected),
            "high_dose_mean_abs_percent": 100.0
            * sum(abs(gpu_value - topas_value) / topas_value for gpu_value, topas_value in selected)
            / len(selected),
            "high_dose_pearson_r": pearson(selected_gpu, selected_topas),
        },
        "sobp": {
            "topas": sobp_metrics(topas, topas_metadata, args.sobp_start_mm, args.sobp_end_mm),
            "gpu": sobp_metrics(gpu, gpu_metadata, args.sobp_start_mm, args.sobp_end_mm),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
