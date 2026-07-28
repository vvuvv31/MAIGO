#!/usr/bin/env python3
"""Compute high-resolution planar gamma for aligned minibeam MHD doses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_minibeam_dose import read_mhd


def shifted_view(values: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = values.shape
    output = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.zeros(values.shape, dtype=bool)
    src_y0, src_y1 = max(0, -dy), min(ny, ny - dy)
    src_x0, src_x1 = max(0, -dx), min(nx, nx - dx)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    output[dst_y0:dst_y1, dst_x0:dst_x1] = values[
        src_y0:src_y1, src_x0:src_x1
    ]
    valid[dst_y0:dst_y1, dst_x0:dst_x1] = True
    return output, valid


def gamma_plane(
    reference: np.ndarray,
    evaluation: np.ndarray,
    spacing_xy_mm: tuple[float, float],
    global_maximum: float,
    threshold_fraction: float,
    dose_percent: float,
    distance_mm: float,
    local: bool,
) -> dict[str, float | int]:
    selected = reference >= threshold_fraction * global_maximum
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"selected_points": 0, "pass_percent": math.nan}
    if local:
        dose_tolerance = dose_percent * 0.01 * reference
    else:
        dose_tolerance = np.full(
            reference.shape,
            dose_percent * 0.01 * global_maximum,
            dtype=np.float64,
        )
    dose_tolerance = np.maximum(dose_tolerance, np.finfo(np.float64).tiny)
    best = np.full(reference.shape, np.inf, dtype=np.float64)
    sx, sy = spacing_xy_mm
    max_dx = int(math.floor(distance_mm / sx + 1.0e-12))
    max_dy = int(math.floor(distance_mm / sy + 1.0e-12))
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            distance = math.hypot(dx * sx, dy * sy)
            if distance > distance_mm + 1.0e-12:
                continue
            shifted, valid = shifted_view(evaluation, dy, dx)
            with np.errstate(over="ignore", invalid="ignore"):
                gamma2 = (distance / distance_mm) ** 2 + (
                    (shifted - reference) / dose_tolerance
                ) ** 2
            best[valid] = np.minimum(best[valid], gamma2[valid])
    gamma = np.sqrt(best[selected])
    return {
        "selected_points": count,
        "pass_percent": 100.0 * float(np.count_nonzero(gamma <= 1.0)) / count,
        "mean_gamma": float(np.mean(gamma)),
        "p95_gamma": float(np.percentile(gamma, 95.0)),
    }


def dose_only(
    reference: np.ndarray,
    evaluation: np.ndarray,
    global_maximum: float,
    threshold_fraction: float,
    dose_percent: float,
    local: bool,
) -> dict[str, float | int]:
    selected = reference >= threshold_fraction * global_maximum
    count = int(np.count_nonzero(selected))
    tolerance = (
        dose_percent * 0.01 * reference
        if local
        else dose_percent * 0.01 * global_maximum
    )
    passed = np.abs(evaluation - reference) <= tolerance
    return {
        "selected_points": count,
        "pass_percent": (
            100.0 * float(np.count_nonzero(passed & selected)) / count
            if count
            else math.nan
        ),
    }


def gamma_line(
    reference: np.ndarray,
    evaluation: np.ndarray,
    spacing_mm: float,
    threshold_fraction: float,
    dose_percent: float,
    distance_mm: float,
    local: bool,
) -> dict[str, float | int]:
    maximum = float(np.max(reference))
    selected = reference >= threshold_fraction * maximum
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"selected_points": 0, "pass_percent": math.nan}
    tolerance = (
        dose_percent * 0.01 * reference
        if local
        else np.full(reference.shape, dose_percent * 0.01 * maximum)
    )
    tolerance = np.maximum(tolerance, np.finfo(np.float64).tiny)
    best = np.full(reference.shape, np.inf, dtype=np.float64)
    maximum_offset = int(math.floor(distance_mm / spacing_mm + 1.0e-12))
    for offset in range(-maximum_offset, maximum_offset + 1):
        distance = abs(offset) * spacing_mm
        if distance > distance_mm + 1.0e-12:
            continue
        shifted = np.full(reference.shape, np.nan, dtype=np.float64)
        valid = np.zeros(reference.shape, dtype=bool)
        if offset < 0:
            shifted[:offset] = evaluation[-offset:]
            valid[:offset] = True
        elif offset > 0:
            shifted[offset:] = evaluation[:-offset]
            valid[offset:] = True
        else:
            shifted[:] = evaluation
            valid[:] = True
        with np.errstate(over="ignore", invalid="ignore"):
            gamma2 = (distance / distance_mm) ** 2 + (
                (shifted - reference) / tolerance
            ) ** 2
        best[valid] = np.minimum(best[valid], gamma2[valid])
    gamma = np.sqrt(best[selected])
    return {
        "selected_points": count,
        "pass_percent": 100.0 * float(np.count_nonzero(gamma <= 1.0)) / count,
        "mean_gamma": float(np.mean(gamma)),
        "p95_gamma": float(np.percentile(gamma, 95.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-scale", type=float, default=1.0)
    parser.add_argument("--depths-mm", type=float, nargs="+", default=[0.5, 35.0, 70.25])
    parser.add_argument(
        "--slab-width-mm",
        type=float,
        default=0.0,
        help="Sum this depth thickness before planar gamma; 0 uses one slice",
    )
    parser.add_argument("--threshold-percent", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_mhd(args.reference)
    evaluation = read_mhd(args.evaluation)
    for field in ("shape_xyz", "spacing_xyz_mm", "offset_xyz_mm"):
        if getattr(reference, field) != getattr(evaluation, field):
            raise ValueError(f"Grid mismatch for {field}")
    spacing_x, spacing_y, spacing_z = reference.spacing_xyz_mm
    offset_z = reference.offset_xyz_mm[2]
    prepared_planes: list[
        tuple[float, float, float, np.ndarray, np.ndarray]
    ] = []
    nz = reference.data_zyx.shape[0]
    z_centers = offset_z + np.arange(nz, dtype=np.float64) * spacing_z
    for requested_depth in args.depths_mm:
        if args.slab_width_mm > 0.0:
            selected_z = np.flatnonzero(
                np.abs(z_centers - requested_depth)
                <= 0.5 * args.slab_width_mm + 1.0e-12
            )
            if selected_z.size == 0:
                selected_z = np.asarray(
                    [int(np.argmin(np.abs(z_centers - requested_depth)))]
                )
        else:
            selected_z = np.asarray(
                [int(np.argmin(np.abs(z_centers - requested_depth)))]
            )
        ref = np.sum(
            reference.data_zyx[selected_z], axis=0, dtype=np.float64
        )
        eva = (
            np.sum(evaluation.data_zyx[selected_z], axis=0, dtype=np.float64)
            * args.evaluation_scale
        )
        prepared_planes.append(
            (
                requested_depth,
                float(z_centers[selected_z[0]]),
                float(z_centers[selected_z[-1]]),
                ref,
                eva,
            )
        )
    global_maximum = max(float(np.max(item[3])) for item in prepared_planes)
    result: dict[str, object] = {
        "normalization": "prescribed history-count scale; no fitted dose scale",
        "evaluation_scale": args.evaluation_scale,
        "reference_global_maximum_Gy": global_maximum,
        "gamma_dimension": "2D transverse x-y plane",
        "slab_width_mm": args.slab_width_mm,
        "note": (
            "The y spacing is 1 mm, so 0.2/0.3 mm searches move only along "
            "the 0.1 mm x axis on this scorer grid. Global dose tolerance is "
            "normalized to the maximum among the evaluated depth planes/slabs."
        ),
        "planes": [],
    }
    for requested_depth, depth_first, depth_last, ref, eva in prepared_planes:
        entry: dict[str, object] = {
            "requested_depth_mm": requested_depth,
            "sampled_depth_span_mm": [depth_first, depth_last],
            "thresholds": {},
            "x_projection_gamma": {},
        }
        for threshold_percent in args.threshold_percent:
            threshold = threshold_percent * 0.01
            metrics: dict[str, object] = {}
            for dose_percent, distance_mm in ((3.0, 0.3), (2.0, 0.2)):
                label = f"{dose_percent:g}pct_{distance_mm:g}mm"
                metrics[f"global_{label}"] = gamma_plane(
                    ref, eva, (spacing_x, spacing_y), global_maximum,
                    threshold, dose_percent, distance_mm, False,
                )
                metrics[f"local_{label}"] = gamma_plane(
                    ref, eva, (spacing_x, spacing_y), global_maximum,
                    threshold, dose_percent, distance_mm, True,
                )
            metrics["global_3pct_0mm"] = dose_only(
                ref, eva, global_maximum, threshold, 3.0, False
            )
            metrics["local_3pct_0mm"] = dose_only(
                ref, eva, global_maximum, threshold, 3.0, True
            )
            entry["thresholds"][f"{threshold_percent:g}pct"] = metrics
            ref_x = np.sum(ref, axis=0, dtype=np.float64)
            eva_x = np.sum(eva, axis=0, dtype=np.float64)
            projected: dict[str, object] = {}
            for dose_percent, distance_mm in ((3.0, 0.3), (2.0, 0.2)):
                label = f"{dose_percent:g}pct_{distance_mm:g}mm"
                projected[f"global_{label}"] = gamma_line(
                    ref_x, eva_x, spacing_x, threshold,
                    dose_percent, distance_mm, False,
                )
                projected[f"local_{label}"] = gamma_line(
                    ref_x, eva_x, spacing_x, threshold,
                    dose_percent, distance_mm, True,
                )
            entry["x_projection_gamma"][f"{threshold_percent:g}pct"] = projected
        result["planes"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
