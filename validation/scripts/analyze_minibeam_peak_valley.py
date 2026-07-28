#!/usr/bin/env python3
"""Measure transverse minibeam peak, valley, FWHM, and PVDR agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_minibeam_1d_dose import load_mhd, transverse_profiles


def window_mean(
    coordinate: np.ndarray,
    values: np.ndarray,
    center_mm: float,
    half_width_mm: float,
) -> float:
    selected = np.abs(coordinate - center_mm) <= half_width_mm
    if not np.any(selected):
        selected[int(np.argmin(np.abs(coordinate - center_mm)))] = True
    return float(np.mean(values[selected], dtype=np.float64))


def crossing(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    left: bool,
) -> float | None:
    if left:
        indices = range(len(y) - 1, 0, -1)
        pairs = ((index - 1, index) for index in indices)
    else:
        indices = range(0, len(y) - 1)
        pairs = ((index, index + 1) for index in indices)
    for first, second in pairs:
        y0, y1 = float(y[first]), float(y[second])
        if (y0 - threshold) * (y1 - threshold) <= 0.0 and y0 != y1:
            fraction = (threshold - y0) / (y1 - y0)
            return float(x[first] + fraction * (x[second] - x[first]))
    return None


def baseline_fwhm(
    coordinate: np.ndarray,
    values: np.ndarray,
    center_mm: float,
    pitch_mm: float,
    baseline: float,
) -> float | None:
    selected = np.abs(coordinate - center_mm) <= 0.5 * pitch_mm
    local_x = coordinate[selected]
    local_y = values[selected]
    if local_x.size < 3:
        return None
    peak_index = int(np.argmax(local_y))
    threshold = baseline + 0.5 * (float(local_y[peak_index]) - baseline)
    left = crossing(
        local_x[: peak_index + 1], local_y[: peak_index + 1], threshold, True
    )
    right = crossing(local_x[peak_index:], local_y[peak_index:], threshold, False)
    return None if left is None or right is None else right - left


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def analyze_depth(
    reference: np.ndarray,
    evaluation: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    depth_mm: float,
    slab_width_mm: float,
    pitch_mm: float,
    slit_count: int,
    window_half_width_mm: float,
    minimum_peak_fraction: float,
    evaluation_scale: float = 1.0,
) -> dict[str, object]:
    ref_x, _, span = transverse_profiles(
        reference, z_mm, depth_mm, slab_width_mm
    )
    eval_x, _, _ = transverse_profiles(
        evaluation, z_mm, depth_mm, slab_width_mm
    )
    eval_x *= evaluation_scale
    centers = (
        np.arange(slit_count, dtype=np.float64) - 0.5 * (slit_count - 1)
    ) * pitch_mm
    candidate_ref_peaks = [
        window_mean(x_mm, ref_x, center, window_half_width_mm)
        for center in centers
    ]
    threshold = minimum_peak_fraction * max(candidate_ref_peaks)
    rows: list[dict[str, float]] = []
    for index in range(1, slit_count - 1):
        center = float(centers[index])
        ref_peak = candidate_ref_peaks[index]
        if ref_peak < threshold:
            continue
        eval_peak = window_mean(
            x_mm, eval_x, center, window_half_width_mm
        )
        left_valley = center - 0.5 * pitch_mm
        right_valley = center + 0.5 * pitch_mm
        ref_valley = 0.5 * (
            window_mean(x_mm, ref_x, left_valley, window_half_width_mm)
            + window_mean(x_mm, ref_x, right_valley, window_half_width_mm)
        )
        eval_valley = 0.5 * (
            window_mean(x_mm, eval_x, left_valley, window_half_width_mm)
            + window_mean(x_mm, eval_x, right_valley, window_half_width_mm)
        )
        ref_fwhm = baseline_fwhm(
            x_mm, ref_x, center, pitch_mm, ref_valley
        )
        eval_fwhm = baseline_fwhm(
            x_mm, eval_x, center, pitch_mm, eval_valley
        )
        row = {
            "center_mm": center,
            "reference_peak_Gy": ref_peak,
            "evaluation_peak_Gy": eval_peak,
            "peak_ratio": eval_peak / ref_peak,
            "reference_valley_Gy": ref_valley,
            "evaluation_valley_Gy": eval_valley,
            "valley_ratio": eval_valley / ref_valley,
            "reference_pvdr": ref_peak / ref_valley,
            "evaluation_pvdr": eval_peak / eval_valley,
            "pvdr_ratio": (eval_peak / eval_valley) / (ref_peak / ref_valley),
        }
        if ref_fwhm is not None and eval_fwhm is not None:
            row["reference_fwhm_mm"] = ref_fwhm
            row["evaluation_fwhm_mm"] = eval_fwhm
            row["fwhm_difference_mm"] = eval_fwhm - ref_fwhm
        rows.append(row)

    result: dict[str, object] = {
        "requested_depth_mm": depth_mm,
        "sampled_depth_span_mm": list(span),
        "slits_in_summary": len(rows),
        "per_slit": rows,
    }
    for name in ("peak_ratio", "valley_ratio", "pvdr_ratio"):
        result[name] = summarize([float(row[name]) for row in rows])
    fwhm_differences = [
        float(row["fwhm_difference_mm"])
        for row in rows
        if "fwhm_difference_mm" in row
    ]
    if fwhm_differences:
        result["fwhm_difference_mm"] = summarize(fwhm_differences)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--depths-mm", type=float, nargs="+", default=[0.5, 35.0, 70.35]
    )
    parser.add_argument("--slab-width-mm", type=float, default=1.0)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    parser.add_argument("--slit-count", type=int, default=15)
    parser.add_argument("--window-half-width-mm", type=float, default=0.15)
    parser.add_argument("--minimum-peak-fraction", type=float, default=0.05)
    parser.add_argument(
        "--evaluation-scale",
        type=float,
        default=1.0,
        help="Prescribed history-count scale applied to evaluation dose",
    )
    args = parser.parse_args()

    reference, x_ref, _, z_ref = load_mhd(args.reference)
    evaluation, x_eval, _, z_eval = load_mhd(args.evaluation)
    if reference.shape != evaluation.shape:
        raise ValueError("Reference and evaluation dose shapes differ")
    if not np.allclose(x_ref, x_eval) or not np.allclose(z_ref, z_eval):
        raise ValueError("Reference and evaluation coordinates differ")

    result = {
        "normalization": "absolute dose; no fitted scale",
        "reference": args.reference.as_posix(),
        "evaluation": args.evaluation.as_posix(),
        "definitions": {
            "peak_and_valley_window_half_width_mm": args.window_half_width_mm,
            "fwhm": "half height above the mean of adjacent valleys",
            "minimum_reference_peak_fraction": args.minimum_peak_fraction,
        },
        "depths": [
            analyze_depth(
                reference,
                evaluation,
                x_ref,
                z_ref,
                depth,
                args.slab_width_mm,
                args.pitch_mm,
                args.slit_count,
                args.window_half_width_mm,
                args.minimum_peak_fraction,
                args.evaluation_scale,
            )
            for depth in args.depths_mm
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
