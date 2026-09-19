#!/usr/bin/env python3
"""Analyze TOPAS minibeam dose carried by delta electrons and descendants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def parse_topas_header(path: Path) -> dict[str, tuple[int, float]]:
    axes: dict[str, tuple[int, float]] = {}
    pattern = re.compile(
        r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins?\s+of\s+"
        r"([0-9.eE+-]+)\s+cm\s*$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            axes[match.group(1)] = (
                int(match.group(2)), float(match.group(3)) * 10.0)
    if "X" not in axes or "Y" not in axes:
        raise ValueError(f"Missing TOPAS X/Y grid metadata in {path}")
    return axes


def assert_close(name: str, left: float, right: float) -> None:
    if not np.isclose(left, right, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(f"{name} mismatch: {left} versus {right}")


def load(path: Path, depth_bins: int, lateral_bins: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    expected = depth_bins * lateral_bins
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {values.size}")
    return values.reshape(depth_bins, lateral_bins)


def feature_curves(dose: np.ndarray, x_mm: np.ndarray, pitch_mm: float,
                   slab_bins: int, reference: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    smoothed = uniform_filter1d(dose, size=slab_bins, axis=0, mode="nearest")
    selection = smoothed if reference is None else uniform_filter1d(
        reference, size=slab_bins, axis=0, mode="nearest")
    peaks, valleys = [], []
    rows = np.arange(dose.shape[0])
    for expected in np.arange(-4, 5, dtype=float) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= 0.7)
        chosen = indices[np.argmax(selection[:, indices], axis=1)]
        peaks.append(smoothed[rows, chosen])
    for expected in (np.arange(-4, 4, dtype=float) + 0.5) * pitch_mm:
        indices = np.flatnonzero(np.abs(x_mm - expected) <= 0.7)
        chosen = indices[np.argmin(selection[:, indices], axis=1)]
        valleys.append(smoothed[rows, chosen])
    return np.median(np.stack(peaks), axis=0), np.median(np.stack(valleys), axis=0)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topas-header", type=Path)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--spacing-mm", type=float,
                        help="Deprecated: asserts both lateral and depth spacing")
    parser.add_argument("--lateral-spacing-mm", type=float)
    parser.add_argument("--depth-spacing-mm", type=float)
    parser.add_argument("--slab-mm", type=float, default=1.0)
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header_path = args.topas_header or args.run_dir / "dose.binheader"
    grid = parse_topas_header(header_path)
    lateral_bins, lateral_spacing = grid["X"]
    depth_bins, depth_spacing = grid["Y"]
    if args.bins is not None and (
            lateral_bins != args.bins or depth_bins != args.bins):
        raise ValueError(
            f"--bins={args.bins} disagrees with TOPAS metadata "
            f"{depth_bins}x{lateral_bins}")
    if args.spacing_mm is not None:
        assert_close("legacy lateral spacing", args.spacing_mm, lateral_spacing)
        assert_close("legacy depth spacing", args.spacing_mm, depth_spacing)
    if args.lateral_spacing_mm is not None:
        assert_close("lateral spacing", args.lateral_spacing_mm, lateral_spacing)
    if args.depth_spacing_mm is not None:
        assert_close("depth spacing", args.depth_spacing_mm, depth_spacing)

    total = load(args.run_dir / "dose.bin", depth_bins, lateral_bins)
    electron = load(
        args.run_dir / "dose_electron_carrier.bin", depth_bins, lateral_bins)
    non_electron = load(
        args.run_dir / "dose_non_electron_carrier.bin", depth_bins, lateral_bins)
    partition = electron + non_electron
    total_sum = float(total.sum())
    closure_l1 = float(np.abs(partition - total).sum() / total_sum)

    depth_mm = (np.arange(depth_bins) + 0.5) * depth_spacing
    x_mm = ((np.arange(lateral_bins) + 0.5) * lateral_spacing -
            0.5 * lateral_bins * lateral_spacing)
    slab_bins = max(1, int(round(args.slab_mm / depth_spacing)))
    total_peak, total_valley = feature_curves(total, x_mm, args.pitch_mm, slab_bins)
    electron_peak, electron_valley = feature_curves(
        electron, x_mm, args.pitch_mm, slab_bins, total)
    non_electron_peak, non_electron_valley = feature_curves(
        non_electron, x_mm, args.pitch_mm, slab_bins, total)
    total_idd = total.sum(axis=1)
    electron_idd = electron.sum(axis=1)
    folded_x = (x_mm + 0.5 * args.pitch_mm) % args.pitch_mm - 0.5 * args.pitch_mm
    central = np.abs(x_mm) <= 18.0
    fixed_masks = {
        "peak": central & (np.abs(folded_x) < 0.25),
        "shoulder": central & (np.abs(folded_x) >= 0.25) &
                    (np.abs(folded_x) < 0.9),
        "valley": central & (np.abs(folded_x) >= 0.9) &
                  (np.abs(folded_x) <= 1.8),
    }
    total_smoothed = uniform_filter1d(
        total, size=slab_bins, axis=0, mode="nearest")
    electron_smoothed = uniform_filter1d(
        electron, size=slab_bins, axis=0, mode="nearest")
    fixed_electron_fractions = {
        name: safe_ratio(
            electron_smoothed[:, mask].sum(axis=1),
            total_smoothed[:, mask].sum(axis=1))
        for name, mask in fixed_masks.items()
    }

    bragg_mask = depth_mm >= 20.0
    bragg_index = np.flatnonzero(bragg_mask)[np.argmax(total_idd[bragg_mask])]
    selected_depths = [0.5, 5.0, 40.0, 60.0, 80.0, 100.0,
                       depth_mm[bragg_index]]
    selected = []
    for requested in selected_depths:
        index = int(np.argmin(np.abs(depth_mm - requested)))
        row = {
            "depth_mm": float(depth_mm[index]),
            "electron_total_fraction": float(electron_idd[index] / total_idd[index]),
            "electron_peak_fraction": float(electron_peak[index] / total_peak[index]),
            "electron_valley_fraction": float(electron_valley[index] / total_valley[index]),
            "total_pvdr": float(total_peak[index] / total_valley[index]),
            "electron_pvdr": float(electron_peak[index] / electron_valley[index]),
            "non_electron_pvdr": float(
                non_electron_peak[index] / non_electron_valley[index]),
            "electron_pvdr_over_total": float(
                electron_peak[index] * total_valley[index] /
                (electron_valley[index] * total_peak[index])),
        }
        for name, fractions in fixed_electron_fractions.items():
            row[f"fixed_{name}_electron_fraction"] = float(fractions[index])
        selected.append(row)
    summary = {
        "grid": {
            "shape_depth_lateral": [depth_bins, lateral_bins],
            "depth_spacing_mm": depth_spacing,
            "lateral_spacing_mm": lateral_spacing,
            "topas_header": str(header_path),
            "peak_valley_depth_smoothing_mm": args.slab_mm,
        },
        "carrier_partition_l1_over_total": closure_l1,
        "carrier_partition_sum_over_total": float(partition.sum() / total_sum),
        "electron_dose_fraction": float(electron.sum() / total_sum),
        "bragg_depth_mm": float(depth_mm[bragg_index]),
        "selected_depths": selected,
    }
    (args.output_dir / "electron_component_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    electron_fraction = safe_ratio(electron_idd, total_idd)
    electron_peak_fraction = safe_ratio(electron_peak, total_peak)
    electron_valley_fraction = safe_ratio(electron_valley, total_valley)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(depth_mm, electron_fraction, label="electron carrier")
    axes[0].set(xlabel="water depth [mm]", ylabel="fraction of total IDD",
                xlim=(0, min(100, depth_mm[-1])), ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(depth_mm, electron_peak_fraction, label="electron / peak")
    axes[1].plot(depth_mm, electron_valley_fraction, label="electron / valley")
    for name, fractions in fixed_electron_fractions.items():
        axes[1].plot(depth_mm, fractions, linestyle="--",
                     label=f"fixed {name}")
    axes[1].set(xlabel="water depth [mm]", ylabel="electron dose fraction",
                xlim=(0, min(100, depth_mm[-1])), ylim=(0, 1))
    axes[1].legend()
    fig.savefig(args.output_dir / "electron_component_depth_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
