#!/usr/bin/env python3
"""Summarize controlled primary/secondary C12 energy-loss path comparisons."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = (
    "primary_nostrag_scale1",
    "secondary_nostrag_scale1",
    "primary_strag_scale1",
    "primary_strag_scale09958",
    "secondary_strag_scale1",
    "secondary_strag_scale09958",
)
PAIRS = (
    ("primary_nostrag_scale1", "secondary_nostrag_scale1"),
    ("primary_strag_scale1", "secondary_strag_scale1"),
    ("primary_strag_scale09958", "secondary_strag_scale09958"),
    ("primary_strag_scale1", "primary_strag_scale09958"),
    ("secondary_strag_scale1", "secondary_strag_scale09958"),
)


def load_planes(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1, 3),
        dtype=np.dtype([("history", "<u4"), ("plane", "u1"), ("energy", "<f4")]),
    )
    return {
        plane: (
            values[values["plane"] == plane]["history"],
            values[values["plane"] == plane]["energy"].astype(np.float64),
        )
        for plane in range(5)
    }


def plane_summary(planes: dict[int, tuple[np.ndarray, np.ndarray]], histories: int) -> list[dict]:
    output = []
    for plane, (_, energy) in planes.items():
        output.append(
            {
                "plane": plane,
                "count": int(energy.size),
                "survival": float(energy.size / histories),
                "mean_MeV": float(np.mean(energy)),
                "std_MeV": float(np.std(energy)),
                "q05_MeV": float(np.quantile(energy, 0.05)),
                "q50_MeV": float(np.quantile(energy, 0.50)),
                "q95_MeV": float(np.quantile(energy, 0.95)),
                "q99_MeV": float(np.quantile(energy, 0.99)),
            }
        )
    return output


def paired_plane_summary(
    left: dict[int, tuple[np.ndarray, np.ndarray]],
    right: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    output = []
    for plane in range(5):
        left_ids, left_e = left[plane]
        right_ids, right_e = right[plane]
        common, left_index, right_index = np.intersect1d(
            left_ids, right_ids, assume_unique=True, return_indices=True
        )
        delta = right_e[right_index] - left_e[left_index]
        output.append(
            {
                "plane": plane,
                "paired_count": int(common.size),
                "right_minus_left_mean_MeV": float(np.mean(delta)),
                "right_minus_left_std_MeV": float(np.std(delta)),
                "right_minus_left_q05_MeV": float(np.quantile(delta, 0.05)),
                "right_minus_left_q95_MeV": float(np.quantile(delta, 0.95)),
            }
        )
    return output


def parse_mhd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    nx, ny, nz = (int(value) for value in metadata["DimSize"].split())
    spacing_z = float(metadata["ElementSpacing"].split()[2])
    offset_z = float(metadata["Offset"].split()[2])
    raw = np.fromfile(path.parent / metadata["ElementDataFile"], dtype="<f4")
    dose = raw.reshape(nz, ny, nx).astype(np.float64)
    depth = offset_z + np.arange(nz) * spacing_z
    return depth, dose.sum(axis=(1, 2))


def dose_summary(depth: np.ndarray, idd: np.ndarray) -> dict:
    maximum = float(np.max(idd))
    peak_index = int(np.argmax(idd))
    distal = np.flatnonzero((np.arange(idd.size) > peak_index) & (idd <= 0.8 * maximum))
    r80 = float(depth[distal[0]]) if distal.size else None
    return {
        "total": float(np.sum(idd)),
        "bragg_depth_mm": float(depth[peak_index]),
        "r80_grid_mm": r80,
    }


def log_audit(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"\[minibeam-water-secondary-c12-post-sample-loss\] "
        r"raw_MeV=([0-9.eE+-]+) scaled_MeV=([0-9.eE+-]+) steps=([0-9]+)",
        text,
    )
    return (
        {
            "raw_MeV": float(match.group(1)),
            "scaled_MeV": float(match.group(2)),
            "steps": int(match.group(3)),
            "observed_ratio": float(match.group(2)) / float(match.group(1)),
        }
        if match
        else {}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=1_622_795)
    args = parser.parse_args()

    plane_data = {}
    dose_data = {}
    result = {"histories": args.histories, "expected_scale_variance_ratio": 0.9958**2}
    result["cases"] = {}
    for case in CASES:
        case_dir = args.root / case
        plane_data[case] = load_planes(case_dir / "primary_planes.csv")
        depth, idd = parse_mhd(case_dir / "dose.mhd")
        dose_data[case] = (depth, idd)
        result["cases"][case] = {
            "planes": plane_summary(plane_data[case], args.histories),
            "dose": dose_summary(depth, idd),
            "loss_audit": log_audit(case_dir / "run.log"),
        }

    result["pairs"] = {}
    for left, right in PAIRS:
        key = f"{left}__vs__{right}"
        left_depth, left_idd = dose_data[left]
        right_depth, right_idd = dose_data[right]
        if not np.array_equal(left_depth, right_depth):
            raise ValueError(f"dose grids differ for {left} and {right}")
        denominator = np.sum(np.abs(left_idd))
        result["pairs"][key] = {
            "planes": paired_plane_summary(plane_data[left], plane_data[right]),
            "idd_l1_over_left": float(np.sum(np.abs(right_idd - left_idd)) / denominator),
            "total_ratio": float(np.sum(right_idd) / np.sum(left_idd)),
        }

    output_dir = args.root / "summary"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "energy_path_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for case in CASES:
        depth, idd = dose_data[case]
        axes[0].plot(depth, idd / np.max(idd), label=case)
        means = [entry["mean_MeV"] for entry in result["cases"][case]["planes"]]
        axes[1].plot((40, 60, 80, 100, 120), means, marker="o", label=case)
    axes[0].set(xlim=(110, 135), xlabel="depth [mm]", ylabel="normalized IDD")
    axes[1].set(xlabel="plane depth [mm]", ylabel="mean C12 kinetic energy [MeV]")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "energy_path_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
