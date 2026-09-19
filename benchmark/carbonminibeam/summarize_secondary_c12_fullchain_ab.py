#!/usr/bin/env python3
"""Summarize staged secondary-C12 full-chain A/B/C/D runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CASES = (
    "A_formal_em_scale1_highland",
    "B_unified_scale1_highland",
    "C_unified_scale09958_highland",
    "D_unified_scale09958_fe",
)
WATER_DESCENDANT_PROXY = (
    "water_p", "water_d", "water_t", "water_he3", "water_he4",
    "water_heavy", "water_other",
)


def grid(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4").astype(np.float64)
    if values.size != 1_000_000:
        raise ValueError(f"expected 1,000,000 FP32 voxels in {path}")
    return values.reshape(1000, 1000)


def relative_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(right - left).sum() / np.abs(left).sum())


def ratio(left: np.ndarray, right: np.ndarray) -> float:
    return float(right.sum() / left.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    selected_cases = tuple(args.cases)
    pairs = tuple(zip(selected_cases, selected_cases[1:]))

    arrays: dict[str, dict[str, np.ndarray]] = {}
    result: dict = {
        "scope": (
            "water-born non-C12 is a descendant-sensitive difference, not an "
            "absolute C12-ancestry scorer"
        ),
        "cases": {},
        "pairs": {},
    }
    for name in selected_cases:
        case_dir = args.root / name
        total = grid(case_dir / "dose.raw")
        secondary_c12 = (
            grid(case_dir / "component_component_copper_secondary_c12.raw") +
            grid(case_dir / "component_component_water_secondary_c12.raw")
        )
        water_descendant_proxy = sum(
            (grid(case_dir / f"component_component_{label}.raw")
             for label in WATER_DESCENDANT_PROXY),
            np.zeros_like(total),
        )
        arrays[name] = {
            "total": total,
            "secondary_c12": secondary_c12,
            "water_descendant_proxy": water_descendant_proxy,
        }
        dose_metrics = json.loads(
            (case_dir / "dose_comparison" / "metrics.json").read_text()
        )
        component_metrics = json.loads(
            (case_dir / "component_analysis" / "metrics.json").read_text()
        )
        topas_bragg = component_metrics["bragg_reference_depth_mm"]
        bragg_index = int(round((topas_bragg - 0.125) / 0.25))
        bragg_slice = slice(max(0, bragg_index - 8), min(1000, bragg_index + 9))
        local = min(
            dose_metrics["selected_depths"],
            key=lambda row: abs(row["grid_depth_mm"] - topas_bragg),
        )
        result["cases"][name] = {
            "gpu_over_topas_total": dose_metrics["gpu_over_topas_dose_sum"],
            "two_dimensional_l1": dose_metrics["two_dimensional_l1_over_topas"],
            "idd_l1": dose_metrics["idd_l1_over_topas"],
            "lateral_l1": dose_metrics["lateral_integral_l1_over_topas"],
            "gpu_bragg_depth_mm": dose_metrics["bragg_peak_depth_mm"]["gpu"],
            "local_bragg_peak_ratio": local["peak_gpu_over_topas"],
            "local_bragg_valley_ratio": local["valley_gpu_over_topas"],
            "local_bragg_pvdr_ratio": local["pvdr_gpu_over_topas"],
            "fixed_bragg_roi": {
                roi: component_metrics["fixed_regions_at_bragg"][roi][
                    "gpu_over_topas_total"
                ]
                for roi in ("peak", "shoulder", "valley")
            },
            "secondary_c12_fraction": float(secondary_c12.sum() / total.sum()),
            "water_descendant_proxy_fraction": float(
                water_descendant_proxy.sum() / total.sum()
            ),
            "bragg_pm2mm_secondary_c12_fraction": float(
                secondary_c12[bragg_slice].sum() / total[bragg_slice].sum()
            ),
            "bragg_pm2mm_water_descendant_proxy_fraction": float(
                water_descendant_proxy[bragg_slice].sum() /
                total[bragg_slice].sum()
            ),
        }

    for left_name, right_name in pairs:
        left = arrays[left_name]
        right = arrays[right_name]
        pair = {}
        for component in left:
            pair[component] = {
                "right_over_left": ratio(left[component], right[component]),
                "l1_over_left": relative_l1(left[component], right[component]),
            }
        result["pairs"][f"{left_name}__to__{right_name}"] = pair

    output = args.root / "ab_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
