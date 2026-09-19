#!/usr/bin/env python3
"""Summarize the controlled species-FE/secondary-EM minibeam A/B.

The script deliberately reuses the canonical fixed-region masks from the dose
comparison.  It reports the requested entrance/40/120 mm/Bragg planes and the
GPU component fractions at those same planes.  All arrays are absolute Gy;
TOPAS is scaled only by the incident-history ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from analyze_minibeam_source_components import GPU_LABELS
from compare_gpu_topas_dose import fixed_minibeam_region_masks, parse_mhd


VARIANTS = (
    "A_shared_fe_legacy_em",
    "B_species_fe_legacy_em",
    "C_species_fe_unified_em",
)
GROUP_MEMBERS = {
    "primary_c12": ("primary_c12",),
    "secondary_c12": ("copper_secondary_c12", "water_secondary_c12"),
    "p": ("copper_p", "water_p"),
    "d": ("copper_d", "water_d"),
    "t": ("copper_t", "water_t"),
    "he3": ("copper_he3", "water_he3"),
    "he4": ("copper_he4", "water_he4"),
    "heavy": ("copper_heavy", "water_heavy"),
    "other": ("copper_other", "water_other"),
    "copper_born": tuple(x for x in GPU_LABELS if x.startswith("copper_")),
    "water_born": tuple(x for x in GPU_LABELS if x.startswith("water_")),
}


def read_grid(path: Path, dtype: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype).astype(np.float64)
    if values.size != shape[0] * shape[1]:
        raise ValueError(f"unexpected element count in {path}: {values.size}")
    return values.reshape(shape)


def ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topas-root", type=Path, required=True)
    parser.add_argument("--energies", type=int, nargs="+", default=(250, 300))
    parser.add_argument("--gpu-histories", type=int, default=256000)
    parser.add_argument("--topas-histories", type=int, default=10000640)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "species_fe_minibeam_ab_summary.json"
    records: list[dict] = []
    component_rows: list[dict] = []

    for energy in args.energies:
        topas_path = args.topas_root / f"e{energy}/dose.bin"
        for variant in VARIANTS:
            directory = args.root / f"e{energy}" / variant
            mhd = parse_mhd(directory / "out/config/dose.mhd")
            dims = mhd["dimensions"]
            spacing = mhd["spacing_mm"]
            offset = mhd["offset_mm"]
            assert isinstance(dims, list) and isinstance(spacing, list)
            assert isinstance(offset, list)
            shape = (int(dims[2]), int(dims[0]))
            gpu = read_grid(directory / "out/config/dose.raw", "<f4", shape)
            topas = read_grid(topas_path, "<f8", shape)
            topas *= args.gpu_histories / args.topas_histories
            depth = offset[2] + np.arange(shape[0]) * spacing[2]
            lateral = offset[0] + np.arange(shape[1]) * spacing[0]
            masks = fixed_minibeam_region_masks(lateral, 3.6)
            slab_bins = max(1, int(round(1.0 / spacing[2])))
            gpu_s = uniform_filter1d(gpu, slab_bins, axis=0, mode="nearest")
            topas_s = uniform_filter1d(topas, slab_bins, axis=0, mode="nearest")
            search = depth >= 20.0
            bragg_i = np.flatnonzero(search)[np.argmax(topas.sum(axis=1)[search])]
            requested = (0.5, 40.0, 120.0, float(depth[bragg_i]))

            raw_components = {
                label: read_grid(
                    directory / f"component_component_{label}.raw", "<f4", shape)
                for label in GPU_LABELS
            }
            groups = {
                name: sum(raw_components[label] for label in labels)
                for name, labels in GROUP_MEMBERS.items()
            }
            smooth_groups = {
                name: uniform_filter1d(values, slab_bins, axis=0, mode="nearest")
                for name, values in groups.items()
            }
            global_fractions = {
                name: float(values.sum() / gpu.sum())
                for name, values in groups.items()
            }
            depths = []
            for requested_depth in requested:
                iz = int(np.argmin(np.abs(depth - requested_depth)))
                entry = {
                    "requested_depth_mm": requested_depth,
                    "grid_depth_mm": float(depth[iz]),
                    "regions": {},
                }
                for region, mask in masks.items():
                    gpu_value = float(gpu_s[iz, mask].sum())
                    topas_value = float(topas_s[iz, mask].sum())
                    fractions = {
                        name: ratio(float(values[iz, mask].sum()), gpu_value)
                        for name, values in smooth_groups.items()
                    }
                    entry["regions"][region] = {
                        "gpu_Gy": gpu_value,
                        "topas_Gy": topas_value,
                        "gpu_over_topas": ratio(gpu_value, topas_value),
                        "gpu_component_fraction": fractions,
                    }
                    for name, fraction in fractions.items():
                        component_rows.append({
                            "energy_MeVu": energy,
                            "variant": variant,
                            "depth_mm": float(depth[iz]),
                            "region": region,
                            "component": name,
                            "gpu_fraction": fraction,
                            "gpu_component_Gy": (
                                None if fraction is None else fraction * gpu_value),
                            "gpu_total_Gy": gpu_value,
                            "topas_total_Gy": topas_value,
                        })
                depths.append(entry)
            records.append({
                "energy_MeVu": energy,
                "variant": variant,
                "global_component_fraction": global_fractions,
                "depths": depths,
            })

    comparisons = []
    by_key = {(r["energy_MeVu"], r["variant"]): r for r in records}
    for energy in args.energies:
        for left_name, right_name, meaning in (
                (VARIANTS[0], VARIANTS[1], "species FE parameters only"),
                (VARIANTS[1], VARIANTS[2], "secondary Unified EM only")):
            left = by_key[(energy, left_name)]
            right = by_key[(energy, right_name)]
            depth_changes = []
            for left_depth, right_depth in zip(left["depths"], right["depths"]):
                changes = {}
                for region in ("peak", "shoulder", "valley"):
                    lreg = left_depth["regions"][region]
                    rreg = right_depth["regions"][region]
                    changes[region] = {
                        "gpu_dose_relative_change": (
                            rreg["gpu_Gy"] / lreg["gpu_Gy"] - 1.0),
                        "gpu_over_topas_change": (
                            rreg["gpu_over_topas"] - lreg["gpu_over_topas"]),
                        "component_Gy_change": {
                            name: (rreg["gpu_component_fraction"][name] *
                                   rreg["gpu_Gy"] -
                                   lreg["gpu_component_fraction"][name] *
                                   lreg["gpu_Gy"])
                            for name in GROUP_MEMBERS
                        },
                    }
                depth_changes.append({
                    "depth_mm": left_depth["grid_depth_mm"],
                    "regions": changes,
                })
            comparisons.append({
                "energy_MeVu": energy,
                "left": left_name,
                "right": right_name,
                "isolated_change": meaning,
                "depth_changes": depth_changes,
            })

    payload = {
        "normalization": (
            "absolute Gy; TOPAS scaled only by 256000/10000640 histories; "
            "1 mm depth smoothing; canonical fixed lateral masks"),
        "records": records,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=component_rows[0].keys())
        writer.writeheader()
        writer.writerows(component_rows)
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
