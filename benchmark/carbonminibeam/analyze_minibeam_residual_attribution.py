#!/usr/bin/env python3
"""Build a TOPAS/GPU minibeam residual attribution table.

Taxonomies are reported separately and must not be added together.

Comparable origin/Z classes pair TOPAS ChargedOriginDoseToMedium nuclear
categories with GPU charged-origin maps. TOPAS electrons inherit a charged
ancestor; GPU condensed electrons stay with the transporting ion. That is a
production-cut mapping, not identity of carrier labels.

Unmatched TOPAS classes (neutral_origin, unclassified) have no GPU scorer.
Their residual is a missing-correspondence term, not evidence that GPU omitted
physical dose.

Raw component closure is the engine's own category sum versus its own total.
It is not an independent proof that the cross-engine origin mapping is
correct, and a constructed GPU remainder is never used as an acceptance
metric.

Uncertainty is unknown unless independent history-level moments or shards are
supplied. ROI dose variance cannot be inferred from the incident-history
count alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from compare_gpu_topas_dose import fixed_minibeam_region_masks, parse_mhd


COMPARABLE_ORIGIN = (
    ("primary_c", "component_primary_c.bin", "primary.raw"),
    ("secondary_c", "component_secondary_c.bin", "secondary_carbon.raw"),
    ("boron", "component_boron.bin", "secondary_boron.raw"),
    ("beryllium", "component_beryllium.bin", "secondary_beryllium.raw"),
    ("lithium", "component_lithium.bin", "secondary_lithium.raw"),
    ("helium", "component_helium.bin", "secondary_helium.raw"),
    ("z1", "component_z1.bin", "secondary_proton.raw"),
    ("other_charged", "component_other_charged.bin", "secondary_other_charged.raw"),
)
UNMATCHED_TOPAS_ORIGIN = (
    ("neutral_origin", "component_neutral_origin.bin"),
    ("unclassified", "component_unclassified.bin"),
)
GPU_SPECIES_GROUPS = {
    "primary_c12": ("primary_c12",),
    "secondary_c12": ("copper_secondary_c12", "water_secondary_c12"),
    "p": ("copper_p", "water_p"),
    "d": ("copper_d", "water_d"),
    "t": ("copper_t", "water_t"),
    "he3": ("copper_he3", "water_he3"),
    "he4": ("copper_he4", "water_he4"),
    "heavy": ("copper_heavy", "water_heavy"),
    "other": ("copper_other", "water_other"),
}
GPU_SPECIES_LABELS = (
    "primary_c12",
    "copper_secondary_c12", "copper_p", "copper_d", "copper_t",
    "copper_he3", "copper_he4", "copper_heavy", "copper_other",
    "water_secondary_c12", "water_p", "water_d", "water_t",
    "water_he3", "water_he4", "water_heavy", "water_other",
)
PLANE_DEPTHS_MM = (0.5, 40.0, 80.0, 120.0)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def read_grid(path: Path, dtype: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype).astype(np.float64)
    if values.size != shape[0] * shape[1]:
        raise ValueError(f"{path}: expected {shape[0] * shape[1]}, got {values.size}")
    return values.reshape(shape)


def detect_gpu_prefix(gpu_dir: Path) -> str:
    if (gpu_dir / "component_primary.raw").is_file():
        return "component"
    if (gpu_dir / "components_primary.raw").is_file():
        return "components"
    raise FileNotFoundError(f"no charged-origin primary map in {gpu_dir}")


def smooth_depth(grid: np.ndarray, slab_bins: int) -> np.ndarray:
    return uniform_filter1d(grid, size=slab_bins, axis=0, mode="nearest")


def roi_sum(grid: np.ndarray, depth_index: np.ndarray | int,
            lateral_mask: np.ndarray) -> float:
    if np.isscalar(depth_index) or (
            isinstance(depth_index, np.ndarray) and depth_index.ndim == 0):
        return float(grid[int(depth_index), lateral_mask].sum())
    return float(grid[depth_index][:, lateral_mask].sum())


def category_entry(gpu: float | None, topas: float | None,
                   topas_total: float) -> dict:
    gpu_value = 0.0 if gpu is None else gpu
    topas_value = 0.0 if topas is None else topas
    comparable = gpu is not None and topas is not None
    return {
        "gpu_Gy": None if gpu is None else gpu,
        "topas_Gy": None if topas is None else topas,
        "gpu_fraction": None,
        "topas_fraction_of_total": (
            None if topas is None else ratio(topas_value, topas_total)),
        "gpu_over_topas": None if not comparable else ratio(gpu_value, topas_value),
        "delta_over_topas_total": (
            None if not comparable or topas_total == 0.0
            else (gpu_value - topas_value) / topas_total),
        "uncertainty": "unknown",
    }


def fill_gpu_fraction(entry: dict, gpu_total: float) -> dict:
    if entry["gpu_Gy"] is not None:
        entry["gpu_fraction"] = ratio(entry["gpu_Gy"], gpu_total)
    return entry


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dose", type=Path)
    parser.add_argument("--gpu-mhd", type=Path)
    parser.add_argument("--gpu-histories", type=int, required=True)
    parser.add_argument("--topas-dir", type=Path, required=True)
    parser.add_argument("--topas-histories", type=int, required=True)
    parser.add_argument("--topas-electron-dir", type=Path)
    parser.add_argument("--topas-electron-histories", type=int)
    parser.add_argument("--bragg-depth-mm", type=float)
    parser.add_argument("--label", default="")
    parser.add_argument("--role", default="current",
                        help="current candidate versus auxiliary historical evidence")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_gpu_dose(args: argparse.Namespace) -> tuple[Path, Path]:
    gpu_dose_path = args.gpu_dose or args.gpu_dir / "out/config/dose.raw"
    if not gpu_dose_path.is_file():
        gpu_dose_path = args.gpu_dir / "dose.raw"
    gpu_mhd_path = args.gpu_mhd or gpu_dose_path.with_suffix(".mhd")
    if not gpu_mhd_path.is_file():
        gpu_mhd_path = args.gpu_dir / "out/config/dose.mhd"
    return gpu_dose_path, gpu_mhd_path


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_dose_path, gpu_mhd_path = resolve_gpu_dose(args)
    mhd = parse_mhd(gpu_mhd_path)
    dims, spacing, offset = mhd["dimensions"], mhd["spacing_mm"], mhd["offset_mm"]
    assert isinstance(dims, list) and isinstance(spacing, list)
    assert isinstance(offset, list)
    shape = (int(dims[2]), int(dims[0]))
    depth = offset[2] + np.arange(shape[0]) * spacing[2]
    lateral = offset[0] + np.arange(shape[1]) * spacing[0]
    masks = fixed_minibeam_region_masks(lateral, 3.6)
    slab_bins = max(1, int(round(1.0 / spacing[2])))
    scale = args.gpu_histories / args.topas_histories

    gpu_total = read_grid(gpu_dose_path, "<f4", shape)
    topas_total = read_grid(args.topas_dir / "dose.bin", "<f8", shape) * scale
    gpu_s = smooth_depth(gpu_total, slab_bins)
    topas_s = smooth_depth(topas_total, slab_bins)
    search = depth >= 20.0
    bragg_i = np.flatnonzero(search)[np.argmax(topas_total.sum(axis=1)[search])]
    bragg_depth = (args.bragg_depth_mm if args.bragg_depth_mm is not None
                   else float(depth[bragg_i]))
    bragg_window = np.abs(depth - bragg_depth) <= 2.0

    prefix = detect_gpu_prefix(args.gpu_dir)
    gpu_comparable = {
        name: read_grid(args.gpu_dir / f"{prefix}_{gpu_suffix}", "<f4", shape)
        for name, _topas, gpu_suffix in COMPARABLE_ORIGIN
    }
    topas_comparable = {
        name: read_grid(args.topas_dir / topas_name, "<f8", shape) * scale
        for name, topas_name, _gpu in COMPARABLE_ORIGIN
    }
    topas_unmatched = {
        name: read_grid(args.topas_dir / filename, "<f8", shape) * scale
        for name, filename in UNMATCHED_TOPAS_ORIGIN
    }
    gpu_species_raw = {
        label: read_grid(
            args.gpu_dir / f"{prefix}_component_{label}.raw", "<f4", shape)
        for label in GPU_SPECIES_LABELS
    }
    gpu_species = {
        name: sum(gpu_species_raw[label] for label in labels)
        for name, labels in GPU_SPECIES_GROUPS.items()
    }
    gpu_species["z1"] = gpu_species["p"] + gpu_species["d"] + gpu_species["t"]
    gpu_species["helium"] = gpu_species["he3"] + gpu_species["he4"]
    gpu_species["z_ge_3_secondary"] = (
        gpu_species["secondary_c12"] + gpu_species["heavy"])

    gpu_comparable_s = {k: smooth_depth(v, slab_bins) for k, v in gpu_comparable.items()}
    topas_comparable_s = {k: smooth_depth(v, slab_bins) for k, v in topas_comparable.items()}
    topas_unmatched_s = {k: smooth_depth(v, slab_bins) for k, v in topas_unmatched.items()}
    gpu_species_s = {k: smooth_depth(v, slab_bins) for k, v in gpu_species.items()}

    gpu_origin_sum = sum(gpu_comparable.values())
    topas_origin_sum = sum(topas_comparable.values()) + sum(topas_unmatched.values())
    gpu_species_sum = sum(gpu_species_raw.values())

    planes = []
    for requested in PLANE_DEPTHS_MM + (bragg_depth,):
        iz = int(np.argmin(np.abs(depth - requested)))
        planes.append({
            "name": ("bragg" if np.isclose(requested, bragg_depth) and
                     requested not in PLANE_DEPTHS_MM else f"{requested:g}mm"),
            "requested_depth_mm": requested,
            "grid_depth_mm": float(depth[iz]),
            "depth_index": iz,
            "window": "1mm_smooth_plane",
            "use_smooth": True,
        })
    planes.append({
        "name": "bragg_pm2mm",
        "requested_depth_mm": bragg_depth,
        "grid_depth_mm": bragg_depth,
        "depth_index": bragg_window,
        "window": "bragg_pm2mm_unsmoothed",
        "use_smooth": False,
    })

    rows = []
    records = []
    for plane in planes:
        depth_index = plane["depth_index"]
        use_smooth = plane["use_smooth"]
        gpu_dose = gpu_s if use_smooth else gpu_total
        topas_dose = topas_s if use_smooth else topas_total
        gpu_cat = gpu_comparable_s if use_smooth else gpu_comparable
        topas_cat = topas_comparable_s if use_smooth else topas_comparable
        unmatched_cat = topas_unmatched_s if use_smooth else topas_unmatched
        species_cat = gpu_species_s if use_smooth else gpu_species
        for region, mask in masks.items():
            gpu_roi = roi_sum(gpu_dose, depth_index, mask)
            topas_roi = roi_sum(topas_dose, depth_index, mask)
            comparable_entries = {}
            unmatched_entries = {}
            sum_comparable_delta = 0.0
            for name, *_ in COMPARABLE_ORIGIN:
                entry = category_entry(
                    roi_sum(gpu_cat[name], depth_index, mask),
                    roi_sum(topas_cat[name], depth_index, mask),
                    topas_roi)
                fill_gpu_fraction(entry, gpu_roi)
                comparable_entries[name] = entry
                if entry["delta_over_topas_total"] is not None:
                    sum_comparable_delta += entry["delta_over_topas_total"]
            unmatched_topas_delta = 0.0
            for name, _filename in UNMATCHED_TOPAS_ORIGIN:
                topas_value = roi_sum(unmatched_cat[name], depth_index, mask)
                entry = {
                    "gpu_Gy": None,
                    "topas_Gy": topas_value,
                    "gpu_fraction": None,
                    "topas_fraction_of_total": ratio(topas_value, topas_roi),
                    "gpu_over_topas": None,
                    "delta_over_topas_total": None,
                    "missing_scorer_residual_over_topas_total": (
                        None if topas_roi == 0.0 else -topas_value / topas_roi),
                    "uncertainty": "unknown",
                    "note": (
                        "TOPAS-only origin class. GPU has no corresponding "
                        "scorer; this is a missing-correspondence residual, "
                        "not GPU missing physical dose."),
                }
                unmatched_entries[name] = entry
                if topas_roi:
                    unmatched_topas_delta -= topas_value / topas_roi
            species_entries = {
                name: {
                    "gpu_Gy": roi_sum(grid, depth_index, mask),
                    "gpu_fraction": ratio(
                        roi_sum(grid, depth_index, mask), gpu_roi),
                }
                for name, grid in species_cat.items()
            }
            total_delta = ratio(gpu_roi - topas_roi, topas_roi)
            ranked = sorted(
                ((name, comparable_entries[name]["delta_over_topas_total"] or 0.0)
                 for name, *_ in COMPARABLE_ORIGIN),
                key=lambda item: abs(item[1]), reverse=True)
            record = {
                "plane": plane["name"],
                "window": plane["window"],
                "requested_depth_mm": plane["requested_depth_mm"],
                "grid_depth_mm": plane["grid_depth_mm"],
                "region": region,
                "gpu_total_Gy": gpu_roi,
                "topas_total_Gy": topas_roi,
                "gpu_over_topas_total": ratio(gpu_roi, topas_roi),
                "total_delta_over_topas_total": total_delta,
                "sum_comparable_delta_over_topas_total": sum_comparable_delta,
                "unmatched_topas_missing_scorer_residual_over_topas_total":
                    unmatched_topas_delta,
                "uncertainty": "unknown",
                "comparable_origin_categories": comparable_entries,
                "unmatched_topas_origin_categories": unmatched_entries,
                "gpu_depositing_species": species_entries,
                "largest_comparable_origin_residual": ranked[0][0] if ranked else None,
            }
            records.append(record)
            for name, entry in comparable_entries.items():
                rows.append({
                    "label": args.label,
                    "role": args.role,
                    "plane": plane["name"],
                    "window": plane["window"],
                    "depth_mm": plane["grid_depth_mm"],
                    "region": region,
                    "category": name,
                    "class": "comparable_origin",
                    "gpu_Gy": entry["gpu_Gy"],
                    "topas_Gy": entry["topas_Gy"],
                    "gpu_fraction": entry["gpu_fraction"],
                    "topas_fraction": entry["topas_fraction_of_total"],
                    "gpu_over_topas": entry["gpu_over_topas"],
                    "delta_over_topas_total": entry["delta_over_topas_total"],
                    "uncertainty": "unknown",
                })
            for name, entry in unmatched_entries.items():
                rows.append({
                    "label": args.label,
                    "role": args.role,
                    "plane": plane["name"],
                    "window": plane["window"],
                    "depth_mm": plane["grid_depth_mm"],
                    "region": region,
                    "category": name,
                    "class": "unmatched_topas_origin",
                    "gpu_Gy": None,
                    "topas_Gy": entry["topas_Gy"],
                    "gpu_fraction": None,
                    "topas_fraction": entry["topas_fraction_of_total"],
                    "gpu_over_topas": None,
                    "delta_over_topas_total": None,
                    "missing_scorer_residual_over_topas_total":
                        entry["missing_scorer_residual_over_topas_total"],
                    "uncertainty": "unknown",
                })

    electron = None
    if args.topas_electron_dir is not None:
        e_hist = args.topas_electron_histories or args.topas_histories
        e_scale = args.gpu_histories / e_hist
        e_total = read_grid(
            args.topas_electron_dir / "dose.bin", "<f8", shape) * e_scale
        e_electron = read_grid(
            args.topas_electron_dir / "dose_electron_carrier.bin",
            "<f8", shape) * e_scale
        e_ion = read_grid(
            args.topas_electron_dir / "dose_non_electron_carrier.bin",
            "<f8", shape) * e_scale
        e_total_s = smooth_depth(e_total, slab_bins)
        e_electron_s = smooth_depth(e_electron, slab_bins)
        e_ion_s = smooth_depth(e_ion, slab_bins)
        electron = []
        for plane in planes:
            depth_index = plane["depth_index"]
            use_smooth = plane["use_smooth"]
            total_g = e_total_s if use_smooth else e_total
            electron_g = e_electron_s if use_smooth else e_electron
            ion_g = e_ion_s if use_smooth else e_ion
            for region, mask in masks.items():
                total = roi_sum(total_g, depth_index, mask)
                electron_value = roi_sum(electron_g, depth_index, mask)
                ion_value = roi_sum(ion_g, depth_index, mask)
                partition_ratio = ratio(electron_value + ion_value, total)
                electron.append({
                    "plane": plane["name"],
                    "window": plane["window"],
                    "region": region,
                    "topas_total_Gy": total,
                    "electron_carrier_Gy": electron_value,
                    "non_electron_carrier_Gy": ion_value,
                    "electron_fraction": ratio(electron_value, total),
                    "carrier_partition_over_total": partition_ratio,
                    "carrier_closure_minus_one": (
                        None if partition_ratio is None else partition_ratio - 1.0),
                    "uncertainty": "unknown",
                    "note": (
                        "TOPAS carrier split only. GPU condensed electrons "
                        "remain inside the transporting ion origin class and "
                        "must be compared through production-cut mapping, "
                        "not unmapped carrier equality."),
                })

    payload = {
        "label": args.label,
        "role": args.role,
        "taxonomy_note": (
            "Comparable classes pair TOPAS origin of charged nuclei with GPU "
            "depositing-ion Z maps, including condensed electrons. "
            "neutral_origin and unclassified are unmatched TOPAS scorers. "
            "GPU other_charged is a depositing-ion Z class and is not the "
            "same object as TOPAS neutral-ancestor attribution."),
        "secondary_energy_loss_closure_scope": (
            "Secondary stopping/straggling closure currently covers p/d/t/"
            "He-4 at 50/150/300 MeV/u on the specified water slabs with the "
            "Unified EM path. It does not cover the <50 MeV/u fragments that "
            "dominate part of the Bragg fragment dose, nor other secondaries."),
        "normalization": (
            "absolute Gy after scaling TOPAS by gpu_histories/topas_histories; "
            "no fitted dose scale; no survivor-count renormalization"),
        "gpu_histories": args.gpu_histories,
        "topas_histories": args.topas_histories,
        "bragg_depth_mm": bragg_depth,
        "uncertainty": "unknown",
        "stat_error_model": (
            "No per-incident-history ROI moments or independent shards were "
            "supplied. Confidence intervals are unknown. abs(D)/sqrt(N) is "
            "not used."),
        "raw_component_closure": {
            "gpu_charged_origin_l1_over_gpu_total": float(
                np.abs(gpu_origin_sum - gpu_total).sum() / gpu_total.sum()),
            "gpu_species_l1_over_gpu_total": float(
                np.abs(gpu_species_sum - gpu_total).sum() / gpu_total.sum()),
            "topas_origin_l1_over_topas_total": float(
                np.abs(topas_origin_sum - topas_total).sum() / topas_total.sum()),
            "note": (
                "These are each engine's internal category-vs-total closure. "
                "They do not validate the cross-engine origin mapping."),
        },
        "records": records,
        "topas_carrier_split": electron,
    }
    (args.output_dir / "residual_attribution.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    fieldnames = [
        "label", "role", "plane", "window", "depth_mm", "region", "category",
        "class", "gpu_Gy", "topas_Gy", "gpu_fraction", "topas_fraction",
        "gpu_over_topas", "delta_over_topas_total",
        "missing_scorer_residual_over_topas_total", "uncertainty",
    ]
    with (args.output_dir / "residual_attribution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_dir / "residual_attribution.json")


if __name__ == "__main__":
    main()
