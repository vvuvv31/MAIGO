#!/usr/bin/env python3
"""Analyze diagnostic p/d/t/He-4 dose by transport-energy band and ROI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from compare_gpu_topas_dose import fixed_minibeam_region_masks, parse_mhd


SPECIES = ("p", "d", "t", "he4")
BANDS = ("lt50", "50to300", "gt300")
REGIONS = ("peak", "shoulder", "valley")
MEV_TO_JOULE = 1.602176634e-13


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bragg-depth-mm", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    csv_path = args.run_dir / "component_energy_band_roi.csv"
    mhd = parse_mhd(args.run_dir / "out/config/dose.mhd")
    dims, spacing, offset = (mhd["dimensions"], mhd["spacing_mm"],
                             mhd["offset_mm"])
    assert isinstance(dims, list) and isinstance(spacing, list)
    assert isinstance(offset, list)
    nz = int(dims[2])
    depth = offset[2] + np.arange(nz) * spacing[2]
    lateral = offset[0] + np.arange(int(dims[0])) * spacing[0]
    values = np.zeros((len(SPECIES), len(BANDS), len(REGIONS), nz))
    species_index = {name: i for i, name in enumerate(SPECIES)}
    band_index = {name: i for i, name in enumerate(BANDS)}
    region_index = {name: i for i, name in enumerate(REGIONS)}
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            iz = int(np.argmin(np.abs(depth - float(row["depth_mm"]))))
            values[species_index[row["species"]],
                   band_index[row["energy_band_MeV_per_u"]],
                   region_index[row["region"]], iz] = float(
                       row["deposited_energy_MeV"])
    slab_bins = max(1, int(round(1.0 / spacing[2])))
    smooth = uniform_filter1d(values, slab_bins, axis=-1, mode="nearest")
    shape = (nz, int(dims[0]))
    component_grids = []
    for species in SPECIES:
        component = sum(
            np.fromfile(
                args.run_dir /
                f"component_component_{origin}_{species}.raw",
                dtype="<f4").astype(np.float64).reshape(shape)
            for origin in ("copper", "water"))
        component_grids.append(
            uniform_filter1d(component, slab_bins, axis=0, mode="nearest"))
    masks = fixed_minibeam_region_masks(lateral, 3.6)
    targets = (0.5, 40.0, 80.0, 120.0, args.bragg_depth_mm)
    records = []
    component_factor = (MEV_TO_JOULE /
                        (spacing[0] * spacing[1] * spacing[2] * 1.0e-6))
    for target in targets:
        iz = int(np.argmin(np.abs(depth - target)))
        for ir, region in enumerate(REGIONS):
            per_species = {}
            combined = smooth[:, :, ir, iz].sum(axis=0)
            for ispecies, species in enumerate(SPECIES):
                band_values = smooth[ispecies, :, ir, iz]
                total = float(band_values.sum())
                per_species[species] = {
                    "deposited_energy_MeV": total,
                    "dose_sum_Gy": total * component_factor,
                    "band_fraction": {
                        band: (float(band_values[ib] / total) if total else 0.0)
                        for ib, band in enumerate(BANDS)
                    },
                }
            combined_total = float(combined.sum())
            component_dose = sum(
                float(grid[iz, masks[region]].sum())
                for grid in component_grids)
            scored_dose = combined_total * component_factor
            records.append({
                "requested_depth_mm": target,
                "grid_depth_mm": float(depth[iz]),
                "depth_window": "1mm_smooth_plane",
                "region": region,
                "species": per_species,
                "all_four_species_band_fraction": {
                    band: (float(combined[ib] / combined_total)
                           if combined_total else 0.0)
                    for ib, band in enumerate(BANDS)
                },
                "all_four_species_dose_sum_Gy": scored_dose,
                "component_map_dose_sum_Gy": component_dose,
                "energy_band_over_component_map": (
                    scored_dose / component_dose if component_dose else None),
            })
    depth_mask = np.abs(depth - args.bragg_depth_mm) <= 2.0
    for ir, region in enumerate(REGIONS):
        per_species = {}
        combined = smooth[:, :, ir, :][:, :, depth_mask].sum(axis=-1).sum(axis=0)
        for ispecies, species in enumerate(SPECIES):
            band_values = smooth[ispecies, :, ir, :][:, depth_mask].sum(axis=-1)
            total = float(band_values.sum())
            per_species[species] = {
                "deposited_energy_MeV": total,
                "dose_sum_Gy": total * component_factor,
                "band_fraction": {
                    band: (float(band_values[ib] / total) if total else 0.0)
                    for ib, band in enumerate(BANDS)
                },
            }
        combined_total = float(combined.sum())
        component_dose = sum(
            float(grid[depth_mask][:, masks[region]].sum())
            for grid in component_grids)
        scored_dose = combined_total * component_factor
        records.append({
            "requested_depth_mm": args.bragg_depth_mm,
            "grid_depth_mm": float(args.bragg_depth_mm),
            "depth_window": "bragg_pm2mm",
            "region": region,
            "species": per_species,
            "all_four_species_band_fraction": {
                band: (float(combined[ib] / combined_total)
                       if combined_total else 0.0)
                for ib, band in enumerate(BANDS)
            },
            "all_four_species_dose_sum_Gy": scored_dose,
            "component_map_dose_sum_Gy": component_dose,
            "energy_band_over_component_map": (
                scored_dose / component_dose if component_dose else None),
        })
    closure_errors = [
        abs(record["energy_band_over_component_map"] - 1.0)
        for record in records
        if record["energy_band_over_component_map"]
    ]
    payload = {
        "definition": (
            "Deposited energy is assigned by the depositing charged track's "
            "step-start kinetic energy per nucleon. Nuclear local deposit is "
            "assigned to the incident parent; below-cutoff product kinetic "
            "energy is assigned to the product. The CINEL03 "
            "process_local_deposit path must call the energy-band ROI scorer "
            "directly because the replay break skips the common-path commit."),
        "energy_bands_MeV_per_u": ["<50", "50-300 inclusive", ">300"],
        "depth_smoothing_mm": 1.0,
        "worst_abs_energy_band_over_component_map_minus_one": (
            max(closure_errors) if closure_errors else None),
        "records": records,
    }
    output = args.output or args.run_dir / "energy_band_roi_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
