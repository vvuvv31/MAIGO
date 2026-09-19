#!/usr/bin/env python3
"""Summarize the independent primary-C12 ROI diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from compare_gpu_topas_dose import parse_mhd


KINDS = (
    "local_total_deposit_MeV", "continuous_sampled_MeV", "delta_sampled_MeV",
    "continuous_after_scale_MeV", "delta_after_scale_MeV", "fluence_mm",
)
BANDS = ("lt50", "50to300", "gt300")
REGIONS = ("peak", "shoulder", "valley")
MEV_TO_JOULE = 1.602176634e-13


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bragg-depth-mm", type=float, required=True)
    parser.add_argument("--gpu-histories", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    csv_path = args.run_dir / "component_primary_c12_roi.csv"
    if not csv_path.is_file():
        csv_path = args.run_dir / "out/config/dose_primary_c12_roi.csv"
    mhd = parse_mhd(args.run_dir / "out/config/dose.mhd")
    dims, spacing, offset = mhd["dimensions"], mhd["spacing_mm"], mhd["offset_mm"]
    assert isinstance(dims, list) and isinstance(spacing, list)
    assert isinstance(offset, list)
    nz = int(dims[2])
    depth = offset[2] + np.arange(nz) * spacing[2]
    values = np.zeros((len(KINDS), len(BANDS), len(REGIONS), nz))
    kind_i = {name: i for i, name in enumerate(KINDS)}
    band_i = {name: i for i, name in enumerate(BANDS)}
    region_i = {name: i for i, name in enumerate(REGIONS)}
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            iz = int(np.argmin(np.abs(depth - float(row["depth_mm"]))))
            values[kind_i[row["kind"]], band_i[row["energy_band_MeV_per_u"]],
                   region_i[row["region"]], iz] = float(row["value"])
    slab_bins = max(1, int(round(1.0 / spacing[2])))
    smooth = uniform_filter1d(values, slab_bins, axis=-1, mode="nearest")
    factor = MEV_TO_JOULE / (spacing[0] * spacing[1] * spacing[2] * 1.0e-6)
    records = []
    for requested in (0.5, 40.0, 80.0, 120.0, args.bragg_depth_mm):
        iz = int(np.argmin(np.abs(depth - requested)))
        for ir, region in enumerate(REGIONS):
            kinds = {}
            for ik, kind in enumerate(KINDS):
                bands = smooth[ik, :, ir, iz]
                total = float(bands.sum())
                kinds[kind] = {
                    "value": total,
                    "dose_sum_Gy": None if kind in {"fluence_mm"} else total * factor,
                    "band_fraction": {
                        band: (float(bands[ib] / total) if total else 0.0)
                        for ib, band in enumerate(BANDS)
                    },
                }
            local = kinds["local_total_deposit_MeV"]["value"]
            continuous = kinds["continuous_sampled_MeV"]["value"]
            delta = kinds["delta_sampled_MeV"]["value"]
            records.append({
                "requested_depth_mm": requested,
                "grid_depth_mm": float(depth[iz]),
                "window": "1mm_smooth_plane",
                "region": region,
                "kinds": kinds,
                "sampled_continuous_plus_delta_over_local_total": (
                    (continuous + delta) / local if local else None),
                "delta_fraction_of_sampled": (
                    delta / (continuous + delta) if (continuous + delta) else None),
                "after_scale_over_sampled": (
                    (kinds["continuous_after_scale_MeV"]["value"] +
                     kinds["delta_after_scale_MeV"]["value"]) /
                    (continuous + delta) if (continuous + delta) else None),
                "per_incident_history": {
                    "fluence_mm": kinds["fluence_mm"]["value"] / args.gpu_histories,
                    "local_total_deposit_MeV": local / args.gpu_histories,
                },
                "uncertainty": "unknown",
            })
    depth_mask = np.abs(depth - args.bragg_depth_mm) <= 2.0
    for ir, region in enumerate(REGIONS):
        kinds = {}
        for ik, kind in enumerate(KINDS):
            bands = values[ik, :, ir, :][:, depth_mask].sum(axis=-1)
            total = float(bands.sum())
            kinds[kind] = {
                "value": total,
                "band_fraction": {
                    band: (float(bands[ib] / total) if total else 0.0)
                    for ib, band in enumerate(BANDS)
                },
            }
        records.append({
            "requested_depth_mm": args.bragg_depth_mm,
            "grid_depth_mm": args.bragg_depth_mm,
            "window": "bragg_pm2mm_unsmoothed",
            "region": region,
            "kinds": kinds,
            "uncertainty": "unknown",
        })
    payload = {
        "definition": (
            "local_total_deposit is the spatially local scored deposit after "
            "any packet relocation. continuous_sampled and delta_sampled are "
            "the same unified_em_loss draw, before the 0.9958 total-loss "
            "scale; after_scale multiplies both by deposited/unscaled. "
            "RNG and transport final state are unchanged."),
        "gpu_histories": args.gpu_histories,
        "uncertainty": "unknown",
        "records": records,
    }
    output = args.output or args.run_dir / "primary_c12_roi_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
