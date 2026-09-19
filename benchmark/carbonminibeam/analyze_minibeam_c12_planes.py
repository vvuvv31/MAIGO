#!/usr/bin/env python3
"""C12 plane crossings at the canonical minibeam ROIs.

Counts, direction, spectrum and survival are per incident history, not per
surviving particle. Uncertainty is unknown unless independent shards exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_gpu_topas_dose import fixed_minibeam_region_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--incident-histories", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.planes.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["atomic_number"]) != 6 or int(row["mass_number"]) != 12:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no C12 plane records in {args.planes}")
    x = np.array([float(row["x_mm"]) for row in rows])
    energy = np.array([float(row["kinetic_energy_MeV"]) for row in rows])
    dz = np.array([float(row["direction_z"]) for row in rows])
    depth = np.array([float(row["depth_mm"]) for row in rows])
    history = np.array([int(row["source_history"]) for row in rows])
    planes = sorted(set(float(row["depth_mm"]) for row in rows))
    dummy_x = np.linspace(-50.0, 49.9, 1000)
    masks = fixed_minibeam_region_masks(dummy_x, 3.6)
    records = []
    for plane_depth in planes:
        select = np.isclose(depth, plane_depth)
        for region, mask in masks.items():
            # Map each x to the nearest dummy bin used by the canonical mask.
            bins = np.clip(np.rint((x[select] + 49.95) / 0.1).astype(int), 0, 999)
            in_region = mask[bins]
            n = int(in_region.sum())
            energies = energy[select][in_region]
            dirs = dz[select][in_region]
            hist = history[select][in_region]
            records.append({
                "depth_mm": plane_depth,
                "region": region,
                "crossings": n,
                "unique_histories": int(np.unique(hist).size) if n else 0,
                "survival_per_incident": n / args.incident_histories,
                "mean_energy_MeV": float(energies.mean()) if n else None,
                "energy_std_MeV": float(energies.std()) if n else None,
                "mean_direction_z": float(dirs.mean()) if n else None,
                "rms_theta_mrad": (
                    float(np.sqrt(np.mean(np.arccos(np.clip(dirs, -1, 1)) ** 2)) * 1.0e3)
                    if n else None),
                "normalization": "per incident history",
                "uncertainty": "unknown",
            })
    args.output.write_text(json.dumps({
        "incident_histories": args.incident_histories,
        "records": records,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
