#!/usr/bin/env python3
"""Compare charged-ion water-entry phase space from GPU and TOPAS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ion_za(pdg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = np.abs(pdg.astype(np.int64))
    ion = code >= 1_000_000_000
    z = np.where(ion, (code // 10_000) % 1000, np.where(code == 2212, 1, 0))
    a = np.where(ion, (code // 10) % 1000, np.where(code == 2212, 1, 0))
    return z, a


def summarize(z: np.ndarray, a: np.ndarray, energy: np.ndarray,
              x_mm: np.ndarray, y_mm: np.ndarray, dx: np.ndarray,
              dy: np.ndarray, longitudinal: np.ndarray
              ) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    theta = np.arctan2(np.hypot(dx, dy), longitudinal)
    radius = np.hypot(x_mm, y_mm)
    for zz, aa in sorted(set(zip(z.tolist(), a.tolist()))):
        mask = (z == zz) & (a == aa)
        values = energy[mask]
        angles = theta[mask]
        radii = radius[mask]
        key = f"Z{zz}_A{aa}"
        result[key] = {
            "count": int(mask.sum()),
            "energy_mean_MeV": float(values.mean()),
            "energy_std_MeV": float(values.std()),
            "energy_q50_MeV": float(np.quantile(values, 0.5)),
            "energy_q95_MeV": float(np.quantile(values, 0.95)),
            "theta_q50_mrad": float(1000.0 * np.quantile(angles, 0.5)),
            "theta_q95_mrad": float(1000.0 * np.quantile(angles, 0.95)),
            "radius_q50_mm": float(np.quantile(radii, 0.5)),
            "radius_q95_mm": float(np.quantile(radii, 0.95)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--topas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer-half-width-mm", type=float, default=70.0)
    parser.add_argument("--gpu-back-project-mm", type=float, default=0.02,
                        help="distance from GPU water entrance back to TOPAS scorer")
    args = parser.parse_args()

    with args.gpu.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    gpu_z = np.asarray([int(row["Z"]) for row in rows])
    gpu_a = np.asarray([int(row["A"]) for row in rows])
    gpu_energy = np.asarray([float(row["kinetic_energy_MeV"]) for row in rows])
    gpu_x = np.asarray([float(row["x_mm"]) for row in rows])
    gpu_y = np.asarray([float(row["y_mm"]) for row in rows])
    gpu_dx = np.asarray([float(row["direction_x"]) for row in rows])
    gpu_dy = np.asarray([float(row["direction_y"]) for row in rows])
    gpu_dz = np.asarray([float(row["direction_z"]) for row in rows])
    gpu_x -= args.gpu_back_project_mm * gpu_dx / gpu_dz
    gpu_y -= args.gpu_back_project_mm * gpu_dy / gpu_dz
    gpu_keep = ((np.abs(gpu_x) <= args.scorer_half_width_mm) &
                (np.abs(gpu_y) <= args.scorer_half_width_mm) & (gpu_dz > 0.0))
    gpu_summary = summarize(
        gpu_z[gpu_keep], gpu_a[gpu_keep], gpu_energy[gpu_keep],
        gpu_x[gpu_keep], gpu_y[gpu_keep], gpu_dx[gpu_keep], gpu_dy[gpu_keep],
        gpu_dz[gpu_keep],
    )

    topas = np.loadtxt(args.topas, ndmin=2)
    topas_z, topas_a = ion_za(topas[:, 7])
    charged_ion = (topas_z > 0) & (topas_a > 0)
    # Parent-0 C-12 is the surviving incident primary, not a Copper fragment.
    primary = (topas_z == 6) & (topas_a == 12) & (topas[:, 13] == 0)
    keep = charged_ion & ~primary
    topas_uz = np.sqrt(np.maximum(
        0.0, 1.0 - topas[:, 3] * topas[:, 3] - topas[:, 4] * topas[:, 4]
    ))
    topas_uz = np.where(topas[:, 8] != 0, -topas_uz, topas_uz)
    topas_summary = summarize(
        topas_z[keep], topas_a[keep], topas[keep, 5],
        # TOPAS beam axis is +world-y; transverse coordinates are world x/z.
        10.0 * topas[keep, 0], 10.0 * topas[keep, 2],
        topas[keep, 3], topas_uz[keep], topas[keep, 4],
    )

    species: dict[str, dict[str, object]] = {}
    for key in sorted(set(gpu_summary) | set(topas_summary)):
        gpu = gpu_summary.get(key)
        ref = topas_summary.get(key)
        entry: dict[str, object] = {"gpu": gpu, "topas": ref}
        if gpu is not None and ref is not None and ref["count"]:
            entry["gpu_over_topas_count"] = (
                float(gpu["count"]) / float(ref["count"])
            )
            for metric in (
                "energy_mean_MeV", "energy_std_MeV", "energy_q50_MeV",
                "energy_q95_MeV",
                "theta_q50_mrad", "theta_q95_mrad", "radius_q50_mm",
                "radius_q95_mm",
            ):
                denominator = float(ref[metric])
                if denominator != 0.0:
                    entry[f"gpu_over_topas_{metric}"] = (
                        float(gpu[metric]) / denominator
                    )
        species[key] = entry

    result = {
        "normalization": (
            "same incident histories and +/- scorer extent; GPU back-projected "
            "to TOPAS plane; TOPAS parent-0 C12 excluded"
        ),
        "scorer_half_width_mm": args.scorer_half_width_mm,
        "gpu_back_project_mm": args.gpu_back_project_mm,
        "gpu_fragment_count_before_scorer_cut": int(len(rows)),
        "gpu_fragment_count": int(gpu_keep.sum()),
        "topas_fragment_count": int(keep.sum()),
        "species": species,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
