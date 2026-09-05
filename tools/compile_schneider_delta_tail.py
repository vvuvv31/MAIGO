#!/usr/bin/env python3
"""Compile a Schneider section-0 C12 delta-tail LUT from TOPAS 3D scorers."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(path: Path, radius_cut_mm: float, q_count: int):
    data = np.loadtxt(path, delimiter=",", comments="#")
    xyz = data[:, :3].astype(np.int32)
    dose = np.zeros((440, 100, 100), dtype=np.float64)
    dose[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = data[:, 3]
    # Independent 3D scorer only; sum a fixed interior 100--200 mm interval.
    lateral = dose[200:400].sum(axis=0)
    yy, xx = np.indices(lateral.shape)
    cy = float((lateral * yy).sum() / lateral.sum())
    cx = float((lateral * xx).sum() / lateral.sum())
    radius = 2.0 * np.hypot(xx - cx, yy - cy)
    select = (radius > radius_cut_mm) & (lateral > 0.0)
    weights = lateral[select]
    radii = radius[select]
    moved = float(weights.sum() / lateral.sum())
    order = np.argsort(radii)
    radii, weights = radii[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()
    quantiles = np.linspace(0.0, 1.0, q_count)
    indices = np.searchsorted(cdf, quantiles, side="left")
    indices = np.clip(indices, 0, len(radii) - 1)
    return moved, quantiles, radii[indices], (cx, cy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="energy_MeV_per_u:path_to_TOPAS_3D_csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--radius-cut-mm", type=float, default=2.01)
    p.add_argument("--quantiles", type=int, default=101)
    args = p.parse_args()
    rows = []
    inputs = []
    for spec in args.input:
        energy_text, path_text = spec.split(":", 1)
        energy, path = float(energy_text), Path(path_text)
        moved, quantiles, radii, center = extract(
            path, args.radius_cut_mm, args.quantiles)
        rows.append((energy, moved, quantiles, radii))
        inputs.append({"energy_MeV_per_u": energy, "path": str(path),
                       "sha256": sha256(path), "center_index": list(center),
                       "measured_moved_fraction": moved})
    rows.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("energy_MeV_per_u,moved_fraction,quantile,radius_mm\n")
        for energy, moved, quantiles, radii in rows:
            for q, radius in zip(quantiles, radii):
                out.write(f"{energy:.8g},{moved:.9g},{q:.8g},{radius:.9g}\n")
    metadata = {
        "schema_version": 1,
        "scope": "primary C12, Schneider section 0 only",
        "source": "TOPAS 4.2.p3 / Geant4 11.3.p02 g4em-standard_opt4 DoseToMedium 3D",
        "production_cut_mm": 0.05,
        "beam": "zero-width C12, 1% energy spread",
        "histories": {"150": 100000, "200": 200000, "225": 100000},
        "lateral_voxel_mm": [2.0, 2.0],
        "depth_voxel_mm": 0.5,
        "summed_depth_interval_mm": [100.0, 200.0],
        "tail_radius_cut_mm": args.radius_cut_mm,
        "quantile_count": args.quantiles,
        "interpolation": "linear in energy and uniform quantile",
        "inputs": inputs,
        "data_file": str(args.output),
        "data_filename": args.output.name,
        "data_sha256": sha256(args.output),
        "data_size_bytes": args.output.stat().st_size,
        "file_size_bytes": args.output.stat().st_size,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
