#!/usr/bin/env python3
"""Resample a CCTG v3 grid to a nominal isotropic voxel spacing.

The exact physical bounds are preserved, so each axis spacing is adjusted
slightly when the original extent is not an integer multiple of the requested
nominal spacing. Density uses separable volume-overlap averaging.  Schneider
sections are kept when that averaged density is valid for the centre voxel's
section; otherwise the closest source section density interval is selected and
the density is clamped to that interval.  This prevents mixed air/tissue or
metal/tissue boundary voxels from creating unsupported material-density pairs.
The v3 section stopping-power tables are copied verbatim.
"""

import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np


HEADER = struct.Struct("<5I6f")
MAGIC = 0x47544343


def read_cctg(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("truncated CCTG header")
        magic, version, nx, ny, nz, ox, oy, oz, sx, sy, sz = HEADER.unpack(raw)
        if magic != MAGIC or version != 3:
            raise ValueError(f"expected CCTG v3, got magic={magic:#x} version={version}")
        count = nx * ny * nz
        density = np.fromfile(stream, dtype="<f4", count=count)
        material = np.fromfile(stream, dtype="u1", count=count)
        raw_n = stream.read(4)
        if density.size != count or material.size != count or len(raw_n) != 4:
            raise ValueError("truncated CCTG voxel arrays")
        section_count = struct.unpack("<I", raw_n)[0]
        za_rel = np.fromfile(stream, dtype="<f4", count=section_count)
        i_ev = np.fromfile(stream, dtype="<f4", count=section_count)
        if za_rel.size != section_count or i_ev.size != section_count or stream.read(1):
            raise ValueError("invalid CCTG v3 section tables or trailing bytes")
    shape = (nz, ny, nx)
    return {
        "shape": shape,
        "origin": np.array([oz, oy, ox], dtype=np.float64),
        "spacing": np.array([sz, sy, sx], dtype=np.float64),
        "density": density.reshape(shape),
        "material": material.reshape(shape),
        "za_rel": za_rel,
        "i_ev": i_ev,
    }


def overlap_weights(in_n, in_origin, in_spacing, out_n, out_origin, out_spacing):
    in_lo = in_origin + np.arange(in_n, dtype=np.float64) * in_spacing
    in_hi = in_lo + in_spacing
    out_lo = out_origin + np.arange(out_n, dtype=np.float64) * out_spacing
    out_hi = out_lo + out_spacing
    weights = np.maximum(
        0.0,
        np.minimum(out_hi[:, None], in_hi[None, :])
        - np.maximum(out_lo[:, None], in_lo[None, :]),
    )
    sums = weights.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError("output grid does not overlap input grid")
    return (weights / sums[:, None]).astype(np.float32)


def apply_axis(values, weights, axis):
    result = np.tensordot(weights, values, axes=(1, axis))
    return np.moveaxis(result, 0, axis)


def compatible_material_and_density(source_density, source_material,
                                    density, centre_material):
    section_count = int(source_material.max()) + 1
    density_min = np.empty(section_count, dtype=np.float32)
    density_max = np.empty(section_count, dtype=np.float32)
    for section in range(section_count):
        values = source_density[source_material == section]
        if values.size == 0:
            raise ValueError(f"source grid has no voxels for section {section}")
        density_min[section] = values.min()
        density_max[section] = values.max()

    flat_density = np.ascontiguousarray(density, dtype=np.float32).reshape(-1)
    flat_material = centre_material.reshape(-1).copy()
    current_min = density_min[flat_material]
    current_max = density_max[flat_material]
    incompatible = (flat_density < current_min) | (flat_density > current_max)
    if np.any(incompatible):
        values = flat_density[incompatible, None]
        distance = np.maximum(
            density_min[None, :] - values,
            np.maximum(values - density_max[None, :], 0.0),
        )
        flat_material[incompatible] = np.argmin(distance, axis=1).astype(np.uint8)

    flat_density[:] = np.clip(
        flat_density,
        density_min[flat_material],
        density_max[flat_material],
    )
    return (flat_material.reshape(density.shape),
            flat_density.reshape(density.shape))


def resample(grid, target_mm):
    in_shape = np.array(grid["shape"], dtype=np.int64)
    in_origin = grid["origin"]
    in_spacing = grid["spacing"]
    extent = in_shape * in_spacing
    out_shape = np.maximum(1, np.floor(extent / target_mm + 0.5).astype(np.int64))
    out_spacing = extent / out_shape
    out_origin = in_origin.copy()

    density = grid["density"].astype(np.float32, copy=False)
    for axis in range(3):
        weights = overlap_weights(
            density.shape[axis], in_origin[axis], in_spacing[axis],
            int(out_shape[axis]), out_origin[axis], out_spacing[axis],
        )
        density = apply_axis(density, weights, axis)

    centres = [out_origin[a] + (np.arange(out_shape[a]) + 0.5) * out_spacing[a]
               for a in range(3)]
    source_indices = [np.clip(np.floor((centres[a] - in_origin[a]) / in_spacing[a]).astype(int),
                              0, in_shape[a] - 1)
                      for a in range(3)]
    centre_material = grid["material"][np.ix_(*source_indices)]
    material, density = compatible_material_and_density(
        grid["density"], grid["material"], density, centre_material)
    return {
        **grid,
        "shape": tuple(map(int, out_shape)),
        "origin": out_origin,
        "spacing": out_spacing,
        "density": np.ascontiguousarray(density, dtype="<f4"),
        "material": np.ascontiguousarray(material, dtype="u1"),
    }


def write_cctg(path: Path, grid):
    nz, ny, nx = grid["shape"]
    oz, oy, ox = grid["origin"]
    sz, sy, sx = grid["spacing"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, 3, nx, ny, nz, ox, oy, oz, sx, sy, sz))
        grid["density"].tofile(stream)
        grid["material"].tofile(stream)
        stream.write(struct.pack("<I", len(grid["za_rel"])))
        np.asarray(grid["za_rel"], dtype="<f4").tofile(stream)
        np.asarray(grid["i_ev"], dtype="<f4").tofile(stream)


def summary(grid):
    voxel_volume = float(np.prod(grid["spacing"]))
    return {
        "shape_zyx": list(grid["shape"]),
        "origin_zyx_mm": grid["origin"].tolist(),
        "spacing_zyx_mm": grid["spacing"].tolist(),
        "extent_zyx_mm": (np.array(grid["shape"]) * grid["spacing"]).tolist(),
        "density_min_max": [float(grid["density"].min()), float(grid["density"].max())],
        "density_volume_sum_g_per_cm3_mm3": float(grid["density"].sum(dtype=np.float64) * voxel_volume),
        "material_ids": sorted(map(int, np.unique(grid["material"]))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing-mm", type=float, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.spacing_mm) or args.spacing_mm <= 0:
        parser.error("--spacing-mm must be positive and finite")
    source = read_cctg(args.input)
    result = resample(source, args.spacing_mm)
    write_cctg(args.output, result)
    report = {"input": summary(source), "output": summary(result)}
    report["density_volume_sum_relative_change"] = (
        report["output"]["density_volume_sum_g_per_cm3_mm3"]
        / report["input"]["density_volume_sum_g_per_cm3_mm3"] - 1.0
    )
    text = json.dumps(report, indent=2)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
