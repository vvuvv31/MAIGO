#!/usr/bin/env python3
"""Block-average a CCTG grid while preserving its material-table tail.

Density is volume averaged, which preserves the mass represented by every
complete output block.  The output material section is the density-weighted
mode of the input sections, avoiding air winning a mixed air/tissue voxel by
count alone.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = 0x47544343
HEADER = struct.Struct("<IIIIIffffff")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factors", nargs=3, type=int, required=True,
                        metavar=("FX", "FY", "FZ"))
    parser.add_argument("--trim-low", nargs=3, type=int, default=(0, 0, 0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--trim-high", nargs=3, type=int, default=(0, 0, 0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    payload = args.input.read_bytes()
    if len(payload) < HEADER.size:
        raise SystemExit(f"Truncated CCTG header: {args.input}")
    values = HEADER.unpack_from(payload)
    magic, version, nx, ny, nz = values[:5]
    ox, oy, oz, sx, sy, sz = values[5:]
    if magic != MAGIC or version not in (1, 2, 3):
        raise SystemExit(f"Unsupported CCTG magic/version: {magic:#x}/v{version}")

    count = nx * ny * nz
    density_start = HEADER.size
    material_start = density_start + 4 * count
    tail_start = material_start + count
    if len(payload) < tail_start:
        raise SystemExit(f"Truncated CCTG voxel arrays: {args.input}")

    density = np.frombuffer(
        payload, dtype="<f4", count=count, offset=density_start
    ).reshape(nz, ny, nx)
    material = np.frombuffer(
        payload, dtype=np.uint8, count=count, offset=material_start
    ).reshape(nz, ny, nx)

    fx, fy, fz = args.factors
    lx, ly, lz = args.trim_low
    hx, hy, hz = args.trim_high
    if min(fx, fy, fz) < 1 or min(lx, ly, lz, hx, hy, hz) < 0:
        raise SystemExit("Factors must be positive and trims non-negative")
    kept_x, kept_y, kept_z = nx - lx - hx, ny - ly - hy, nz - lz - hz
    if kept_x <= 0 or kept_y <= 0 or kept_z <= 0:
        raise SystemExit("Trims remove the complete grid")
    if kept_x % fx or kept_y % fy or kept_z % fz:
        raise SystemExit(
            f"Trimmed shape {(kept_x, kept_y, kept_z)} is not divisible by "
            f"factors {(fx, fy, fz)}"
        )

    density = density[lz:nz - hz if hz else None,
                      ly:ny - hy if hy else None,
                      lx:nx - hx if hx else None]
    material = material[lz:nz - hz if hz else None,
                        ly:ny - hy if hy else None,
                        lx:nx - hx if hx else None]
    nz2, ny2, nx2 = kept_z // fz, kept_y // fy, kept_x // fx
    block_shape = (nz2, fz, ny2, fy, nx2, fx)
    density_blocks = density.reshape(block_shape).transpose(0, 2, 4, 1, 3, 5)
    material_blocks = material.reshape(block_shape).transpose(0, 2, 4, 1, 3, 5)
    density_out = density_blocks.mean(axis=(3, 4, 5), dtype=np.float64).astype("<f4")

    # At most 256 section IDs exist. Accumulate density (mass per equal-volume
    # fine voxel) per section and retain the dominant section in each block.
    mass_by_section = np.zeros((nz2, ny2, nx2, 256), dtype=np.float32)
    for section in np.unique(material_blocks):
        mass_by_section[..., int(section)] = np.sum(
            density_blocks * (material_blocks == section), axis=(3, 4, 5)
        )
    material_out = np.argmax(mass_by_section, axis=-1).astype(np.uint8)

    new_origin = (ox + lx * sx, oy + ly * sy, oz + lz * sz)
    new_spacing = (sx * fx, sy * fy, sz * fz)
    header = HEADER.pack(
        magic, version, nx2, ny2, nz2,
        *new_origin, *new_spacing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(density_out).tobytes())
        handle.write(np.ascontiguousarray(material_out).tobytes())
        handle.write(payload[tail_start:])

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_shape_xyz": [nx, ny, nz],
        "output_shape_xyz": [nx2, ny2, nz2],
        "factors_xyz": [fx, fy, fz],
        "trim_low_xyz": [lx, ly, lz],
        "trim_high_xyz": [hx, hy, hz],
        "origin_xyz_mm": list(new_origin),
        "spacing_xyz_mm": list(new_spacing),
        "density_method": "volume mean",
        "material_method": "density-weighted mode",
        "input_bytes": len(payload),
        "output_bytes": args.output.stat().st_size,
        "voxel_reduction": count / (nx2 * ny2 * nz2),
    }
    metadata = args.metadata or args.output.with_suffix(".metadata.json")
    metadata.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
