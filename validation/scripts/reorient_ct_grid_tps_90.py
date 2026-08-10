#!/usr/bin/env python3
"""Reorient a CCTG patient grid for the TPS 90-degree carbon field.

Spot RotX/RotY establish the fixed TPS 0° source direction. GPU transport is
fixed along +Z, so the grid is permuted according to the actual beam direction:

    patient +X beam: GPU (x, y, z) = patient (y, z, +x)
    patient -X beam: GPU (x, y, z) = patient (y, z, -x)

The patient-X voxel order must be reversed for a -X beam. Reversing only the
spot coordinates would make particles traverse a depth-mirrored anatomy.

No interpolation: density, material id, and CCTG table tail are bit-preserved
aside from the axis permutation.

CCTG origin is the **low edge** of the first voxel (matches ct_sample).
"""

from __future__ import annotations

import argparse
import array
import json
import struct
import sys
from pathlib import Path

MAGIC = 0x47544343
HEADER = struct.Struct("<IIIIIffffff")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ct/grid/patient_ct.bin"))
    parser.add_argument(
        "--output", type=Path, default=Path("ct/grid/patient_ct_tps_90.bin")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("ct/grid/patient_ct_tps_90.metadata.json"),
    )
    parser.add_argument(
        "--beam-patient-x-direction",
        choices=("positive", "negative"),
        default="positive",
        help=(
            "Patient-X direction of beam travel. 'negative' reverses the patient-X "
            "voxel order so beam depth still increases along GPU +Z."
        ),
    )
    args = parser.parse_args()

    payload = args.input.read_bytes()
    if len(payload) < HEADER.size:
        raise SystemExit(f"Truncated CCTG header: {args.input}")
    (
        magic,
        version,
        nx,
        ny,
        nz,
        origin_x,
        origin_y,
        origin_z,
        spacing_x,
        spacing_y,
        spacing_z,
    ) = HEADER.unpack_from(payload)
    if magic != MAGIC or version not in {1, 2, 3}:
        raise SystemExit(f"Unsupported CCTG magic/version in {args.input}")

    count = nx * ny * nz
    density_begin = HEADER.size
    material_begin = density_begin + 4 * count
    tail_begin = material_begin + count
    if len(payload) < tail_begin:
        raise SystemExit(f"Truncated CCTG voxel arrays: {args.input}")

    density = array.array("f")
    density.frombytes(payload[density_begin:material_begin])
    if sys.byteorder != "little":
        density.byteswap()
    material = payload[material_begin:tail_begin]

    # GPU (x,y,z) indices = (patient_y, patient_z, patient_x) for a +X beam,
    # or (patient_y, patient_z, nx-1-patient_x) for a -X beam.
    density_new = array.array("f", [0.0]) * count
    material_new = bytearray(count)
    old_plane = nx * ny
    for patient_x in range(nx):
        gpu_z = (
            patient_x
            if args.beam_patient_x_direction == "positive"
            else nx - 1 - patient_x
        )
        for patient_z in range(nz):
            old_index = patient_z * old_plane + patient_x
            new_index = (gpu_z * nz + patient_z) * ny
            for patient_y in range(ny):
                density_new[new_index + patient_y] = density[old_index]
                material_new[new_index + patient_y] = material[old_index]
                old_index += nx

    nx_new, ny_new, nz_new = ny, nz, nx
    spacing_new = (spacing_y, spacing_z, spacing_x)
    # GPU x origin = patient y low edge
    origin_x_new = origin_y
    # Center patient Z about 0 (edge convention)
    volume_center_z = origin_z + 0.5 * nz * spacing_z
    origin_y_new = origin_z - volume_center_z
    # Patient X low edge -> GPU z = 0
    origin_z_new = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header_new = HEADER.pack(
        magic,
        version,
        nx_new,
        ny_new,
        nz_new,
        float(origin_x_new),
        float(origin_y_new),
        float(origin_z_new),
        *spacing_new,
    )
    with args.output.open("wb") as handle:
        handle.write(header_new)
        density_bytes = array.array("f", density_new)
        if sys.byteorder != "little":
            density_bytes.byteswap()
        handle.write(density_bytes.tobytes())
        handle.write(bytes(material_new))
        handle.write(payload[tail_begin:])

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "grid_version": version,
        "axis_mapping": (
            "gpu(x,y,z)=patient(y,z,x)"
            if args.beam_patient_x_direction == "positive"
            else "gpu(x,y,z)=patient(y,z,-x)"
        ),
        "beam_patient_x_direction": args.beam_patient_x_direction,
        "tps_angle_deg": 90.0,
        "beam_axis": (
            "patient +X -> GPU +Z"
            if args.beam_patient_x_direction == "positive"
            else "patient -X -> GPU +Z"
        ),
        "interpolation": False,
        "shape_xyz": [nx_new, ny_new, nz_new],
        "origin_xyz_mm": [origin_x_new, origin_y_new, origin_z_new],
        "origin_convention": "low_edge_of_first_voxel",
        "spacing_xyz_mm": list(spacing_new),
        "patient_shape_xyz": [nx, ny, nz],
        "patient_origin_xyz_mm": [origin_x, origin_y, origin_z],
        "patient_spacing_xyz_mm": [spacing_x, spacing_y, spacing_z],
        "phantom_length_mm": nx * spacing_x,
        "voxel_count": count,
        "density_min": float(min(density_new)),
        "density_max": float(max(density_new)),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
