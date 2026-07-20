#!/usr/bin/env python3
"""Reorient a CCTG patient grid for the TPS 90-degree carbon field.

The spots RotX/RotY values establish the fixed TPS 0-degree source direction.
The TOPAS run then rotates the patient by RotZ=90 deg, producing TPS 90-degree
incidence. After undoing that patient rotation, the beam travels along patient +X.
The GPU transport/scorer axis is +Z, so this utility permutes the grid as:

    GPU (x, y, z) = patient (y, z, x)

No interpolation is performed; density/material values and the CCTG material
table tail are preserved bit-for-bit.
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

    # new[patient_x, patient_z, patient_y] = old[patient_z, patient_y, patient_x].
    # Use only the standard library so plan preparation has no NumPy dependency.
    density_new = array.array("f", [0.0]) * count
    material_new = bytearray(count)
    old_plane = nx * ny
    for patient_x in range(nx):
        for patient_z in range(nz):
            old_index = patient_z * old_plane + patient_x
            new_index = (patient_x * nz + patient_z) * ny
            for patient_y in range(ny):
                density_new[new_index + patient_y] = density[old_index]
                material_new[new_index + patient_y] = material[old_index]
                old_index += nx
    nx_new, ny_new, nz_new = ny, nz, nx
    spacing_new = (spacing_y, spacing_z, spacing_x)
    # prepare_ct_grid stores the first DICOM slice at z=0. TOPAS centers the
    # DICOM volume on the patient component, so restore the centered Z coordinate.
    patient_z_min = origin_z - 0.5 * (nz - 1) * spacing_z
    origin_new = (origin_y, patient_z_min, 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header_new = HEADER.pack(
        magic,
        version,
        nx_new,
        ny_new,
        nz_new,
        *origin_new,
        *spacing_new,
    )
    with args.output.open("wb") as handle:
        handle.write(header_new)
        density_bytes = array.array("f", density_new)
        if sys.byteorder != "little":
            density_bytes.byteswap()
        handle.write(density_bytes.tobytes())
        handle.write(material_new)
        handle.write(payload[tail_begin:])

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "grid_version": version,
        "axis_mapping": "gpu(x,y,z)=patient(y,z,x)",
        "tps_angle_deg": 90.0,
        "beam_axis": "TPS 90: patient +X -> GPU +Z",
        "interpolation": False,
        "shape_xyz": [nx_new, ny_new, nz_new],
        "origin_xyz_mm": list(origin_new),
        "spacing_xyz_mm": list(spacing_new),
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
