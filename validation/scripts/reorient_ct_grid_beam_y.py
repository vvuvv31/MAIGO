#!/usr/bin/env python3
"""Reorient a CCTG patient grid for a TOPAS beam along world/patient +Y.

TOPAS setup (20022516 lung): Patient RotZ≈0, BeamPosition at TransY=-SAD with
Rx≈90 so the central ray travels +world-Y. GPU transport is fixed along +Z, so:

    GPU (x, y, z) = patient (x, z, y)

with GPU z=0 at the low patient-Y edge (beam entrance from −Y).
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--beam-patient-y-direction",
        choices=("positive", "negative"),
        default="positive",
        help="Patient-Y direction of beam travel (+Y is the TOPAS lung setup)",
    )
    args = parser.parse_args()

    payload = args.input.read_bytes()
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
        raise SystemExit(f"Unsupported CCTG in {args.input}")

    count = nx * ny * nz
    density_begin = HEADER.size
    material_begin = density_begin + 4 * count
    tail_begin = material_begin + count
    density = array.array("f")
    density.frombytes(payload[density_begin:material_begin])
    if sys.byteorder != "little":
        density.byteswap()
    material = payload[material_begin:tail_begin]
    tail = payload[tail_begin:]

    # Old layout: index = iz * (nx*ny) + iy * nx + ix  (x fastest, then y, then z)
    # New: GPU (x,y,z) = (patient_x, patient_z, patient_y)
    # index_new = iz_g * (nx*nz) + iy_g * nx + ix   with iy_g=patient_z, iz_g=patient_y
    density_new = array.array("f", [0.0]) * count
    material_new = bytearray(count)
    old_plane = nx * ny
    for patient_y in range(ny):
        gpu_z = (
            patient_y
            if args.beam_patient_y_direction == "positive"
            else ny - 1 - patient_y
        )
        for patient_z in range(nz):
            for patient_x in range(nx):
                old_index = patient_z * old_plane + patient_y * nx + patient_x
                new_index = (gpu_z * nz + patient_z) * nx + patient_x
                density_new[new_index] = density[old_index]
                material_new[new_index] = material[old_index]

    nx_new, ny_new, nz_new = nx, nz, ny
    spacing_new = (spacing_x, spacing_z, spacing_y)
    # Keep patient X origin; center patient Z about 0 for lateral Y; depth at 0.
    origin_x_new = origin_x
    volume_center_z = origin_z + 0.5 * nz * spacing_z
    origin_y_new = origin_z - volume_center_z
    origin_z_new = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(
            HEADER.pack(
                magic,
                version,
                nx_new,
                ny_new,
                nz_new,
                float(origin_x_new),
                float(origin_y_new),
                float(origin_z_new),
                float(spacing_new[0]),
                float(spacing_new[1]),
                float(spacing_new[2]),
            )
        )
        if sys.byteorder != "little":
            density_new.byteswap()
        handle.write(density_new.tobytes())
        handle.write(bytes(material_new))
        handle.write(tail)

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "beam_patient_y_direction": args.beam_patient_y_direction,
        "old_shape_xyz": [nx, ny, nz],
        "new_shape_xyz": [nx_new, ny_new, nz_new],
        "old_origin_xyz_mm": [origin_x, origin_y, origin_z],
        "new_origin_xyz_mm": [origin_x_new, origin_y_new, origin_z_new],
        "old_spacing_xyz_mm": [spacing_x, spacing_y, spacing_z],
        "new_spacing_xyz_mm": list(spacing_new),
        "mapping": "GPU(x,y,z)=patient(x,z,±y)",
        "phantom_length_mm": float(nz_new * spacing_new[2]),
        "spots_ct_axis_min_mm_note": (
            "For tps_gantry_y geometry use patient-Y low edge as entrance; "
            f"depth extent {nz_new * spacing_new[2]:.4g} mm"
        ),
    }
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
