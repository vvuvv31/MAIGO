#!/usr/bin/env python3
"""Map GPU TPS-90 dose MHD into the original patient / DICOM axis frame.

GPU transport stores dose with:
    GPU (x, y, z) = patient (y, z, x)

physical_dose.mhd (and DICOM) use patient (x, y, z). After transform_tps_90 maps
GPU x = -patient_y, matching physical_dose usually only needs a patient-X flip
for distal/entrance orientation of this plan:

    patient[ix, iy, iz] = GPU[iy, iz, nx-1-ix]   # flip X only (default)

Legacy mirrored runs (pre patient-Y sign fix) may still need flip Y as well.
Default output spacing matches the fine patient grid (0.5, 0.5, 2.0) mm.
Pass --like ct/physical_dose.mhd to resample to that grid (2 mm isotropic).
"""

from __future__ import annotations

import argparse
import array
import math
import sys
from pathlib import Path


def read_mhd(path: Path) -> tuple[dict[str, str], array.array]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    dim = [int(v) for v in meta["DimSize"].split()]
    raw_path = path.parent / meta["ElementDataFile"]
    data = array.array("f")
    data.frombytes(raw_path.read_bytes())
    expected = dim[0] * dim[1] * dim[2]
    if len(data) != expected:
        raise SystemExit(f"{raw_path}: expected {expected} floats, got {len(data)}")
    if sys.byteorder != "little":
        data.byteswap()
    return meta, data


def write_mhd(
    path: Path,
    data: array.array,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    offset: tuple[float, float, float],
    units: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    out = array.array("f", data)
    if sys.byteorder != "little":
        out.byteswap()
    raw_path.write_bytes(out.tobytes())
    nx, ny, nz = shape
    sx, sy, sz = spacing
    ox, oy, oz = offset
    path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                f"Offset = {ox:.12g} {oy:.12g} {oz:.12g}",
                "CenterOfRotation = 0 0 0",
                "AnatomicalOrientation = RAI",
                f"ElementSpacing = {sx:.12g} {sy:.12g} {sz:.12g}",
                f"DimSize = {nx} {ny} {nz}",
                "ElementType = MET_FLOAT",
                f"DoseUnits = {units}",
                f"ElementDataFile = {raw_path.name}",
                "",
            ]
        ),
        encoding="ascii",
    )
    print(f"Wrote {path}")
    print(f"Wrote {raw_path} ({raw_path.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_mhd", type=Path, help="GPU-frame dose MHD (505 35 417)")
    parser.add_argument("output_mhd", type=Path, help="Patient-frame dose MHD")
    parser.add_argument(
        "--patient-shape",
        nargs=3,
        type=int,
        default=(417, 505, 35),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument(
        "--patient-spacing-mm",
        nargs=3,
        type=float,
        default=(0.5, 0.5, 2.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument(
        "--patient-origin-mm",
        nargs=3,
        type=float,
        default=(-104.0, -126.0, -814.19),
        metavar=("OX", "OY", "OZ"),
        help="First-voxel center in patient/DICOM frame (default matches CT prep + dicom Z)",
    )
    parser.add_argument(
        "--like",
        type=Path,
        help="Resample output to this MHD grid (e.g. ct/physical_dose.mhd)",
    )
    parser.add_argument(
        "--no-flip-x",
        action="store_true",
        help="Do not flip patient X when mapping from GPU Z",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Do not flip patient Y when mapping from GPU X",
    )
    parser.add_argument("--units", default="Gy")
    args = parser.parse_args()

    gpu_meta, gpu = read_mhd(args.gpu_mhd)
    gx, gy, gz = (int(v) for v in gpu_meta["DimSize"].split())
    if (gx, gy, gz) != (505, 35, 417):
        print(
            f"Warning: unexpected GPU DimSize {(gx, gy, gz)}; expected (505, 35, 417)",
            file=sys.stderr,
        )

    px_n, py_n, pz_n = args.patient_shape
    if gz != px_n or gx != py_n or gy != pz_n:
        raise SystemExit(
            f"GPU shape {(gx, gy, gz)} incompatible with patient shape {(px_n, py_n, pz_n)}"
        )

    patient = array.array("f", [0.0]) * (px_n * py_n * pz_n)
    plane = gx * gy
    flip_x = not args.no_flip_x
    flip_y = not args.no_flip_y
    for iz in range(gz):
        patient_x = (px_n - 1 - iz) if flip_x else iz
        for iy in range(gy):
            patient_z = iy
            for ix in range(gx):
                patient_y = (py_n - 1 - ix) if flip_y else ix
                value = gpu[iz * plane + iy * gx + ix]
                if value == 0.0:
                    continue
                patient[patient_z * px_n * py_n + patient_y * px_n + patient_x] = value

    # First-voxel centers.
    sx, sy, sz = args.patient_spacing_mm
    ox, oy, oz = args.patient_origin_mm
    # Convert edge-style origins to centers if the user passed edge coords.
    # Default values are already centers for X/Y ( -104+0.25 ) style? metadata
    # origin is edge; physical_dose Offset is first center. Defaults use DICOM Z
    # center and X/Y edge+half-voxel.
    offset = (ox + 0.0, oy + 0.0, oz + 0.0)
    # If defaults look like edges (-104, -126), shift by half voxel.
    if math.isclose(ox, -104.0) and math.isclose(sx, 0.5):
        offset = (ox + 0.5 * sx, oy + 0.5 * sy, oz)

    if args.like is not None:
        like_meta, _ = read_mhd(args.like)
        ldim = [int(v) for v in like_meta["DimSize"].split()]
        lsp = [float(v) for v in like_meta["ElementSpacing"].split()]
        loff = [float(v) for v in like_meta.get("Offset", "0 0 0").split()]
        lnx, lny, lnz = ldim
        # Block-average fine patient grid into like grid.
        bx = max(1, px_n // lnx)
        by = max(1, py_n // lny)
        bz = max(1, pz_n // lnz)
        coarse = array.array("f", [0.0]) * (lnx * lny * lnz)
        counts = [0] * (lnx * lny * lnz)
        for iz in range(pz_n):
            oz_i = iz // bz
            if oz_i >= lnz:
                continue
            for iy in range(py_n):
                oy_i = iy // by
                if oy_i >= lny:
                    continue
                for ix in range(px_n):
                    ox_i = ix // bx
                    if ox_i >= lnx:
                        continue
                    value = patient[iz * px_n * py_n + iy * px_n + ix]
                    if value == 0.0:
                        continue
                    linear = oz_i * lnx * lny + oy_i * lnx + ox_i
                    coarse[linear] += value
                    counts[linear] += 1
        for i, count in enumerate(counts):
            if count > 1:
                coarse[i] /= float(count)
        write_mhd(
            args.output_mhd,
            coarse,
            (lnx, lny, lnz),
            (lsp[0], lsp[1], lsp[2]),
            (loff[0], loff[1], loff[2]),
            args.units,
        )
        return 0

    write_mhd(
        args.output_mhd,
        patient,
        (px_n, py_n, pz_n),
        (sx, sy, sz),
        offset,
        args.units,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
