#!/usr/bin/env python3
"""Convert sparse TOPAS or CarbonGPU 3D dose CSV to MetaImage MHD/RAW."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from array import array
from pathlib import Path


def parse_triplet(values: list[str], name: str, cast):
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} requires exactly three values")
    return tuple(cast(value) for value in values)


def read_sparse(
    path: Path,
    shape: tuple[int, int, int],
    input_format: str,
    value_column: int,
    scale: float,
) -> array:
    nx, ny, nz = shape
    values = array("f", [0.0]) * (nx * ny * nz)
    seen: set[int] = set()
    rows = 0

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(line for line in stream if line.strip() and not line.startswith("#"))
        for fields in reader:
            fields = [field.strip() for field in fields]
            if input_format == "gpu" and rows == 0 and fields[0].lower() == "ix":
                rows += 1
                continue
            if len(fields) <= value_column:
                raise ValueError(
                    f"{path}: expected value column {value_column}, found {len(fields)} columns"
                )
            try:
                ix, iy, iz = (int(fields[index]) for index in range(3))
                value = float(fields[value_column]) * scale
            except ValueError as exc:
                raise ValueError(f"{path}: invalid row: {fields}") from exc
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                raise ValueError(f"{path}: voxel index {(ix, iy, iz)} outside {shape}")
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite dose at {(ix, iy, iz)}")
            linear = iz * nx * ny + iy * nx + ix
            if linear in seen:
                raise ValueError(f"{path}: duplicate voxel index {(ix, iy, iz)}")
            seen.add(linear)
            values[linear] = value
            rows += 1

    if sys.byteorder != "little":
        values.byteswap()
    print(f"Read {len(seen)} nonzero voxels from {path}")
    return values


def write_mhd(
    output: Path,
    values: array,
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float],
    units: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw")
    with raw_path.open("wb") as stream:
        values.tofile(stream)

    nx, ny, nz = shape
    sx, sy, sz = spacing_mm
    ox, oy, oz = origin_mm
    header = "\n".join(
        (
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            f"Offset = {ox:.12g} {oy:.12g} {oz:.12g}",
            "CenterOfRotation = 0 0 0",
            f"ElementSpacing = {sx:.12g} {sy:.12g} {sz:.12g}",
            f"DimSize = {nx} {ny} {nz}",
            "ElementType = MET_FLOAT",
            f"DoseUnits = {units}",
            f"ElementDataFile = {raw_path.name}",
            "",
        )
    )
    output.write_text(header, encoding="ascii", newline="\n")
    print(f"Wrote {output}")
    print(f"Wrote {raw_path} ({raw_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-format", choices=("topas", "gpu"), required=True)
    parser.add_argument("--shape", nargs=3, default=(50, 50, 50), metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--spacing-mm", nargs=3, default=(3.0, 3.0, 3.0), metavar=("SX", "SY", "SZ")
    )
    parser.add_argument(
        "--origin-mm",
        nargs=3,
        default=(-73.5, -73.5, 1.5),
        metavar=("OX", "OY", "OZ"),
        help="physical coordinate of the first voxel center",
    )
    parser.add_argument(
        "--value-column",
        type=int,
        help="zero-based value column; defaults to 3 for TOPAS and 6 for GPU",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every input value by this factor",
    )
    parser.add_argument("--units", default="Gy")
    args = parser.parse_args()

    shape = parse_triplet(args.shape, "--shape", int)
    spacing = parse_triplet(args.spacing_mm, "--spacing-mm", float)
    origin = parse_triplet(args.origin_mm, "--origin-mm", float)
    if any(value <= 0 for value in shape) or any(value <= 0.0 for value in spacing):
        parser.error("shape and spacing values must be positive")
    value_column = args.value_column
    if value_column is None:
        value_column = 3 if args.input_format == "topas" else 6

    values = read_sparse(args.input, shape, args.input_format, value_column, args.scale)
    write_mhd(args.output, values, shape, spacing, origin, args.units)


if __name__ == "__main__":
    main()
