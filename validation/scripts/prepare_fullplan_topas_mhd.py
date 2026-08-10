#!/usr/bin/env python3
"""Convert TOPAS full-plan binary dose/LET scorers to patient-axis MHD."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


AXIS_RE = re.compile(
    r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins?\s+of\s+"
    r"([0-9.eE+-]+)\s+(mm|cm)\s*$"
)
SCORERS = {
    "dose": ("OSMK_Dtotal_full_plan", "DoseUnits = Gy"),
    "let_primary_c12": (
        "LET_primary_c12_full_plan",
        "LETUnits = MeV/mm/(g/cm3)",
    ),
    "let_all_hadron": (
        "LET_all_hadron_full_plan",
        "LETUnits = MeV/mm/(g/cm3)",
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def geometry(header: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    axes: dict[str, tuple[int, float]] = {}
    for line in header.read_text(encoding="utf-8").splitlines():
        match = AXIS_RE.match(line.strip())
        if match:
            axis, count, width, unit = match.groups()
            axes[axis] = (int(count), float(width) * (10.0 if unit == "cm" else 1.0))
    if set(axes) != set("XYZ"):
        raise ValueError(f"{header}: incomplete XYZ geometry")
    return (
        (axes["X"][0], axes["Y"][0], axes["Z"][0]),
        (axes["X"][1], axes["Y"][1], axes["Z"][1]),
    )


def infer_dtype(path: Path, count: int) -> np.dtype:
    if path.stat().st_size == count * 8:
        return np.dtype("<f8")
    if path.stat().st_size == count * 4:
        return np.dtype("<f4")
    raise ValueError(f"{path}: binary size does not match {count} float values")


def write_mhd(
    source: Path,
    header: Path,
    output: Path,
    origin: tuple[float, float, float],
    units: str,
) -> dict[str, object]:
    shape, spacing = geometry(header)
    count = int(np.prod(shape))
    dtype = infer_dtype(source, count)
    values = np.memmap(source, dtype=dtype, mode="r", shape=(shape[2], shape[1], shape[0]))
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_suffix(".raw")
    np.asarray(values, dtype="<f4").tofile(raw)
    output.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                f"Offset = {' '.join(f'{v:.12g}' for v in origin)}",
                "CenterOfRotation = 0 0 0",
                f"ElementSpacing = {' '.join(f'{v:.12g}' for v in spacing)}",
                f"DimSize = {' '.join(str(v) for v in shape)}",
                "ElementType = MET_FLOAT",
                units,
                f"ElementDataFile = {raw.name}",
                "",
            ]
        ),
        encoding="ascii",
    )
    positive = values[values > 0]
    return {
        "source": str(source),
        "source_sha256": digest(source),
        "header": str(header),
        "shape_xyz": list(shape),
        "spacing_xyz_mm": list(spacing),
        "offset_first_center_xyz_mm": list(origin),
        "dtype": dtype.name,
        "voxel_count": count,
        "nonzero_voxels": int(np.count_nonzero(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "sum": float(np.sum(values, dtype=np.float64)),
        "positive_mean": float(np.mean(positive)) if positive.size else 0.0,
        "output_mhd": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--ct-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.ct_metadata.read_text(encoding="utf-8"))
    edge = tuple(float(v) for v in metadata["origin_xyz_mm"])
    spacing = tuple(float(v) for v in metadata["spacing_xyz_mm"])
    origin = tuple(edge[i] + 0.5 * spacing[i] for i in range(3))

    report: dict[str, object] = {
        "case_dir": str(args.case_dir),
        "ct_metadata": str(args.ct_metadata),
        "scorers": {},
    }
    for name, (stem, units) in SCORERS.items():
        report["scorers"][name] = write_mhd(
            args.case_dir / f"{stem}.bin",
            args.case_dir / f"{stem}.binheader",
            args.output_dir / f"{name}.mhd",
            origin,
            units,
        )
    report_path = args.output_dir / "conversion_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
