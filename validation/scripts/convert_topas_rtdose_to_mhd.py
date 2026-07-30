#!/usr/bin/env python3
"""Convert a TOPAS RTDOSE DICOM scorer to a patient-coordinate MHD volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dicom", type=Path)
    parser.add_argument("output_mhd", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()

    dataset = pydicom.dcmread(str(args.input_dicom))
    if str(dataset.Modality) != "RTDOSE":
        raise ValueError(f"expected RTDOSE, got {dataset.Modality!r}")
    if str(dataset.DoseUnits).upper() != "GY":
        raise ValueError(f"expected DoseUnits=GY, got {dataset.DoseUnits!r}")

    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
    identity_orientation = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float64)
    if not np.allclose(orientation, identity_orientation, atol=1.0e-8):
        raise ValueError(
            "only identity patient orientation is currently supported; "
            f"got {orientation.tolist()}"
        )

    values = (
        dataset.pixel_array.astype(np.float32)
        * np.float32(dataset.DoseGridScaling)
    )
    if values.ndim != 3:
        raise ValueError(f"expected frames×rows×columns dose, got {values.shape}")
    frames, rows, columns = values.shape
    frame_offsets = np.asarray(dataset.GridFrameOffsetVector, dtype=np.float64)
    if frame_offsets.size != frames:
        raise ValueError(
            f"frame offset count {frame_offsets.size} != frames {frames}"
        )
    if frames > 1:
        frame_spacing = float(np.median(np.diff(frame_offsets)))
        if not np.allclose(
            np.diff(frame_offsets), frame_spacing, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("nonuniform GridFrameOffsetVector is unsupported")
    else:
        frame_spacing = float(getattr(dataset, "SliceThickness", 1.0))

    # DICOM PixelSpacing is row, column. MHD spacing is x, y, z.
    row_spacing, column_spacing = (
        float(value) for value in dataset.PixelSpacing
    )
    spacing_xyz = (column_spacing, row_spacing, frame_spacing)
    origin = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
    origin[2] += frame_offsets[0]

    args.output_mhd.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_mhd.with_suffix(".raw")
    values.astype("<f4", copy=False).tofile(raw_path)
    args.output_mhd.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                f"Offset = {origin[0]:.12g} {origin[1]:.12g} {origin[2]:.12g}",
                "CenterOfRotation = 0 0 0",
                "AnatomicalOrientation = RAI",
                f"ElementSpacing = {spacing_xyz[0]:.12g} "
                f"{spacing_xyz[1]:.12g} {spacing_xyz[2]:.12g}",
                f"DimSize = {columns} {rows} {frames}",
                "ElementType = MET_FLOAT",
                f"ElementDataFile = {raw_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = {
        "input_dicom": str(args.input_dicom),
        "output_mhd": str(args.output_mhd),
        "shape_zyx": [frames, rows, columns],
        "spacing_xyz_mm": list(spacing_xyz),
        "origin_xyz_mm": origin.tolist(),
        "dose_grid_scaling": float(dataset.DoseGridScaling),
        "maximum_dose_Gy": float(values.max()),
        "integral_voxel_dose_Gy": float(values.sum(dtype=np.float64)),
        "nonzero_voxels": int(np.count_nonzero(values)),
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
