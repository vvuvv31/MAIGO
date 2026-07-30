#!/usr/bin/env python3
"""Rasterize the RTSTRUCT BODY/External ROI onto the native CT voxel grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw


def read_dicom(directory: Path):
    ct = []
    rtstruct = []
    for path in directory.iterdir():
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(dataset, "Modality", "") == "CT":
            ct.append((path, dataset))
        elif getattr(dataset, "Modality", "") == "RTSTRUCT":
            rtstruct.append((path, dataset))
    if not ct:
        raise ValueError(f"{directory}: no CT slices")
    if len(rtstruct) != 1:
        raise ValueError(f"{directory}: expected one RTSTRUCT, found {len(rtstruct)}")
    return ct, rtstruct[0]


def select_body_roi(dataset, requested: str | None) -> tuple[int, str]:
    rois = {
        int(item.ROINumber): str(item.ROIName)
        for item in dataset.StructureSetROISequence
    }
    if requested:
        matches = [
            (number, name)
            for number, name in rois.items()
            if name.casefold() == requested.casefold()
        ]
    else:
        matches = [
            (number, name)
            for number, name in rois.items()
            if name.strip().casefold() in {"body", "external", "external contour"}
        ]
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely select body ROI; available names: {rois}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicom_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-name")
    args = parser.parse_args()

    ct_items, (rt_path, rt) = read_dicom(args.dicom_dir)
    first = ct_items[0][1]
    row_direction = np.asarray(first.ImageOrientationPatient[:3], dtype=np.float64)
    column_direction = np.asarray(first.ImageOrientationPatient[3:], dtype=np.float64)
    normal = np.cross(row_direction, column_direction)
    ct_items.sort(
        key=lambda item: float(
            np.dot(np.asarray(item[1].ImagePositionPatient, dtype=np.float64), normal)
        )
    )
    rows = int(first.Rows)
    columns = int(first.Columns)
    row_spacing = float(first.PixelSpacing[0])
    column_spacing = float(first.PixelSpacing[1])
    positions = np.asarray(
        [
            np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
            for _, dataset in ct_items
        ]
    )
    slice_coordinates = positions @ normal
    slice_spacing = (
        float(np.median(np.diff(slice_coordinates)))
        if len(ct_items) > 1
        else float(first.SliceThickness)
    )
    uid_to_slice = {
        str(dataset.SOPInstanceUID): index
        for index, (_, dataset) in enumerate(ct_items)
    }

    roi_number, roi_name = select_body_roi(rt, args.roi_name)
    roi_contours = [
        item
        for item in rt.ROIContourSequence
        if int(item.ReferencedROINumber) == roi_number
    ]
    if len(roi_contours) != 1:
        raise ValueError(f"ROI {roi_name}: expected one ROIContourSequence entry")

    mask = np.zeros((len(ct_items), rows, columns), dtype=np.uint8)
    contour_count = 0
    fallback_slice_matches = 0
    for contour in getattr(roi_contours[0], "ContourSequence", []):
        points = np.asarray(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] < 3:
            continue
        slice_index = None
        references = getattr(contour, "ContourImageSequence", [])
        if references:
            slice_index = uid_to_slice.get(str(references[0].ReferencedSOPInstanceUID))
        if slice_index is None:
            coordinate = float(np.mean(points @ normal))
            slice_index = int(np.argmin(np.abs(slice_coordinates - coordinate)))
            if abs(slice_coordinates[slice_index] - coordinate) > 0.51 * abs(slice_spacing):
                raise ValueError(
                    f"ROI contour at {coordinate:g} does not match a CT slice"
                )
            fallback_slice_matches += 1

        ipp = positions[slice_index]
        relative = points - ipp
        columns_float = relative @ row_direction / column_spacing
        rows_float = relative @ column_direction / row_spacing
        polygon = list(zip(columns_float.tolist(), rows_float.tolist()))
        image = Image.new("1", (columns, rows), 0)
        ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
        mask[slice_index] |= np.asarray(image, dtype=np.uint8)
        contour_count += 1

    if not np.any(mask):
        raise ValueError(f"ROI {roi_name}: rasterized mask is empty")
    contour_slices = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
    interpolated_slices = 0
    # Many clinical External ROIs are stored only every second/third CT slice.
    # For an exterior body mask, nearest-contour interpolation is preferable to
    # treating those intervening anatomical slices as outside air.
    for slice_index in range(int(contour_slices[0]), int(contour_slices[-1]) + 1):
        if np.any(mask[slice_index]):
            continue
        nearest = int(contour_slices[np.argmin(np.abs(contour_slices - slice_index))])
        mask[slice_index] = mask[nearest]
        interpolated_slices += 1
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_suffix(".raw")
    mask.astype("<f4").tofile(raw)
    origin = positions[0]
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
                f"ElementSpacing = {column_spacing:.12g} {row_spacing:.12g} {slice_spacing:.12g}",
                f"DimSize = {columns} {rows} {len(ct_items)}",
                "ElementType = MET_FLOAT",
                "MaskMeaning = RTSTRUCT external body contour",
                f"ElementDataFile = {raw.name}",
                "",
            ]
        ),
        encoding="ascii",
    )
    occupied = np.argwhere(mask > 0)
    metrics = {
        "dicom_directory": str(args.dicom_dir),
        "rtstruct": str(rt_path),
        "roi_number": roi_number,
        "roi_name": roi_name,
        "shape_zyx": list(mask.shape),
        "spacing_xyz_mm": [column_spacing, row_spacing, slice_spacing],
        "contours": contour_count,
        "slices_with_explicit_contours": int(contour_slices.size),
        "nearest_interpolated_slices": interpolated_slices,
        "fallback_slice_matches": fallback_slice_matches,
        "body_voxels": int(mask.sum()),
        "body_volume_cm3": float(
            mask.sum() * column_spacing * row_spacing * abs(slice_spacing) / 1000.0
        ),
        "occupied_bbox_zyx": [
            occupied.min(axis=0).tolist(),
            occupied.max(axis=0).tolist(),
        ],
        "occupied_slices": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).any(axis=1))),
        "output_mhd": str(output),
    }
    metrics_path = output.with_name(output.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
