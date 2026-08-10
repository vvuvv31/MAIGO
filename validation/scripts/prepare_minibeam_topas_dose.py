#!/usr/bin/env python3
"""Convert dense TOPAS minibeam DoseToMedium binary output to GPU-axis MHD.

TOPAS writes X fastest with a logical array (topas_z, topas_y, topas_x).
The minibeam water beam travels along TOPAS world +Y, so the GPU/MHD mapping is:

    GPU x = TOPAS X
    GPU y = TOPAS Z
    GPU z = TOPAS Y - water entrance Y

The output RAW therefore has the project's normal z-major, y, x-fastest layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


AXIS_RE = re.compile(
    r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins?\s+of\s+"
    r"([0-9.eE+-]+)\s+(mm|cm)\s*$"
)


@dataclass(frozen=True)
class TopasDoseGeometry:
    bins_x: int
    bins_y: int
    bins_z: int
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float

    @property
    def count(self) -> int:
        return self.bins_x * self.bins_y * self.bins_z


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_binheader(path: Path) -> TopasDoseGeometry:
    axes: dict[str, tuple[int, float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = AXIS_RE.match(raw_line.strip())
        if not match:
            continue
        axis, count, spacing, unit = match.groups()
        spacing_mm = float(spacing) * (10.0 if unit == "cm" else 1.0)
        axes[axis] = (int(count), spacing_mm)
    missing = sorted(set("XYZ") - axes.keys())
    if missing:
        raise ValueError(f"{path}: missing axis definitions: {', '.join(missing)}")
    return TopasDoseGeometry(
        bins_x=axes["X"][0],
        bins_y=axes["Y"][0],
        bins_z=axes["Z"][0],
        spacing_x_mm=axes["X"][1],
        spacing_y_mm=axes["Y"][1],
        spacing_z_mm=axes["Z"][1],
    )


def infer_dtype(path: Path, geometry: TopasDoseGeometry) -> np.dtype:
    size = path.stat().st_size
    if size == geometry.count * np.dtype("<f8").itemsize:
        return np.dtype("<f8")
    if size == geometry.count * np.dtype("<f4").itemsize:
        return np.dtype("<f4")
    raise ValueError(
        f"{path}: {size} bytes does not match {geometry.count} float32/float64 values"
    )


def write_mhd(
    output_mhd: Path,
    input_binary: Path,
    geometry: TopasDoseGeometry,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    dtype = infer_dtype(input_binary, geometry)
    native = np.memmap(
        input_binary,
        dtype=dtype,
        mode="r",
        shape=(geometry.bins_z, geometry.bins_y, geometry.bins_x),
    )
    if not np.isfinite(native).all():
        raise ValueError(f"{input_binary}: non-finite dose values")
    minimum = float(np.min(native))
    if minimum < 0.0:
        raise ValueError(f"{input_binary}: negative dose value {minimum}")

    output_mhd.parent.mkdir(parents=True, exist_ok=True)
    output_raw = output_mhd.with_suffix(".raw")
    gpu_shape = (geometry.bins_y, geometry.bins_z, geometry.bins_x)
    gpu = np.memmap(output_raw, dtype="<f4", mode="w+", shape=gpu_shape)
    # Copy one TOPAS-Z slab at a time. This avoids materializing a second
    # 320 MB float64 array while transposing to GPU (depth, y, x).
    for topas_z in range(geometry.bins_z):
        gpu[:, topas_z, :] = native[topas_z, :, :]
    gpu.flush()
    del gpu

    depth_profile = np.asarray(native.sum(axis=(0, 2)), dtype=np.float64)
    lateral_x_profile = np.asarray(native.sum(axis=(0, 1)), dtype=np.float64)
    lateral_y_profile = np.asarray(native.sum(axis=(1, 2)), dtype=np.float64)

    maximum_flat = int(np.argmax(native))
    topas_z, topas_y, topas_x = np.unravel_index(maximum_flat, native.shape)
    maximum = float(native[topas_z, topas_y, topas_x])
    total = float(np.sum(native, dtype=np.float64))
    nonzero = int(np.count_nonzero(native))

    origin_x = (
        0.5 * geometry.spacing_x_mm
        - 0.5 * geometry.bins_x * geometry.spacing_x_mm
    )
    origin_y = (
        0.5 * geometry.spacing_z_mm
        - 0.5 * geometry.bins_z * geometry.spacing_z_mm
    )
    origin_z = 0.5 * geometry.spacing_y_mm
    header = "\n".join(
        (
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            f"Offset = {origin_x:.12g} {origin_y:.12g} {origin_z:.12g}",
            "CenterOfRotation = 0 0 0",
            (
                f"ElementSpacing = {geometry.spacing_x_mm:.12g} "
                f"{geometry.spacing_z_mm:.12g} {geometry.spacing_y_mm:.12g}"
            ),
            (
                f"DimSize = {geometry.bins_x} {geometry.bins_z} "
                f"{geometry.bins_y}"
            ),
            "ElementType = MET_FLOAT",
            "DoseUnits = Gy",
            f"ElementDataFile = {output_raw.name}",
            "",
        )
    )
    output_mhd.write_text(header, encoding="ascii", newline="\n")

    metrics: dict[str, object] = {
        "input_dtype": dtype.name,
        "input_shape_topas_zyx": [
            geometry.bins_z,
            geometry.bins_y,
            geometry.bins_x,
        ],
        "output_shape_gpu_zyx": list(gpu_shape),
        "spacing_gpu_xyz_mm": [
            geometry.spacing_x_mm,
            geometry.spacing_z_mm,
            geometry.spacing_y_mm,
        ],
        "offset_first_voxel_center_gpu_xyz_mm": [origin_x, origin_y, origin_z],
        "maximum_dose_Gy": maximum,
        "maximum_index_gpu_zyx": [int(topas_y), int(topas_z), int(topas_x)],
        "maximum_position_gpu_xyz_mm": [
            origin_x + topas_x * geometry.spacing_x_mm,
            origin_y + topas_z * geometry.spacing_z_mm,
            origin_z + topas_y * geometry.spacing_y_mm,
        ],
        "summed_voxel_dose_Gy": total,
        "nonzero_voxels": nonzero,
        "voxel_count": geometry.count,
        "depth_profile_max_center_mm": (
            int(np.argmax(depth_profile)) + 0.5
        )
        * geometry.spacing_y_mm,
    }
    return (
        output_raw,
        depth_profile,
        lateral_x_profile,
        lateral_y_profile,
        metrics,
    )


def periodicity_mm(profile: np.ndarray, spacing_mm: float) -> tuple[int, float]:
    centered = np.asarray(profile, dtype=np.float64) - float(np.mean(profile))
    spectrum = np.abs(np.fft.rfft(centered))
    # Suppress the broad field envelope. With the reference 100 mm FOV, k>=10
    # means periods <=10 mm and cleanly isolates the minibeam modulation.
    first = min(10, max(1, spectrum.size - 1))
    harmonic = int(np.argmax(spectrum[first:]) + first)
    if harmonic <= 0 or spectrum[harmonic] <= 0.0:
        return 0, math.inf
    return harmonic, profile.size * spacing_mm / harmonic


def write_profile(path: Path, coordinate_name: str, coordinates: np.ndarray,
                  values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([coordinate_name, "summed_voxel_dose_Gy"])
        writer.writerows(zip(coordinates, values, strict=True))


def convert(input_binary: Path, binheader: Path, output_mhd: Path,
            metrics_json: Path, expected_pitch_mm: float | None) -> dict[str, object]:
    geometry = parse_binheader(binheader)
    (
        output_raw,
        depth_profile,
        lateral_x_profile,
        lateral_y_profile,
        metrics,
    ) = write_mhd(output_mhd, input_binary, geometry)

    harmonic, pitch_mm = periodicity_mm(
        lateral_x_profile, geometry.spacing_x_mm
    )
    metrics.update(
        {
            "source_binary": input_binary.as_posix(),
            "source_binheader": binheader.as_posix(),
            "source_binary_sha256": sha256(input_binary),
            "source_binheader_sha256": sha256(binheader),
            "output_mhd": output_mhd.as_posix(),
            "output_raw": output_raw.as_posix(),
            "output_raw_sha256": sha256(output_raw),
            "lateral_x_dominant_harmonic": harmonic,
            "lateral_x_dominant_period_mm": pitch_mm,
        }
    )
    if expected_pitch_mm is not None:
        pitch_error = pitch_mm - expected_pitch_mm
        metrics["expected_slit_pitch_mm"] = expected_pitch_mm
        metrics["slit_pitch_error_mm"] = pitch_error
        if not math.isfinite(pitch_mm) or abs(pitch_error) > 0.25:
            raise ValueError(
                f"dominant period {pitch_mm:.6g} mm is inconsistent with "
                f"expected slit pitch {expected_pitch_mm:.6g} mm"
            )

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    base = metrics_json.with_suffix("")
    depth_coordinates = (
        np.arange(geometry.bins_y, dtype=np.float64) + 0.5
    ) * geometry.spacing_y_mm
    x_coordinates = (
        np.arange(geometry.bins_x, dtype=np.float64) + 0.5
    ) * geometry.spacing_x_mm - 0.5 * geometry.bins_x * geometry.spacing_x_mm
    y_coordinates = (
        np.arange(geometry.bins_z, dtype=np.float64) + 0.5
    ) * geometry.spacing_z_mm - 0.5 * geometry.bins_z * geometry.spacing_z_mm
    write_profile(
        base.parent / f"{base.name}_depth.csv",
        "depth_mm",
        depth_coordinates,
        depth_profile,
    )
    write_profile(
        base.parent / f"{base.name}_x.csv",
        "x_mm",
        x_coordinates,
        lateral_x_profile,
    )
    write_profile(
        base.parent / f"{base.name}_y.csv",
        "y_mm",
        y_coordinates,
        lateral_y_profile,
    )

    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--binheader", type=Path, required=True)
    parser.add_argument("--output-mhd", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument(
        "--expected-slit-pitch-mm",
        type=float,
        default=3.6,
        help="Set a non-positive value to disable the periodicity check.",
    )
    args = parser.parse_args()
    expected = (
        args.expected_slit_pitch_mm
        if args.expected_slit_pitch_mm > 0.0
        else None
    )
    metrics = convert(
        args.input,
        args.binheader,
        args.output_mhd,
        args.metrics_json,
        expected,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
