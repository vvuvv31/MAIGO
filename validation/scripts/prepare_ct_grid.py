#!/usr/bin/env python3
"""Convert a DICOM CT series under ct/dicom into a GPU binary grid (7c).

HU → density / material class uses TOPAS HUtoMaterialSchneider.txt (same table
as TsDicomPatient), not a hand-written piecewise curve.

Transport frame: beam +z, entrance near z=0; CT first slice maps to z=0.
xy is centered on the CT volume.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pydicom is required: pip install pydicom") from exc

from schneider_hu import load_schneider_table, hu_to_density_material

MAGIC = 0x47544343  # 'CCTG'
VERSION = 1


def load_series(dicom_dir: Path) -> tuple[np.ndarray, dict]:
    files = sorted(dicom_dir.glob("*.dcm")) + sorted(dicom_dir.glob("IMG*"))
    files = [f for f in files if f.suffix.lower() in {".dcm", ""} or f.name.startswith("IMG")]
    files = sorted(set(files), key=lambda p: p.name)
    if not files:
        raise SystemExit(f"No DICOM files in {dicom_dir}")

    slices: list[tuple[float, np.ndarray, object]] = []
    for path in files:
        ds = pydicom.dcmread(str(path), force=True)
        if not hasattr(ds, "PixelData"):
            continue
        z = float(ds.ImagePositionPatient[2])
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        slices.append((z, arr, ds))
    if not slices:
        raise SystemExit("No pixel slices found")
    slices.sort(key=lambda item: item[0])
    volume = np.stack([item[1] for item in slices], axis=0)  # z,y,x
    ds0 = slices[0][2]
    spacing_row_col = [float(v) for v in ds0.PixelSpacing]
    spacing_y = spacing_row_col[0]
    spacing_x = spacing_row_col[1]
    zs = np.array([item[0] for item in slices], dtype=np.float64)
    spacing_z = float(np.median(np.diff(zs))) if len(zs) > 1 else float(
        getattr(ds0, "SliceThickness", 1.0)
    )
    meta = {
        "n_slices": int(volume.shape[0]),
        "rows": int(volume.shape[1]),
        "cols": int(volume.shape[2]),
        "spacing_xyz_mm": [spacing_x, spacing_y, abs(spacing_z)],
        "dicom_z_first": float(zs[0]),
        "dicom_z_last": float(zs[-1]),
        "hu_min": float(volume.min()),
        "hu_max": float(volume.max()),
        "hu_mean": float(volume.mean()),
    }
    return volume, meta


def resolve_schneider_path(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("ct/HUtoMaterialSchneider.txt"),
            Path("validation/topas/HUtoMaterialSchneider.txt"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(
        "Schneider HU table not found. Place HUtoMaterialSchneider.txt under "
        "ct/ or pass --schneider-file"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", type=Path, default=Path("ct/dicom"))
    parser.add_argument("--output", type=Path, default=Path("ct/grid/patient_ct.bin"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("ct/grid/patient_ct.metadata.json")
    )
    parser.add_argument(
        "--schneider-file",
        type=Path,
        default=None,
        help="TOPAS Schneider HU table (default: ct/HUtoMaterialSchneider.txt)",
    )
    parser.add_argument(
        "--center-xy",
        action="store_true",
        default=True,
        help="Center CT in x/y about origin (default true)",
    )
    args = parser.parse_args()

    schneider_path = resolve_schneider_path(args.schneider_file)
    table = load_schneider_table(schneider_path)

    volume_zyx, meta = load_series(args.dicom_dir)
    nz, ny, nx = volume_zyx.shape
    density, material = hu_to_density_material(volume_zyx, table)
    spacing_x, spacing_y, spacing_z = meta["spacing_xyz_mm"]

    origin_x = -0.5 * (nx - 1) * spacing_x if args.center_xy else 0.0
    origin_y = -0.5 * (ny - 1) * spacing_y if args.center_xy else 0.0
    origin_z = 0.0

    density_out = np.ascontiguousarray(density.reshape(-1), dtype=np.float32)
    material_out = np.ascontiguousarray(material.reshape(-1), dtype=np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<II", MAGIC, VERSION))
        handle.write(struct.pack("<III", nx, ny, nz))
        handle.write(
            struct.pack(
                "<ffffff",
                float(origin_x),
                float(origin_y),
                float(origin_z),
                float(spacing_x),
                float(spacing_y),
                float(spacing_z),
            )
        )
        handle.write(density_out.tobytes(order="C"))
        handle.write(material_out.tobytes(order="C"))

    # Sample axis density for QA
    ix = int((0.0 - origin_x) / spacing_x)
    iy = int((0.0 - origin_y) / spacing_y)
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    axis_dens = [
        float(density[iz, iy, ix]) for iz in range(min(nz, 8))
    ]

    meta_out = {
        **meta,
        "output": str(args.output),
        "origin_xyz_mm": [origin_x, origin_y, origin_z],
        "hu_conversion": "Schneider",
        "schneider_file": str(schneider_path),
        "material_map": {"0": "air", "1": "lung", "2": "water", "3": "bone"},
        "material_histogram": {
            str(i): int((material_out == i).sum()) for i in range(4)
        },
        "density_min": float(density_out.min()),
        "density_max": float(density_out.max()),
        "axis_density_first_slices": axis_dens,
        "phantom_length_hint_mm": float(nz * spacing_z),
        "notes": (
            "HU→ρ/material from TOPAS HUtoMaterialSchneider.txt (same as TsDicomPatient). "
            "Material class collapses Schneider tissues to air/lung/water/bone for GPU tables. "
            "Transport frame: z=0 at first DICOM slice; xy centered."
        ),
    }
    args.metadata.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta_out, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
