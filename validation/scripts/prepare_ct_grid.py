#!/usr/bin/env python3
"""Convert DICOM CT under ct/dicom into GPU binary grid (CCTG v3).

Uses TOPAS HUtoMaterialSchneider.txt for density + Schneider section id +
(Z/A)_rel and Bragg mean I for energy-dependent mass-SP vs water.

Transport frame: beam +z, first slice near z=0; xy centered (TOPAS-style).

CCTG origin is the **low edge** of the first voxel (matches GPU ct_sample).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pydicom is required: pip install pydicom") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schneider_hu import hu_to_density_material, load_schneider_table  # noqa: E402

MAGIC = 0x47544343  # 'CCTG'
VERSION = 3


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
    volume = np.stack([item[1] for item in slices], axis=0)
    ds0 = slices[0][2]
    spacing_row_col = [float(v) for v in ds0.PixelSpacing]
    spacing_y = spacing_row_col[0]
    spacing_x = spacing_row_col[1]
    zs = np.array([item[0] for item in slices], dtype=np.float64)
    spacing_z = float(np.median(np.diff(zs))) if len(zs) > 1 else float(
        getattr(ds0, "SliceThickness", 1.0)
    )
    ipp0 = [float(v) for v in ds0.ImagePositionPatient]
    meta = {
        "n_slices": int(volume.shape[0]),
        "rows": int(volume.shape[1]),
        "cols": int(volume.shape[2]),
        "spacing_xyz_mm": [spacing_x, spacing_y, abs(spacing_z)],
        "dicom_x_first": ipp0[0],
        "dicom_y_first": ipp0[1],
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


def write_water_cube(output: Path, size: int = 80, spacing: float = 1.0) -> None:
    """Synthetic unit-density water cube (za_rel=1, I=75 eV)."""
    n = size
    count = n * n * n
    density = np.ones(count, dtype=np.float32)
    material = np.full(count, 0, dtype=np.uint8)
    za = np.array([1.0], dtype=np.float32)
    Iev = np.array([75.0], dtype=np.float32)
    origin = -0.5 * (n - 1) * spacing
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<II", MAGIC, VERSION))
        handle.write(struct.pack("<III", n, n, n))
        handle.write(
            struct.pack(
                "<ffffff",
                float(origin),
                float(origin),
                0.0,
                float(spacing),
                float(spacing),
                float(spacing),
            )
        )
        handle.write(density.tobytes(order="C"))
        handle.write(material.tobytes(order="C"))
        handle.write(struct.pack("<I", 1))
        handle.write(za.tobytes(order="C"))
        handle.write(Iev.tobytes(order="C"))
    print(f"Wrote water cube {output} ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", type=Path, default=Path("ct/dicom"))
    parser.add_argument("--output", type=Path, default=Path("ct/grid/patient_ct.bin"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("ct/grid/patient_ct.metadata.json")
    )
    parser.add_argument("--schneider-file", type=Path, default=None)
    parser.add_argument(
        "--water-cube",
        action="store_true",
        help="Write synthetic water cube instead of DICOM",
    )
    parser.add_argument("--center-xy", action="store_true", default=True)
    args = parser.parse_args()

    if args.water_cube:
        write_water_cube(args.output)
        return

    schneider_path = resolve_schneider_path(args.schneider_file)
    table = load_schneider_table(schneider_path)
    volume_zyx, meta = load_series(args.dicom_dir)
    nz, ny, nx = volume_zyx.shape
    density, section = hu_to_density_material(volume_zyx, table, use_section_id=True)
    spacing_x, spacing_y, spacing_z = meta["spacing_xyz_mm"]

    # TOPAS TsDicomPatient ignores IPP and places the imaging volume so its
    # center is at the component origin (see TOPAS patient docs / README CT section).
    # First-voxel CENTER is then -0.5*(n-1)*spacing on each centered axis.
    # CCTG stores the low EDGE of that first voxel so ct_sample's floor((x-o)/s)
    # returns the correct index.
    if args.center_xy:
        first_center_x = -0.5 * (nx - 1) * spacing_x
        first_center_y = -0.5 * (ny - 1) * spacing_y
    else:
        first_center_x = 0.0
        first_center_y = 0.0
    first_center_z = 0.0  # slice 0 center at z=0 for transport
    origin_x = first_center_x - 0.5 * spacing_x
    origin_y = first_center_y - 0.5 * spacing_y
    origin_z = first_center_z - 0.5 * spacing_z

    density_out = np.ascontiguousarray(density.reshape(-1), dtype=np.float32)
    section_out = np.ascontiguousarray(section.reshape(-1), dtype=np.uint8)
    za = np.ascontiguousarray(table.mass_sp_factor, dtype=np.float32)
    Iev = np.ascontiguousarray(table.mass_sp_I_eV, dtype=np.float32)

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
        handle.write(section_out.tobytes(order="C"))
        handle.write(struct.pack("<I", int(za.size)))
        handle.write(za.tobytes(order="C"))
        handle.write(Iev.tobytes(order="C"))

    ix = int((0.0 - origin_x) / spacing_x)
    iy = int((0.0 - origin_y) / spacing_y)
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    axis = []
    for iz in range(min(nz, 8)):
        sec = int(section[iz, iy, ix])
        si = min(sec, za.size - 1)
        axis.append(
            {
                "z_index": iz,
                "density": float(density[iz, iy, ix]),
                "section": sec,
                "za_rel": float(za[si]),
                "I_eV": float(Iev[si]),
            }
        )

    meta_out = {
        **meta,
        "output": str(args.output),
        "origin_xyz_mm": [origin_x, origin_y, origin_z],
        "origin_convention": "low_edge_of_first_voxel",
        "first_center_xyz_mm": [
            origin_x + 0.5 * spacing_x,
            origin_y + 0.5 * spacing_y,
            origin_z + 0.5 * spacing_z,
        ],
        "dicom_ipp_first_center_xyz_mm": [
            float(meta.get("dicom_x_first", float("nan"))),
            float(meta.get("dicom_y_first", float("nan"))),
            float(meta.get("dicom_z_first", float("nan"))),
        ],
        "hu_conversion": "Schneider",
        "sp_model": "SP_water(E) * f_E(za_rel,I,E) * density",
        "schneider_file": str(schneider_path),
        "grid_version": VERSION,
        "n_material_sections": int(table.n_material_sections),
        "mass_sp_za_rel": [float(x) for x in za],
        "mass_sp_I_eV": [float(x) for x in Iev],
        "section_histogram": {
            str(i): int((section_out == i).sum()) for i in range(int(za.size))
        },
        "density_min": float(density_out.min()),
        "density_max": float(density_out.max()),
        "axis_samples": axis,
        "phantom_length_hint_mm": float(nz * spacing_z),
        "notes": (
            "CCTG v3: Schneider density + section id + (Z/A, I) for energy-dependent "
            "mass-SP vs water. GPU SP = water_table(E) * f_E(za,I,E) * rho."
        ),
    }
    args.metadata.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta_out, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    # fix typo in import error path
    try:
        main()
    except NameError:
        # re-raise with pydicom message if any
        raise
