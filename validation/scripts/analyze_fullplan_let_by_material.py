#!/usr/bin/env python3
"""Localize full-plan LET_d disagreement by CT material and density.

The GPU CT grid is stored in beam coordinates for the TPS 90-degree, patient
-X field:

    GPU (x, y, z) = patient (y, z, reversed x)

All statistics use the same validation selection as the full-plan LET gamma:
RTSTRUCT BODY and TOPAS dose >= 10% of its BODY maximum.  The script reports
primary-C12, all-hadron, and fragment-only (all minus primary) LET_d errors.
The fragment-only difference is a diagnostic, not a separately scored LET_d.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import read_mhd
from schneider_hu import load_schneider_table


HEADER = struct.Struct("<IIIIIffffff")
MAGIC = 0x47544343


def read_volume(path: Path) -> np.ndarray:
    meta, values = read_mhd(path)
    nx, ny, nz = (int(value) for value in meta["DimSize"].split())
    return np.asarray(values, dtype=np.float64).reshape(nz, ny, nx)


def patient_to_gpu(volume: np.ndarray) -> np.ndarray:
    """Map patient (z,y,x) to TPS-xneg GPU (depth,patient-z,patient-y)."""
    return np.transpose(volume, (2, 0, 1))[::-1, :, :]


def read_ct_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError(f"truncated CCTG header: {path}")
    magic, version, nx, ny, nz, *_ = HEADER.unpack_from(payload)
    if magic != MAGIC or version not in {1, 2, 3}:
        raise ValueError(f"unsupported CCTG magic/version in {path}")
    count = nx * ny * nz
    density_offset = HEADER.size
    material_offset = density_offset + 4 * count
    tail_offset = material_offset + count
    if len(payload) < tail_offset:
        raise ValueError(f"truncated CCTG arrays: {path}")
    shape = (nz, ny, nx)
    density = np.frombuffer(
        payload, dtype="<f4", count=count, offset=density_offset
    ).reshape(shape)
    material = np.frombuffer(
        payload, dtype=np.uint8, count=count, offset=material_offset
    ).reshape(shape)
    return density, material


def metrics(gpu: np.ndarray, topas: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(gpu) & np.isfinite(topas)
    g = gpu[valid]
    t = topas[valid]
    if not g.size:
        return {"voxels": 0}
    delta = g - t
    relative = 100.0 * delta / np.maximum(np.abs(t), 1.0e-12)
    absolute_relative = np.abs(relative)
    return {
        "voxels": int(g.size),
        "topas_mean": float(np.mean(t)),
        "gpu_mean": float(np.mean(g)),
        "mean_bias": float(np.mean(delta)),
        "mean_relative_percent": float(np.mean(relative)),
        "median_absolute_relative_percent": float(np.median(absolute_relative)),
        "p90_absolute_relative_percent": float(np.percentile(absolute_relative, 90)),
        "p95_absolute_relative_percent": float(np.percentile(absolute_relative, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-dir",
        type=Path,
        help="GPU result directory (default: CASE_DIR/gpu)",
    )
    parser.add_argument("--ct-grid", type=Path, required=True)
    parser.add_argument("--schneider-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    args = parser.parse_args()

    gpu_dir = args.gpu_dir if args.gpu_dir else args.case_dir / "gpu"
    topas_dir = args.case_dir / "topas"
    density, section = read_ct_grid(args.ct_grid)
    table = load_schneider_table(args.schneider_file)
    if int(section.max()) >= table.material_class.size:
        raise ValueError(
            f"grid section {int(section.max())} exceeds Schneider table "
            f"size {table.material_class.size}"
        )
    material_class = table.material_class[section]

    dose = patient_to_gpu(read_volume(topas_dir / "dose.mhd"))
    body = patient_to_gpu(read_volume(args.case_dir / "body_mask.mhd")) > 0.5
    topas_primary = patient_to_gpu(read_volume(topas_dir / "let_primary_c12.mhd"))
    topas_all = patient_to_gpu(read_volume(topas_dir / "let_all_hadron.mhd"))
    gpu_primary = read_volume(gpu_dir / "letd_primary_c12.mhd")
    gpu_all = read_volume(gpu_dir / "letd_all_hadron.mhd")

    expected = density.shape
    for label, volume in {
        "dose": dose,
        "body": body,
        "topas_primary": topas_primary,
        "topas_all": topas_all,
        "gpu_primary": gpu_primary,
        "gpu_all": gpu_all,
    }.items():
        if volume.shape != expected:
            raise ValueError(f"{label} shape {volume.shape} != CT grid {expected}")

    selected = body & (
        dose >= args.threshold_percent / 100.0 * float(np.max(dose[body]))
    )
    quantities = {
        "primary_c12": (gpu_primary, topas_primary),
        "all_hadron": (gpu_all, topas_all),
        # This difference localizes the non-primary component responsible for
        # all-hadron disagreement; it is not the ratio of separate LET moments.
        "all_minus_primary_diagnostic": (
            gpu_all - gpu_primary,
            topas_all - topas_primary,
        ),
    }
    class_labels = {0: "air", 1: "lung", 2: "soft_tissue", 3: "bone"}
    report: dict[str, object] = {
        "case_dir": str(args.case_dir),
        "ct_grid": str(args.ct_grid),
        "selection": (
            f"BODY and TOPAS dose >= {args.threshold_percent:g}% of BODY maximum"
        ),
        "selected_voxels": int(selected.sum()),
        "material_classes": {},
        "schneider_sections": {},
        "density_bins_g_cm3": {},
    }
    for class_id, label in class_labels.items():
        region = selected & (material_class == class_id)
        report["material_classes"][label] = {
            "class_id": class_id,
            "voxels": int(region.sum()),
            "fraction_of_selection": float(region.sum() / max(selected.sum(), 1)),
            "density_mean_g_cm3": (
                float(np.mean(density[region])) if np.any(region) else None
            ),
            **{
                name: metrics(gpu, topas, region)
                for name, (gpu, topas) in quantities.items()
            },
        }

    for section_id in range(int(section.max()) + 1):
        region = selected & (section == section_id)
        if not np.any(region):
            continue
        report["schneider_sections"][str(section_id)] = {
            "section_id": section_id,
            "material_class": int(table.material_class[section_id]),
            "voxels": int(region.sum()),
            "fraction_of_selection": float(region.sum() / max(selected.sum(), 1)),
            "density_mean_g_cm3": float(np.mean(density[region])),
            **{
                name: metrics(gpu, topas, region)
                for name, (gpu, topas) in quantities.items()
            },
        }

    density_edges = (0.0, 0.1, 0.5, 0.9, 1.05, 1.2, 1.6, 10.0)
    for low, high in zip(density_edges[:-1], density_edges[1:]):
        label = f"[{low:g},{high:g})"
        region = selected & (density >= low) & (density < high)
        report["density_bins_g_cm3"][label] = {
            "voxels": int(region.sum()),
            "fraction_of_selection": float(region.sum() / max(selected.sum(), 1)),
            **{
                name: metrics(gpu, topas, region)
                for name, (gpu, topas) in quantities.items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
