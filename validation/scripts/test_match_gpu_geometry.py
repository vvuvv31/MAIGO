#!/usr/bin/env python3
"""Unit tests for GPU↔physical dose mapping geometry (no free translation).

Locks the production convention:
  - flip_x=True maps GPU z=0 → patient X max (xneg reorient entrance face)
  - flip_x=False leaves a large COM offset along patient X (depth)
  - beam_y maps GPU (x,y,z) → patient (x,z,y) for TPS 0° lung cases
  - Round-trip of a synthetic field through map_gpu_to_physical is lossless
    on the coarse grid (block-average identity for constant blocks)
"""

from __future__ import annotations

import array
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_gpu_to_physical_dose import map_gpu_to_physical  # noqa: E402
from rebin_topas_patient_dose import rebin  # noqa: E402


def _com_x(values: list[float], shape: tuple[int, int, int], thr: float) -> float:
    nx, ny, nz = shape
    sx = w = 0.0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v = values[iz * nx * ny + iy * nx + ix]
                if v < thr:
                    continue
                sx += ix * v
                w += v
    return sx / w if w else float("nan")


def test_flip_x_required_for_depth_face() -> None:
    """A dose peak near GPU z=0 must land near patient X max when flip_x=True."""
    # GPU shape: (patient_y=8, patient_z=4, patient_x=16) → phys 2x1x4 with 4x4x1
    gx, gy, gz = 8, 4, 16
    patient = (16, 8, 4)
    phys = (4, 2, 4)
    gpu = array.array("f", [0.0] * (gx * gy * gz))
    # Put all dose in the first two GPU-z planes (entrance for xneg).
    plane = gx * gy
    for iz in (0, 1):
        for iy in range(gy):
            for ix in range(gx):
                gpu[iz * plane + iy * gx + ix] = 1.0

    mapped_fx = map_gpu_to_physical(gpu, (gx, gy, gz), patient, phys, True, False)
    mapped_nofx = map_gpu_to_physical(gpu, (gx, gy, gz), patient, phys, False, False)

    # With flip_x, entrance planes map to high patient-X coarse bins (ox near 3).
    com_fx = _com_x(mapped_fx, phys, 0.1)
    com_nofx = _com_x(mapped_nofx, phys, 0.1)
    assert com_fx > 2.5, f"flip_x should put entrance near high X, got COM_x={com_fx}"
    assert com_nofx < 1.5, f"no flip_x should put entrance near low X, got COM_x={com_nofx}"
    assert com_fx - com_nofx > 1.5, "flip_x must materially move depth COM"


def test_block_avg_constant_field() -> None:
    """Constant GPU field → constant phys field after 4× block average."""
    gx, gy, gz = 8, 4, 16
    patient = (16, 8, 4)
    phys = (4, 2, 4)
    gpu = array.array("f", [2.5] * (gx * gy * gz))
    mapped = map_gpu_to_physical(gpu, (gx, gy, gz), patient, phys, True, False)
    assert all(abs(v - 2.5) < 1e-5 for v in mapped if v != 0.0)
    # All voxels should be filled for full-coverage constant field.
    assert all(v > 0.0 for v in mapped)


def test_sparse_block_average_includes_zero_voxels() -> None:
    """A sparse hit is divided by the full geometric block volume."""
    # Each output voxel covers 2 patient-X × 2 patient-Y × 1 patient-Z.
    gpu = array.array("f", [0.0] * 16)
    gpu[0] = 1.0
    mapped = map_gpu_to_physical(
        gpu, (4, 1, 4), (4, 4, 1), (2, 2, 1), False, False
    )
    assert abs(mapped[0] - 0.25) < 1e-7, mapped

    topas = array.array("f", [0.0] * 16)
    topas[0] = 1.0
    rebinned = rebin(topas, (4, 4, 1), (2, 2, 1))
    assert abs(rebinned[0] - 0.25) < 1e-7, rebinned


def test_beam_y_axis_mapping() -> None:
    """beam_y must map GPU (x,y,z) to patient (x,z,y) without a free shift."""
    # Patient shape (X,Y,Z)=(4,6,2), hence GPU shape=(4,2,6).
    patient = (4, 6, 2)
    gpu_shape = (4, 2, 6)
    gpu = array.array("f", [0.0] * math.prod(gpu_shape))
    # GPU (x=3,y=1,z=4) must become patient (x=3,y=4,z=1).
    gx, gy, _ = gpu_shape
    gpu[4 * gx * gy + 1 * gx + 3] = 8.0
    mapped = map_gpu_to_physical(
        gpu,
        gpu_shape,
        patient,
        patient,
        flip_x=False,
        flip_y=False,
        mapping="beam_y",
    )
    expected = 1 * patient[0] * patient[1] + 4 * patient[0] + 3
    assert mapped[expected] == 8.0, mapped[expected]
    assert sum(mapped) == 8.0


def test_gamma_identity_on_self() -> None:
    """Self-match of a smooth blob must pass global 3%/3mm at thr 10%."""
    from match_gpu_to_physical_dose import fit_scale, gamma_3d

    nx, ny, nz = 20, 20, 10
    n = nx * ny * nz
    ref = array.array("f", [0.0] * n)
    # Gaussian-like blob at center
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                r2 = (ix - 10) ** 2 + (iy - 10) ** 2 + (iz - 5) ** 2
                ref[iz * nx * ny + iy * nx + ix] = math.exp(-r2 / 18.0)
    eval_ = list(ref)
    scale = fit_scale(eval_, ref, 0.1)
    assert abs(scale - 1.0) < 1e-6
    g = gamma_3d(
        ref,
        [scale * v for v in eval_],
        (nx, ny, nz),
        (2.0, 2.0, 2.0),
        dose_percent=3.0,
        distance_mm=3.0,
        thr_percent=10.0,
        max_points=5000,
        seed=0,
    )
    assert g["pass_percent"] >= 99.0, g


def main() -> None:
    test_flip_x_required_for_depth_face()
    test_block_avg_constant_field()
    test_sparse_block_average_includes_zero_voxels()
    test_beam_y_axis_mapping()
    test_gamma_identity_on_self()
    print("test_match_gpu_geometry OK")


if __name__ == "__main__":
    main()
