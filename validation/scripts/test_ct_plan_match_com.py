#!/usr/bin/env python3
"""Regression: production GPU CT-plan dose COM-aligns to physical_dose under fixed flips.

Requires on-disk products:
  - out/ct/h9p17M_xneg/dose.mhd (or env CT_GPU_MHD)
  - ct/physical_dose.mhd

Asserts the locked geometric convention yields |COM| residual < 1 phys voxel
and cosine > 0.95 (shape). Does **not** require gamma ≥95% (matRad cube residual).
"""

from __future__ import annotations

import array
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_gpu_to_physical_dose import (  # noqa: E402
    fit_scale,
    map_gpu_to_physical,
    read_mhd,
)


def _com(vals, ref, shape, thr):
    nx, ny, nz = shape
    sx = sy = sz = w = 0.0
    for i, b in enumerate(ref):
        if b < thr:
            continue
        v = vals[i]
        iz = i // (nx * ny)
        rem = i % (nx * ny)
        iy = rem // nx
        ix = rem % nx
        sx += ix * v
        sy += iy * v
        sz += iz * v
        w += v
    if w <= 0:
        return (float("nan"),) * 3
    return (sx / w, sy / w, sz / w)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    gpu_path = Path(os.environ.get("CT_GPU_MHD", root / "out/ct/h9p17M_xneg/dose.mhd"))
    phys_path = root / "ct/physical_dose.mhd"
    if not gpu_path.is_file() or not phys_path.is_file():
        print(f"SKIP missing dose files: {gpu_path} or {phys_path}")
        return

    gmeta, gpu = read_mhd(gpu_path)
    pmeta, pref = read_mhd(phys_path)
    gdim = tuple(int(x) for x in gmeta["DimSize"].split())
    pdim = tuple(int(x) for x in pmeta["DimSize"].split())
    patient = (417, 505, 35)
    mapped = map_gpu_to_physical(gpu, gdim, patient, pdim, flip_x=True, flip_y=False)
    thr = 0.1 * max(pref)
    scale = fit_scale(mapped, pref, 0.1)
    scaled = [scale * v for v in mapped]

    sab = saa = sbb = 0.0
    for a, b in zip(scaled, pref):
        sab += a * b
        saa += a * a
        sbb += b * b
    cos = sab / math.sqrt(saa * sbb)

    pcom = _com(pref, pref, pdim, thr)
    gcom = _com(scaled, pref, pdim, thr)
    dcom = tuple(pcom[i] - gcom[i] for i in range(3))

    assert cos > 0.95, f"cosine too low: {cos}"
    assert all(abs(d) < 1.0 for d in dcom), f"COM residual too large: {dcom}"
    print(
        "test_ct_plan_match_com OK:",
        f"cos={cos:.4f}",
        f"dCOM_vox=({dcom[0]:.3f},{dcom[1]:.3f},{dcom[2]:.3f})",
        f"scale={scale:.4g}",
    )


if __name__ == "__main__":
    main()
