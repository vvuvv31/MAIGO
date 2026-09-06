#!/usr/bin/env python3
"""Compile an UNVALIDATED Schneider section-0 slab-response fit.

This reproduces a historical fit, not a microscopic electron kernel. New
outputs require independent validation and cannot replace runtime candidate pins.

Method: for each TOPAS HU-1000 slab run, the lateral dose integral per 0.5 mm
depth slice is the broad-beam depth-dose shape (lateral transport cancels).
The GPU condenses all delta energy locally, so its lateral integral is flat
when normalized to the deep (50-100 mm) mean. The TOPAS deficit
``1 - lateral(z)/deep_mean`` is fit to ``A*exp(-z/lambda)`` over z in
[2, 120] mm, i.e. a forward exponential transport of fraction A of the local
deposit with mean range lambda. The z < 2 mm bins are excluded: the 200 MeV/u
run shows a run-specific entrance spike there that the independent 150/225
energy-scan runs do not reproduce.

The transverse tail table (v1) is untouched; this file is supplementary.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lateral_integral(path: Path) -> np.ndarray:
    # This compiler only accepts the frozen extraction geometry. Do not
    # silently reinterpret arbitrary scorer headers as the same 0.5 mm grid.
    with path.open() as source:
        header = "".join(next(source) for _ in range(8))
    for axis, count, pitch in [("X", 100, 0.2), ("Y", 100, 0.2), ("Z", 440, 0.05)]:
        match = re.search(rf"# {axis} in (\d+) bins of ([\d.]+) cm", header)
        if not match or int(match[1]) != count or float(match[2]) != pitch:
            raise ValueError("Unexpected frozen slab geometry")
    if "DoseToMedium ( Gy ) : Sum" not in header:
        raise ValueError("Expected 3D DoseToMedium Sum")
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.shape != (440 * 100 * 100, 4) or not np.isfinite(data).all():
        raise ValueError("Incomplete or nonfinite 3D scorer")
    if np.any(data[:, :3] != np.floor(data[:, :3])) or np.any(data < 0):
        raise ValueError("Invalid voxel indices or dose")
    xyz = data[:, :3].astype(np.int32)
    if np.any(xyz >= np.array([100, 100, 440])):
        raise ValueError("Scorer index out of bounds")
    indices = (xyz[:, 2] * 100 + xyz[:, 1]) * 100 + xyz[:, 0]
    if np.unique(indices).size != indices.size:
        raise ValueError("Duplicate scorer voxels")
    dose = np.zeros((440, 100, 100), dtype=np.float64)
    dose[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = data[:, 3]
    return dose.sum(axis=(1, 2))


def fit_forward(depth_mm: np.ndarray, lateral: np.ndarray,
               lo_mm: float = 2.0, hi_mm: float = 120.0):
    ref = lateral[(depth_mm >= 50.0) & (depth_mm < 100.0)].mean()
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Invalid reference plateau")
    sel = (depth_mm >= lo_mm) & (depth_mm < hi_mm)
    deficit = 1.0 - lateral[sel] / ref
    use = deficit > 0.002
    z = depth_mm[sel][use]
    if len(z) < 3:
        raise ValueError("Insufficient points for forward fit")
    mat = np.vstack([np.ones_like(z), -z]).T
    (ln_a, inv_lam), *_ = np.linalg.lstsq(mat, np.log(deficit[use]), rcond=None)
    if not np.isfinite([ln_a, inv_lam]).all() or inv_lam <= 0:
        raise ValueError("Nonphysical forward fit")
    return float(np.exp(ln_a)), float(1.0 / inv_lam), float(ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="energy_MeV_per_u:path_to_TOPAS_3D_csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    # Kernel mapping: the GPU deposits the sampled range uniformly along the
    # ray, whose effective deficit is shorter and weaker than a point deposit
    # with the same nominal (A, lambda). Measured on the 200 MeV/u slab with
    # the production kernel: A_eff ~= 0.83 * A_nom, lambda_eff ~= 0.76 *
    # lambda_nom. The stored table holds NOMINAL values pre-scaled so the
    # effective kernel reproduces the physical TOPAS fit.
    p.add_argument("--kernel-a-scale", type=float, default=1.0)
    p.add_argument("--kernel-lambda-scale", type=float, default=1.0)
    args = p.parse_args()
    if args.output.exists() or args.metadata.exists():
        p.error("Refusing to overwrite frozen data; use new output paths")
    if args.output.resolve() == args.metadata.resolve():
        p.error("Data and metadata paths must differ")
    if not np.isfinite([args.kernel_a_scale, args.kernel_lambda_scale]).all() or min(
            args.kernel_a_scale, args.kernel_lambda_scale) <= 0:
        p.error("Kernel scales must be finite and positive")
    rows = []
    inputs = []
    for spec in sorted(args.input):
        energy_text, path_text = spec.split(":", 1)
        energy, path = float(energy_text), Path(path_text)
        lateral = lateral_integral(path)
        depth_mm = (np.arange(lateral.size) + 0.5) * 0.5
        frac, lam, ref = fit_forward(depth_mm, lateral)
        rows.append((energy, frac * args.kernel_a_scale,
                     lam * args.kernel_lambda_scale))
        inputs.append({"energy_MeV_per_u": energy, "path": str(path),
                       "sha256": sha256(path),
                       "deep_mean_per_slice": ref,
                       "physical_forward_fraction": frac,
                       "physical_lambda_mm": lam,
                       "fitted_forward_fraction": frac * args.kernel_a_scale,
                       "fitted_lambda_mm": lam * args.kernel_lambda_scale})
    rows.sort(key=lambda item: item[0])
    if len({row[0] for row in rows}) != len(rows) or any(
            not np.isfinite(row).all() or row[0] <= 0 or not 0 < row[1] < 0.5 or row[2] <= 0
            for row in rows):
        raise ValueError("Invalid or duplicate candidate energy rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("energy_MeV_per_u,forward_fraction,lambda_mm\n")
        for energy, frac, lam in rows:
            out.write(f"{energy:.8g},{frac:.9g},{lam:.9g}\n")
    metadata = {
        "status": "unvalidated_diagnostic_fit",
        "compiler_sha256": sha256(Path(__file__)),
        "scope_warning": "No patient-specific calibration, extrapolation or cross-material authorization",
        "schema_version": 1,
        "scope": "primary C12, Schneider section 0 only, longitudinal supplement",
        "source": "TOPAS 4.2.p3 / Geant4 11.3.p02 g4em-standard_opt4 DoseToMedium 3D",
        "production_cut_mm": 0.05,
        "beam": "zero-width C12, 1% energy spread",
        "model": "forward exponential transport of fraction A with mean range lambda; "
                 "fit deficit(z) = A*exp(-z/lambda) to lateral-integral deficit over z in [2,120] mm",
        "kernel_mapping": {
            "a_scale": args.kernel_a_scale,
            "lambda_scale": args.kernel_lambda_scale,
            "reason": "uniform-along-ray deposits deliver A_eff ~= 0.83*A_nom, "
                      "lambda_eff ~= 0.76*lambda_nom (measured 200 MeV/u slab); "
                      "stored values are pre-scaled nominals",
        },
        "excluded": "z < 2 mm (run-specific entrance spike in the 200 MeV/u run, "
                    "absent from independent 150/225 scans)",
        "interpolation": "linear in energy",
        "inputs": inputs,
        "data_file": str(args.output),
        "data_filename": args.output.name,
        "data_sha256": sha256(args.output),
        "data_size_bytes": args.output.stat().st_size,
        "file_size_bytes": args.output.stat().st_size,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
