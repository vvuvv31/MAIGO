#!/usr/bin/env python3
"""Compile a Schneider section-0 C12 longitudinal (forward) delta kernel.

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
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lateral_integral(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", comments="#")
    xyz = data[:, :3].astype(np.int32)
    dose = np.zeros((440, 100, 100), dtype=np.float64)
    dose[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = data[:, 3]
    return dose.sum(axis=(1, 2))


def fit_forward(depth_mm: np.ndarray, lateral: np.ndarray,
               lo_mm: float = 2.0, hi_mm: float = 120.0):
    ref = lateral[(depth_mm >= 50.0) & (depth_mm < 100.0)].mean()
    sel = (depth_mm >= lo_mm) & (depth_mm < hi_mm)
    deficit = 1.0 - lateral[sel] / ref
    use = deficit > 0.002
    z = depth_mm[sel][use]
    mat = np.vstack([np.ones_like(z), -z]).T
    (ln_a, inv_lam), *_ = np.linalg.lstsq(mat, np.log(deficit[use]), rcond=None)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("energy_MeV_per_u,forward_fraction,lambda_mm\n")
        for energy, frac, lam in rows:
            out.write(f"{energy:.8g},{frac:.9g},{lam:.9g}\n")
    metadata = {
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
