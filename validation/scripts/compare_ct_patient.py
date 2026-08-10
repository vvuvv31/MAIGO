#!/usr/bin/env python3
"""Compare GPU patient CT IDD vs TOPAS TsDicomPatient (slice or stitched exit)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_gpu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    z = np.array([float(r["depth_mm"]) for r in rows], dtype=np.float64)
    d = np.array(
        [float(r["energy_deposition_MeV_per_primary"]) for r in rows], dtype=np.float64
    )
    return z, d


def load_topas_csv(path: Path, n_hist: int) -> np.ndarray:
    """Load TOPAS EnergyDeposit CSV.

    Supports either a pre-collapsed 1×1×Nz file (parts[3] = sum) or a full
    Nx×Ny×Nz patient grid (ix,iy,iz,sum,...) which is summed over X,Y per Z.
    """
    rows: list[tuple[int, int, int, float]] = []
    simple: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                ix, iy, iz = int(float(parts[0])), int(float(parts[1])), int(
                    float(parts[2])
                )
                val = float(parts[3])
            except ValueError:
                continue
            rows.append((ix, iy, iz, val))
            simple.append(val)
    if not rows:
        return np.zeros(0, dtype=np.float64)
    max_ix = max(r[0] for r in rows)
    max_iy = max(r[1] for r in rows)
    max_iz = max(r[2] for r in rows)
    # Full 3D grid → sum laterally.
    if max_ix > 0 or max_iy > 0:
        z_sum = np.zeros(max_iz + 1, dtype=np.float64)
        for ix, iy, iz, val in rows:
            z_sum[iz] += val
        return z_sum / float(n_hist)
    # Already collapsed in X/Y (or single column).
    by_z: dict[int, float] = {}
    for ix, iy, iz, val in rows:
        by_z[iz] = by_z.get(iz, 0.0) + val
    n_z = max(by_z) + 1
    out = np.zeros(n_z, dtype=np.float64)
    for iz, val in by_z.items():
        out[iz] = val
    return out / float(n_hist)


def r80(z: np.ndarray, d: np.ndarray) -> float:
    ip = int(np.argmax(d))
    peak = d[ip]
    target = 0.8 * peak
    for i in range(ip, len(d)):
        if d[i] < target:
            if i == 0:
                return float(z[i])
            frac = (target - d[i - 1]) / (d[i] - d[i - 1] + 1e-30)
            return float(z[i - 1] + frac * (z[i] - z[i - 1]))
    return float(z[-1])


def rebin_sum(z: np.ndarray, d: np.ndarray, bin_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-bin fine IDD into coarser bins by summing energy in each bin."""
    z_max = float(z[-1] + (z[1] - z[0]) * 0.5)
    edges = np.arange(0.0, z_max + bin_mm, bin_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.zeros(len(centers), dtype=np.float64)
    half = 0.5 * (z[1] - z[0]) if len(z) > 1 else 0.25
    for zi, di in zip(z, d):
        # assign by center
        j = int(np.floor(zi / bin_mm))
        if 0 <= j < len(out):
            out[j] += di
    return centers, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("out/ct/e150_patient_multimat_idd.csv"),
    )
    parser.add_argument(
        "--topas",
        type=Path,
        default=Path("validation/topas/output/ct_patient_development_energy_deposit.csv"),
    )
    parser.add_argument("--histories", type=int, default=20000)
    parser.add_argument(
        "--bin-mm",
        type=float,
        default=0.5,
        help="TOPAS IDD tube bin width (mm)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("validation/results/ct/compare_patient.metrics.json"),
    )
    parser.add_argument(
        "--output-topas-idd",
        type=Path,
        default=Path("validation/results/ct/topas_e150_patient_idd.csv"),
    )
    args = parser.parse_args()

    zg, dg = load_gpu(args.gpu)
    dt = load_topas_csv(args.topas, args.histories)
    zt = (np.arange(len(dt)) + 0.5) * args.bin_mm
    n = min(len(zg), len(zt))
    zg, dg = zg[:n], dg[:n]
    zt, dt = zt[:n], dt[:n]

    metrics = {
        "gpu_file": str(args.gpu),
        "topas_file": str(args.topas),
        "n_histories": args.histories,
        "gpu_R80_mm": r80(zg, dg),
        "gpu_peak_z_mm": float(zg[int(np.argmax(dg))]),
        "gpu_entrance_0p5mm": float(dg[0]),
        "gpu_integral_MeV": float(dg.sum()),
        "topas_R80_mm": r80(zt, dt),
        "topas_peak_z_mm": float(zt[int(np.argmax(dt))]),
        "topas_entrance": float(dt[0]),
        "topas_integral_MeV": float(dt.sum()),
        "delta_R80_mm": r80(zg, dg) - r80(zt, dt),
        "integral_rel_diff": float((dg.sum() - dt.sum()) / (dt.sum() + 1e-30)),
        "nrmse": float(np.sqrt(np.mean((dg - dt) ** 2)) / (np.max(dt) + 1e-30)),
        "notes": (
            "Parallel-world IDD tube vs GPU patient CT. "
            "HU conversion differs (GPU 4-class density map vs TOPAS Schneider)."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with args.output_topas_idd.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_mm", "energy_deposition_MeV_per_primary"])
        for z, d in zip(zt, dt):
            writer.writerow([z, d])

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
