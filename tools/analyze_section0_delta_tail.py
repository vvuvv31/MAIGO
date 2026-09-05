#!/usr/bin/env python3
"""Diagnose section-0 delta-tail redistribution against RT06423 TOPAS.

This is deliberately an offline diagnostic.  It never writes a corrected dose
for production use.  The candidate conserves deposited energy by converting Gy
to MeV-proportional mass-weighted values before moving a measured transverse
tail and converts back with the destination voxel mass.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, zoom
from scipy.signal import fftconvolve


GPU_SHAPE = (440, 34, 440)  # native z,y,x
TOPAS_SHAPE = (34, 440, 440)  # mapped y,z,x
TOPAS_SPACING_MM = (2.0, 0.5, 0.5)


def map_gpu(a: np.ndarray) -> np.ndarray:
    return np.flip(np.transpose(a.reshape(GPU_SHAPE), (1, 2, 0)), axis=2)


def read_cctg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = f.read(44)
        magic, version, nx, ny, nz, *_ = struct.unpack("<5I6f", header)
        if magic != 0x47544343 or version not in (1, 2, 3):
            raise ValueError(f"unsupported CCTG header: magic={magic:#x}, version={version}")
        n = nx * ny * nz
        rho = np.fromfile(f, dtype="<f4", count=n)
        section = np.fromfile(f, dtype=np.uint8, count=n)
    if (nx, ny, nz) != GPU_SHAPE:
        raise ValueError(f"unexpected CCTG dimensions {(nx, ny, nz)}")
    return map_gpu(rho), map_gpu(section)


def measured_tail_kernel(csv_path: Path, radius_mm: float) -> tuple[np.ndarray, float]:
    # TOPAS CSV columns are x,y,z,value.  Average interior depth slices to
    # suppress Monte Carlo noise without using a 1-D scorer.
    data = np.loadtxt(csv_path, delimiter=",", comments="#")
    xyz = data[:, :3].astype(np.int32)
    dose = np.zeros((440, 100, 100), dtype=np.float64)
    dose[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = data[:, 3]
    lateral = dose[200:400].sum(axis=0)
    yy, xx = np.indices(lateral.shape)
    cy = float((lateral * yy).sum() / lateral.sum())
    cx = float((lateral * xx).sum() / lateral.sum())
    radius = 2.0 * np.hypot(xx - cx, yy - cy)
    tail = np.where(radius > radius_mm, lateral, 0.0)
    moved_fraction = float(tail.sum() / lateral.sum())
    if not 0.0 < moved_fraction < 1.0:
        raise ValueError("invalid measured tail fraction")
    tail /= tail.sum()
    # Empirical grid is 2x2 mm.  Patient lateral grid is 2x0.5 mm.
    tail = zoom(tail, (1.0, 4.0), order=1, mode="constant", grid_mode=False)
    tail /= tail.sum()
    return tail, moved_fraction


def gamma_1mm(eval_dose: np.ndarray, ref: np.ndarray, sample: np.ndarray,
              dd_percent: float, local: bool) -> float:
    peak = float(ref.max())
    rz, ry, rx = sample.T
    rr = ref[rz, ry, rx]
    tol = dd_percent / 100.0 * (rr if local else peak)
    passed = np.zeros(len(sample), dtype=bool)
    # Same 0.5 mm continuous-search lattice as the frozen evaluation.
    offsets = []
    for dz in (-0.5, 0.0, 0.5):
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
                dist = float(np.sqrt(dz * dz + dy * dy + dx * dx))
                if dist <= 1.0 + 1.0e-9:
                    offsets.append((dist, dz / 2.0, dy / 0.5, dx / 0.5))
    for dist, dz, dy, dx in sorted(offsets):
        vals = map_coordinates(eval_dose, np.vstack((rz + dz, ry + dy, rx + dx)),
                               order=1, mode="nearest")
        passed |= (dist * dist + ((vals - rr) / tol) ** 2) <= 1.0
    return float(100.0 * passed.mean())


def direct_stats(dose: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                 section: np.ndarray, boundary_mm: np.ndarray) -> dict:
    peak = float(ref.max())
    delta = dose - ref
    fail = mask & (np.abs(delta) > 0.03 * peak)
    hot = fail & (delta > 0.0)
    cold = fail & (delta < 0.0)
    out = {
        "pass_percent": float(100.0 * (1.0 - fail.sum() / mask.sum())),
        "fail_count": int(fail.sum()),
        "hot_count": int(hot.sum()),
        "cold_count": int(cold.sum()),
        "section0_fail_count": int((fail & (section == 0)).sum()),
    }
    bins = [0.0, 0.51, 1.01, 2.01, 4.01, 8.01, 1.0e9]
    out["failure_distance_to_section0_boundary_mm"] = {
        f"{bins[i]:g}-{bins[i + 1]:g}": {
            "all": int((fail & (boundary_mm >= bins[i]) & (boundary_mm < bins[i + 1])).sum()),
            "hot": int((hot & (boundary_mm >= bins[i]) & (boundary_mm < bins[i + 1])).sum()),
            "cold": int((cold & (boundary_mm >= bins[i]) & (boundary_mm < bins[i + 1])).sum()),
        }
        for i in range(len(bins) - 1)
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--topas", type=Path, required=True)
    p.add_argument("--ct", type=Path, required=True)
    p.add_argument("--primary-origin", type=Path, required=True)
    p.add_argument("--origin-total", type=Path, required=True)
    p.add_argument("--kernel-csv", type=Path, required=True)
    p.add_argument("--tail-radius-mm", type=float, default=2.01)
    p.add_argument("--moved-fraction", type=float)
    p.add_argument("--source-boundary-clearance-mm", type=float, default=0.0)
    p.add_argument("--nonsection0-destination", choices=("deposit", "local"), default="deposit")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    ref = np.fromfile(args.topas, dtype="<f4").reshape(TOPAS_SHAPE).astype(np.float64)
    gpu = map_gpu(np.fromfile(args.gpu, dtype="<f4")).astype(np.float64)
    origin_primary = map_gpu(np.fromfile(args.primary_origin, dtype="<f4")).astype(np.float64)
    origin_total = map_gpu(np.fromfile(args.origin_total, dtype="<f4")).astype(np.float64)
    rho, section = read_cctg(args.ct)
    rho = rho.astype(np.float64)
    primary_fraction = np.divide(origin_primary, origin_total, out=np.zeros_like(gpu),
                                 where=origin_total > 0.0)
    primary = gpu * np.clip(primary_fraction, 0.0, 1.0)

    kernel, measured_moved_fraction = measured_tail_kernel(args.kernel_csv, args.tail_radius_mm)
    moved_fraction = (args.moved_fraction if args.moved_fraction is not None
                      else measured_moved_fraction)
    sec0 = section == 0
    source_distance = distance_transform_edt(sec0, sampling=TOPAS_SPACING_MM)
    source_mask = sec0 & (source_distance > args.source_boundary_clearance_mm)
    # Voxel volumes are common, so rho is sufficient for relative energy.
    removed_dose = primary * source_mask * moved_fraction
    removed_energy = removed_dose * rho
    deposited_energy = np.empty_like(removed_energy)
    for x in range(removed_energy.shape[2]):
        deposited_energy[:, :, x] = fftconvolve(removed_energy[:, :, x], kernel,
                                                mode="same")
    if args.nonsection0_destination == "local":
        destination_mask = sec0.astype(np.float64)
        accepted_at_source = np.empty_like(removed_energy)
        for x in range(removed_energy.shape[2]):
            deposited_energy[:, :, x] *= destination_mask[:, :, x]
            accepted_at_source[:, :, x] = fftconvolve(
                destination_mask[:, :, x], kernel[::-1, ::-1], mode="same")
        deposited_energy += removed_energy * np.clip(1.0 - accepted_at_source, 0.0, 1.0)
    candidate = gpu - removed_dose + np.divide(
        deposited_energy, rho, out=np.zeros_like(deposited_energy), where=rho > 0.0)

    mask = ref >= 0.10 * ref.max()
    rng = np.random.default_rng(42)
    eligible = np.argwhere(mask)
    sample = eligible[rng.choice(len(eligible), size=50000, replace=False)]
    # Physical distance to an interface, evaluated on both sides.
    distance = np.where(
        sec0,
        distance_transform_edt(sec0, sampling=TOPAS_SPACING_MM),
        distance_transform_edt(~sec0, sampling=TOPAS_SPACING_MM),
    )

    report = {
        "tail_radius_mm": args.tail_radius_mm,
        "source_boundary_clearance_mm": args.source_boundary_clearance_mm,
        "nonsection0_destination": args.nonsection0_destination,
        "measured_moved_fraction": measured_moved_fraction,
        "applied_moved_fraction": moved_fraction,
        "energy_closure_relative": float(
            abs(removed_energy.sum() - deposited_energy.sum()) / removed_energy.sum()),
        "baseline": direct_stats(gpu, ref, mask, section, distance),
        "candidate": direct_stats(candidate, ref, mask, section, distance),
    }
    for label, dose in (("baseline", gpu), ("candidate", candidate)):
        report[label]["global_1pct_1mm"] = gamma_1mm(dose, ref, sample, 1.0, False)
        report[label]["local_1pct_1mm"] = gamma_1mm(dose, ref, sample, 1.0, True)
        report[label]["mean_high_dose_ratio"] = float((dose[mask] / ref[mask]).mean())
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
