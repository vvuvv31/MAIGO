#!/usr/bin/env python3
"""Compare pre-optimization vs batch/persistent 917k CT dose baselines (P0).

Stdlib-only (no numpy). Compares IDD and sparse/dense voxel dose.
Reports R80, peak, integral dose, max voxel, peak position, and 2%/2 mm gamma.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
from array import array
from pathlib import Path


def load_idd_csv(path: Path) -> tuple[list[float], list[float]]:
    depths: list[float] = []
    doses: list[float] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(
            line for line in stream if line.strip() and not line.startswith("#")
        )
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        fields = [name.strip() for name in reader.fieldnames]
        dose_key = None
        for key in (
            "dose_Gy",
            "dose_Gy_per_primary",
            "energy_deposition_MeV",
            "energy_deposition_MeV_per_primary",
            "dose",
        ):
            if key in fields:
                dose_key = key
                break
        if "depth_mm" not in fields or dose_key is None:
            raise ValueError(f"{path}: need depth_mm and dose column; found {fields}")
        for row in reader:
            depths.append(float(row["depth_mm"]))
            doses.append(float(row[dose_key]))
    if len(depths) < 3:
        raise ValueError(f"{path}: too few IDD bins")
    return depths, doses


def crossing(depth: list[float], normalized: list[float], level: float) -> float:
    peak = max(range(len(normalized)), key=lambda i: normalized[i])
    for index in range(peak + 1, len(depth)):
        if normalized[index] <= level < normalized[index - 1]:
            x0, x1 = depth[index - 1], depth[index]
            y0, y1 = normalized[index - 1], normalized[index]
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def trapz(y: list[float], x: list[float]) -> float:
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def idd_metrics(depth: list[float], dose: list[float]) -> dict[str, float]:
    peak_idx = max(range(len(dose)), key=lambda i: dose[i])
    peak = float(dose[peak_idx])
    if peak <= 0.0:
        raise ValueError("non-positive IDD peak")
    normalized = [v / peak for v in dose]
    return {
        "peak_depth_mm": float(depth[peak_idx]),
        "peak_value": peak,
        "integral": trapz(dose, depth),
        "R80_mm": crossing(depth, normalized, 0.8),
        "R50_mm": crossing(depth, normalized, 0.5),
        "R20_mm": crossing(depth, normalized, 0.2),
        "n_bins": float(len(depth)),
    }


def gamma_1d(
    ref_depth: list[float],
    ref_dose: list[float],
    eval_depth: list[float],
    eval_dose: list[float],
    dose_percent: float,
    distance_mm: float,
    threshold_percent: float,
) -> float:
    ref_max = max(ref_dose)
    threshold = threshold_percent / 100.0 * ref_max
    dose_crit = dose_percent / 100.0 * ref_max
    selected = [i for i, v in enumerate(ref_dose) if v >= threshold]
    if not selected:
        return float("nan")
    passed = 0
    for index in selected:
        best = float("inf")
        for j, (ed, ev) in enumerate(zip(eval_depth, eval_dose)):
            g2 = ((ed - ref_depth[index]) / distance_mm) ** 2 + (
                (ev - ref_dose[index]) / dose_crit
            ) ** 2
            if g2 < best:
                best = g2
        if best <= 1.0:
            passed += 1
    return 100.0 * passed / len(selected)


def load_sparse_to_dense(
    path: Path, shape: tuple[int, int, int]
) -> tuple[array, int]:
    nx, ny, nz = shape
    n = nx * ny * nz
    grid = array("d", [0.0]) * n
    nonzero = 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(
            line for line in stream if line.strip() and not line.startswith("#")
        )
        header = next(reader, None)
        if header is None:
            return grid, 0
        # skip header row if present
        first = [c.strip() for c in header]
        rows_iter = reader
        if first and first[0].lower() != "ix":
            # first row is data
            rows_iter = __import__("itertools").chain([header], reader)

        for fields in rows_iter:
            fields = [f.strip() for f in fields]
            if not fields:
                continue
            if fields[0].lower() == "ix":
                continue
            ix, iy, iz = int(fields[0]), int(fields[1]), int(fields[2])
            value = float(fields[6])
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                raise ValueError(f"{path}: index {(ix, iy, iz)} outside {shape}")
            linear = ix + nx * (iy + ny * iz)
            if grid[linear] != 0.0:
                raise ValueError(f"{path}: duplicate {(ix, iy, iz)}")
            grid[linear] = value
            nonzero += 1
    return grid, nonzero


def read_mhd_raw(mhd_path: Path) -> tuple[array, tuple[int, int, int], dict[str, str]]:
    meta: dict[str, str] = {}
    for line in mhd_path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    dim = tuple(int(x) for x in meta["DimSize"].split())  # nx, ny, nz
    nx, ny, nz = dim
    raw_path = mhd_path.parent / meta["ElementDataFile"]
    raw = raw_path.read_bytes()
    expected = nx * ny * nz * 4
    if len(raw) != expected:
        raise ValueError(f"{raw_path}: expected {expected} bytes, got {len(raw)}")
    # RAW layout: z major, then y, then x (from sparse_dose_to_mhd)
    floats = array("f")
    floats.frombytes(raw)
    if sys_byteorder_is_big():
        floats.byteswap()
    dense = array("d", [0.0]) * (nx * ny * nz)
    # convert float32 z-y-x order into same linear ix + nx*(iy + ny*iz)
    for linear_zyx, value in enumerate(floats):
        iz = linear_zyx // (nx * ny)
        rem = linear_zyx % (nx * ny)
        iy = rem // nx
        ix = rem % nx
        dense[ix + nx * (iy + ny * iz)] = float(value)
    return dense, dim, meta


def sys_byteorder_is_big() -> bool:
    return __import__("sys").byteorder == "big"


def unravel(linear: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    nx, ny, nz = shape
    iz = linear // (nx * ny)
    rem = linear % (nx * ny)
    iy = rem // nx
    ix = rem % nx
    return ix, iy, iz


def voxel_metrics(
    ref: array,
    eval_: array,
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> dict:
    nx, ny, nz = shape
    n = nx * ny * nz
    if len(ref) != n or len(eval_) != n:
        raise ValueError("grid size mismatch")

    ref_max = 0.0
    eval_max = 0.0
    ref_peak = 0
    eval_peak = 0
    ref_sum = 0.0
    eval_sum = 0.0
    nonzero_ref = 0
    nonzero_eval = 0
    max_abs_diff = 0.0
    sum_sq_diff = 0.0
    sum_abs_rel_10 = 0.0
    count_10 = 0
    sum_sq_rel_10 = 0.0
    sum_sq_rel_50 = 0.0
    count_50 = 0
    selected: list[int] = []

    for i in range(n):
        r = ref[i]
        e = eval_[i]
        if r > ref_max:
            ref_max = r
            ref_peak = i
        if e > eval_max:
            eval_max = e
            eval_peak = i
        ref_sum += r
        eval_sum += e
        if r > 0.0:
            nonzero_ref += 1
        if e > 0.0:
            nonzero_eval += 1
        d = e - r
        ad = abs(d)
        if ad > max_abs_diff:
            max_abs_diff = ad
        sum_sq_diff += d * d

    thr10 = 0.10 * ref_max
    thr50 = 0.50 * ref_max
    thr_gamma = 0.10 * ref_max
    for i in range(n):
        r = ref[i]
        e = eval_[i]
        d = e - r
        if r >= thr10 and r > 0.0:
            rel = abs(d) / r
            sum_abs_rel_10 += rel
            sum_sq_rel_10 += (d / ref_max) ** 2
            count_10 += 1
        if r >= thr50:
            sum_sq_rel_50 += (d / ref_max) ** 2
            count_50 += 1
        if r >= thr_gamma:
            selected.append(i)

    # 3D gamma 2%/2mm, subsample if huge
    dose_percent = 2.0
    distance_mm = 2.0
    dose_crit = dose_percent / 100.0 * ref_max if ref_max > 0 else 1.0
    sx, sy, sz = spacing_mm
    rx = max(1, int(math.ceil(distance_mm / sx)))
    ry = max(1, int(math.ceil(distance_mm / sy)))
    rz = max(1, int(math.ceil(distance_mm / sz)))
    max_points = 50_000
    total_sel = len(selected)
    if total_sel > max_points:
        rng = random.Random(0)
        selected = rng.sample(selected, max_points)
    passed = 0
    for linear in selected:
        ix, iy, iz = unravel(linear, shape)
        ref_val = ref[linear]
        best = float("inf")
        x0 = max(0, ix - rx)
        x1 = min(nx, ix + rx + 1)
        y0 = max(0, iy - ry)
        y1 = min(ny, iy + ry + 1)
        z0 = max(0, iz - rz)
        z1 = min(nz, iz + rz + 1)
        for jx in range(x0, x1):
            dx = (jx - ix) * sx
            for jy in range(y0, y1):
                dy = (jy - iy) * sy
                for jz in range(z0, z1):
                    dz = (jz - iz) * sz
                    e = eval_[jx + nx * (jy + ny * jz)]
                    g2 = (dx / distance_mm) ** 2 + (dy / distance_mm) ** 2 + (
                        dz / distance_mm
                    ) ** 2 + ((e - ref_val) / dose_crit) ** 2
                    if g2 < best:
                        best = g2
        if best <= 1.0:
            passed += 1
    gamma_pass = 100.0 * passed / len(selected) if selected else float("nan")

    def safe_div(a: float, b: float) -> float:
        return a / b if b != 0.0 else float("nan")

    return {
        "shape": list(shape),
        "ref_max": ref_max,
        "eval_max": eval_max,
        "max_rel_diff": safe_div(eval_max - ref_max, ref_max),
        "ref_peak_index": list(unravel(ref_peak, shape)),
        "eval_peak_index": list(unravel(eval_peak, shape)),
        "peak_index_match": unravel(ref_peak, shape) == unravel(eval_peak, shape),
        "integral_ref": ref_sum,
        "integral_eval": eval_sum,
        "integral_rel_diff": safe_div(eval_sum - ref_sum, ref_sum),
        "nonzero_ref": nonzero_ref,
        "nonzero_eval": nonzero_eval,
        "max_abs_diff": max_abs_diff,
        "rmse_over_global_max": math.sqrt(sum_sq_diff / n) / ref_max if ref_max else float("nan"),
        "rel_rmse_mask10": math.sqrt(sum_sq_rel_10 / count_10) if count_10 else float("nan"),
        "rel_rmse_mask50": math.sqrt(sum_sq_rel_50 / count_50) if count_50 else float("nan"),
        "mean_abs_rel_diff_mask10": sum_abs_rel_10 / count_10 if count_10 else float("nan"),
        "gamma_2pct_2mm_thr10_pass_percent": gamma_pass,
        "gamma_points_evaluated": len(selected),
        "gamma_points_available": total_sel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-idd", type=Path, required=True)
    parser.add_argument("--eval-idd", type=Path, required=True)
    parser.add_argument("--ref-voxels", type=Path)
    parser.add_argument("--eval-voxels", type=Path)
    parser.add_argument("--ref-mhd", type=Path)
    parser.add_argument("--eval-mhd", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=(505, 35, 417))
    parser.add_argument("--spacing-mm", nargs=3, type=float, default=(0.5, 2.0, 0.5))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-gamma", action="store_true")
    args = parser.parse_args()

    ref_d, ref_dose = load_idd_csv(args.ref_idd)
    ev_d, ev_dose = load_idd_csv(args.eval_idd)
    if len(ref_d) != len(ev_d) or any(abs(a - b) > 1e-9 for a, b in zip(ref_d, ev_d)):
        raise SystemExit("IDD depth grids differ")

    ref_m = idd_metrics(ref_d, ref_dose)
    ev_m = idd_metrics(ev_d, ev_dose)
    peak = ref_m["peak_value"]
    nrmse = math.sqrt(
        sum(((a - b) / peak) ** 2 for a, b in zip(ev_dose, ref_dose)) / len(ref_dose)
    )
    idd = {
        "ref": ref_m,
        "eval": ev_m,
        "peak_value_rel_diff": (ev_m["peak_value"] - ref_m["peak_value"]) / ref_m["peak_value"],
        "peak_depth_diff_mm": ev_m["peak_depth_mm"] - ref_m["peak_depth_mm"],
        "R80_diff_mm": ev_m["R80_mm"] - ref_m["R80_mm"],
        "integral_rel_diff": (ev_m["integral"] - ref_m["integral"]) / ref_m["integral"],
        "nrmse": nrmse,
        "gamma_2pct_2mm_thr10": gamma_1d(
            ref_d, ref_dose, ev_d, ev_dose, 2.0, 2.0, 10.0
        ),
        "gamma_1pct_1mm_thr10": gamma_1d(
            ref_d, ref_dose, ev_d, ev_dose, 1.0, 1.0, 10.0
        ),
    }
    report: dict = {"idd": idd}
    print("IDD comparison:")
    print(json.dumps(idd, indent=2))

    shape = tuple(args.shape)
    spacing = tuple(args.spacing_mm)
    ref_v = None
    ev_v = None
    if args.ref_voxels and args.eval_voxels:
        print(f"Loading sparse voxels {args.ref_voxels} ...", flush=True)
        ref_v, nz_r = load_sparse_to_dense(args.ref_voxels, shape)
        print(f"  nonzero={nz_r}", flush=True)
        print(f"Loading sparse voxels {args.eval_voxels} ...", flush=True)
        ev_v, nz_e = load_sparse_to_dense(args.eval_voxels, shape)
        print(f"  nonzero={nz_e}", flush=True)
    elif args.ref_mhd and args.eval_mhd:
        print(f"Loading MHD {args.ref_mhd} ...", flush=True)
        ref_v, dim_r, _ = read_mhd_raw(args.ref_mhd)
        print(f"Loading MHD {args.eval_mhd} ...", flush=True)
        ev_v, dim_e, _ = read_mhd_raw(args.eval_mhd)
        if dim_r != shape or dim_e != shape:
            raise SystemExit(f"MHD shape {dim_r}/{dim_e} != {shape}")

    if ref_v is not None and ev_v is not None:
        print("Computing voxel metrics ...", flush=True)
        metrics = voxel_metrics(ref_v, ev_v, shape, spacing)
        if args.skip_gamma:
            metrics["gamma_2pct_2mm_thr10_pass_percent"] = None
        report["voxels"] = metrics
        print(json.dumps(metrics, indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
