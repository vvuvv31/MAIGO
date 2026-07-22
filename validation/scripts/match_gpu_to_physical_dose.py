#!/usr/bin/env python3
"""Match GPU TPS-90 dose to ct/physical_dose.mhd (matRad plan cube / TOPAS dij·x).

Production geometry convention (do not free-translate):
  - CT: patient_ct_tps_90_xneg.bin (beam along −patient X, GPU +Z depth)
  - spots_patient_rot_z_deg: −90 (physical_dose COM; run_c lists +90)
  - flip_x=True: GPU z=0 → patient X max (entrance face for xneg)
  - flip_y=False

Steps:
  1. Map GPU MHD (505×35×417, beam +Z) into patient frame with X/Y flips.
  2. Block-average to the physical_dose grid (default 104×126×35 @ 2 mm).
  3. Fit a single global scale:  s * GPU  ≈  reference  (LS on high-dose voxels).
  4. Write scaled patient-frame MHD and a JSON metrics report.

The scale absorbs MC history budget and missing MU→ions calibration.
Note: physical_dose is matRad-optimized (not full TOPAS MC 3D); residual after
geometry lock is local MC vs plan-cube morphology (see SCRATCH ROOT_CAUSE.md).
"""

from __future__ import annotations

import argparse
import array
import json
import math
import random
import sys
from pathlib import Path


def read_mhd(path: Path) -> tuple[dict[str, str], array.array]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    dim = [int(v) for v in meta["DimSize"].split()]
    raw_path = path.parent / meta["ElementDataFile"]
    data = array.array("f")
    data.frombytes(raw_path.read_bytes())
    expected = dim[0] * dim[1] * dim[2]
    if len(data) != expected:
        raise SystemExit(f"{raw_path}: expected {expected} floats, got {len(data)}")
    if sys.byteorder != "little":
        data.byteswap()
    return meta, data


def write_mhd(
    path: Path,
    data: array.array,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    offset: tuple[float, float, float],
    units: str,
    extra_lines: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    out = array.array("f", data)
    if sys.byteorder != "little":
        out.byteswap()
    raw_path.write_bytes(out.tobytes())
    nx, ny, nz = shape
    sx, sy, sz = spacing
    ox, oy, oz = offset
    lines = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        "CompressedData = False",
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        f"Offset = {ox:.12g} {oy:.12g} {oz:.12g}",
        "CenterOfRotation = 0 0 0",
        "AnatomicalOrientation = RAI",
        f"ElementSpacing = {sx:.12g} {sy:.12g} {sz:.12g}",
        f"DimSize = {nx} {ny} {nz}",
        "ElementType = MET_FLOAT",
        f"DoseUnits = {units}",
        f"ElementDataFile = {raw_path.name}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")
    path.write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {path}")
    print(f"Wrote {raw_path} ({raw_path.stat().st_size} bytes)")


def map_gpu_to_physical(
    gpu: array.array,
    gpu_shape: tuple[int, int, int],
    patient_shape: tuple[int, int, int],
    phys_shape: tuple[int, int, int],
    flip_x: bool,
    flip_y: bool,
) -> list[float]:
    gx, gy, gz = gpu_shape
    px_n, py_n, pz_n = patient_shape
    lnx, lny, lnz = phys_shape
    if gz != px_n or gx != py_n or gy != pz_n:
        raise SystemExit(
            f"GPU shape {gpu_shape} incompatible with patient shape {patient_shape}"
        )
    bx = max(1, px_n // lnx)
    by = max(1, py_n // lny)
    bz = max(1, pz_n // lnz)
    acc = [0.0] * (lnx * lny * lnz)
    counts = [0] * (lnx * lny * lnz)
    plane = gx * gy
    for iz in range(gz):
        patient_x = (px_n - 1 - iz) if flip_x else iz
        ox = patient_x // bx
        if ox >= lnx:
            continue
        for iy in range(gy):
            patient_z = iy
            oz = patient_z // bz
            if oz >= lnz:
                continue
            for ix in range(gx):
                patient_y = (py_n - 1 - ix) if flip_y else ix
                oy = patient_y // by
                if oy >= lny:
                    continue
                value = gpu[iz * plane + iy * gx + ix]
                if value == 0.0:
                    continue
                linear = oz * lnx * lny + oy * lnx + ox
                acc[linear] += value
                counts[linear] += 1
    for i, count in enumerate(counts):
        if count > 1:
            acc[i] /= float(count)
    return acc


def fit_scale(gpu: list[float], ref: array.array, thr_frac: float) -> float:
    ref_max = max(ref)
    thr = thr_frac * ref_max
    num = 0.0
    den = 0.0
    for a, b in zip(gpu, ref):
        if b >= thr:
            num += a * b
            den += a * a
    if den <= 0.0:
        raise SystemExit("No overlapping high-dose voxels for scale fit")
    return num / den


def idd_x(values: list[float] | array.array, shape: tuple[int, int, int]) -> list[float]:
    nx, ny, nz = shape
    out = [0.0] * nx
    for iz in range(nz):
        for iy in range(ny):
            base = iz * nx * ny + iy * nx
            for ix in range(nx):
                out[ix] += values[base + ix]
    return out


def gamma_3d(
    ref: array.array,
    eval_: list[float],
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    dose_percent: float,
    distance_mm: float,
    thr_percent: float,
    max_points: int,
    seed: int,
) -> dict[str, float]:
    nx, ny, nz = shape
    sx, sy, sz = spacing_mm
    ref_max = max(ref)
    thr = thr_percent / 100.0 * ref_max
    dose_crit = dose_percent / 100.0 * ref_max
    rx = max(1, int(math.ceil(distance_mm / sx)))
    ry = max(1, int(math.ceil(distance_mm / sy)))
    rz = max(1, int(math.ceil(distance_mm / sz)))
    selected = [i for i, v in enumerate(ref) if v >= thr]
    total_available = len(selected)
    if total_available == 0:
        return {"pass_percent": float("nan"), "points": 0, "available": 0}
    if total_available > max_points:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_points)
    passed = 0
    for linear in selected:
        iz = linear // (nx * ny)
        rem = linear % (nx * ny)
        iy = rem // nx
        ix = rem % nx
        ref_val = ref[linear]
        x0 = max(0, ix - rx)
        x1 = min(nx, ix + rx + 1)
        y0 = max(0, iy - ry)
        y1 = min(ny, iy + ry + 1)
        z0 = max(0, iz - rz)
        z1 = min(nz, iz + rz + 1)
        best = float("inf")
        for jx in range(x0, x1):
            dx = (jx - ix) * sx
            for jy in range(y0, y1):
                dy = (jy - iy) * sy
                for jz in range(z0, z1):
                    dz = (jz - iz) * sz
                    e = eval_[jz * nx * ny + jy * nx + jx]
                    g2 = (
                        (dx / distance_mm) ** 2
                        + (dy / distance_mm) ** 2
                        + (dz / distance_mm) ** 2
                        + ((e - ref_val) / dose_crit) ** 2
                    )
                    if g2 < best:
                        best = g2
        if best <= 1.0:
            passed += 1
    return {
        "pass_percent": 100.0 * passed / len(selected),
        "points": len(selected),
        "available": total_available,
    }


def peak_index(values: list[float] | array.array, shape: tuple[int, int, int]):
    nx, ny, nz = shape
    maximum = -1.0
    best = 0
    for i, value in enumerate(values):
        if value > maximum:
            maximum = value
            best = i
    iz = best // (nx * ny)
    rem = best % (nx * ny)
    iy = rem // nx
    ix = rem % nx
    return (ix, iy, iz), maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_mhd", type=Path)
    parser.add_argument("physical_mhd", type=Path, default=Path("ct/physical_dose.mhd"), nargs="?")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/ct/match_physical"),
    )
    parser.add_argument("--patient-shape", nargs=3, type=int, default=(417, 505, 35))
    parser.add_argument("--histories", type=float, default=917000.0)
    parser.add_argument("--thr-frac", type=float, default=0.10)
    parser.add_argument("--flip-x", action="store_true", default=True,
                        help="Flip patient X when mapping GPU Z (default True)")
    parser.add_argument("--flip-y", action="store_true", default=False,
                        help="Flip patient Y when mapping GPU X (default False; "
                             "needed only for pre-Y-sign-fix dose dumps)")
    parser.add_argument("--no-flip-x", action="store_true")
    parser.add_argument("--no-flip-y", action="store_true",
                        help="Force no Y flip (default already false)")
    parser.add_argument("--gamma-points", type=int, default=50000)
    parser.add_argument("--skip-gamma", action="store_true")
    args = parser.parse_args()
    flip_x = args.flip_x and not args.no_flip_x
    # --flip-y stores True only when the flag is passed; default False.
    flip_y = bool(args.flip_y) and not args.no_flip_y

    phys_meta, phys = read_mhd(args.physical_mhd)
    gpu_meta, gpu = read_mhd(args.gpu_mhd)
    phys_shape = tuple(int(v) for v in phys_meta["DimSize"].split())
    phys_spacing = tuple(float(v) for v in phys_meta["ElementSpacing"].split())
    phys_offset = tuple(float(v) for v in phys_meta.get("Offset", "0 0 0").split())
    gpu_shape = tuple(int(v) for v in gpu_meta["DimSize"].split())

    print(f"Physical: {args.physical_mhd} shape={phys_shape} max={max(phys):.6g}")
    print(f"GPU:      {args.gpu_mhd} shape={gpu_shape} max={max(gpu):.6g}")
    print(f"Mapping flips: x={flip_x} y={flip_y}")

    mapped = map_gpu_to_physical(
        gpu,
        gpu_shape,
        tuple(args.patient_shape),
        phys_shape,
        flip_x,
        flip_y,
    )
    scale = fit_scale(mapped, phys, args.thr_frac)
    scaled = array.array("f", (scale * v for v in mapped))

    phys_max = max(phys)
    thr = args.thr_frac * phys_max
    ss = sp = 0.0
    n = 0
    for a, b in zip(scaled, phys):
        if b >= thr:
            e = a - b
            ss += e * e
            sp += b * b
            n += 1
    rmse = math.sqrt(ss / n) if n else float("nan")
    nrmse = rmse / phys_max if phys_max else float("nan")
    sab = saa = sbb = 0.0
    for a, b in zip(scaled, phys):
        sab += a * b
        saa += a * a
        sbb += b * b
    cosine = sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else float("nan")

    idd_p = idd_x(phys, phys_shape)
    idd_g = idd_x(scaled, phys_shape)
    idd_peak_p = idd_p.index(max(idd_p))
    idd_peak_g = idd_g.index(max(idd_g))
    idd_corr = sum(a * b for a, b in zip(idd_p, idd_g)) / math.sqrt(
        sum(a * a for a in idd_p) * sum(b * b for b in idd_g)
    )

    peak_p, pmax = peak_index(phys, phys_shape)
    peak_g, gmax = peak_index(scaled, phys_shape)

    report = {
        "gpu_mhd": str(args.gpu_mhd),
        "physical_mhd": str(args.physical_mhd),
        "histories": args.histories,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "scale_gpu_to_physical": scale,
        "scale_per_history": scale / args.histories if args.histories else None,
        "threshold_fraction_of_peak": args.thr_frac,
        "high_dose_voxels": n,
        "rmse_high_dose": rmse,
        "nrmse_high_dose": nrmse,
        "cosine_3d": cosine,
        "physical_max": pmax,
        "scaled_gpu_max": gmax,
        "physical_peak_index_xyz": list(peak_p),
        "scaled_gpu_peak_index_xyz": list(peak_g),
        "idd_peak_bin_physical": idd_peak_p,
        "idd_peak_bin_gpu": idd_peak_g,
        "idd_correlation": idd_corr,
        "integral_physical": float(sum(phys)),
        "integral_scaled_gpu": float(sum(scaled)),
        "integral_rel_diff": (sum(scaled) - sum(phys)) / sum(phys),
    }

    if not args.skip_gamma:
        print("Computing 3D gamma 2%/2mm (subsample)...")
        g2 = gamma_3d(
            phys,
            list(scaled),
            phys_shape,
            phys_spacing,
            dose_percent=2.0,
            distance_mm=2.0,
            thr_percent=10.0,
            max_points=args.gamma_points,
            seed=0,
        )
        print("Computing 3D gamma 3%/3mm (subsample)...")
        g3 = gamma_3d(
            phys,
            list(scaled),
            phys_shape,
            phys_spacing,
            dose_percent=3.0,
            distance_mm=3.0,
            thr_percent=10.0,
            max_points=args.gamma_points,
            seed=0,
        )
        report["gamma_2pct_2mm_thr10"] = g2
        report["gamma_3pct_3mm_thr10"] = g3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_mhd = args.output_dir / "gpu_scaled_to_physical.mhd"
    write_mhd(
        out_mhd,
        scaled,
        phys_shape,
        phys_spacing,
        phys_offset,
        units="Gy (scaled to physical_dose)",
        extra_lines=[
            f"Comment = GPU dose scaled by {scale:.8g} to match {args.physical_mhd.name}",
            f"ScaleFactor = {scale:.12g}",
            f"GPUHistories = {args.histories:g}",
        ],
    )
    # Also write unscaled mapped GPU for debugging.
    write_mhd(
        args.output_dir / "gpu_mapped_unscaled.mhd",
        array.array("f", mapped),
        phys_shape,
        phys_spacing,
        phys_offset,
        units="Gy (GPU total, unscaled)",
    )

    # IDD CSV
    idd_path = args.output_dir / "idd_compare.csv"
    with idd_path.open("w", encoding="utf-8") as stream:
        stream.write("ix,x_mm,physical,gpu_scaled,gpu_unscaled\n")
        for ix in range(phys_shape[0]):
            x_mm = phys_offset[0] + ix * phys_spacing[0]
            stream.write(
                f"{ix},{x_mm:.6g},{idd_p[ix]:.8g},{idd_g[ix]:.8g},{idd_x(mapped, phys_shape)[ix]:.8g}\n"
            )
    print(f"Wrote {idd_path}")

    report_path = args.output_dir / "match_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
