#!/usr/bin/env python3
"""Compare GPU CT-plan MHD vs TOPAS DoseToMedium CSV (prelim single/layer).

Reports (scale-only; no free translation):
  - IDD along patient X (depth after mapping)
  - Lateral dose: σ_rms(depth), COM_y/z(depth), FWHM, Y & Z profiles at peak
  - Global gamma 3%/3 mm, threshold ≥ 10% peak — primary acceptance metric.
  - Local gamma 3%/3 mm and 3%/5 mm — stricter diagnostic of residual
    physics/statistical differences.

Geometry: GPU (505×35×417) → patient (104×126×35) via map_gpu_to_physical.

Prelim vs TOPAS DoseToMedium on Patient (RotZ=+90, native DICOM bins):
  - flip_x=True  — GPU +Z is −patient-X for the xneg CT packing
  - flip_y=False — GPU x is patient Y with the same index sense
"""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_gpu_to_physical_dose import (  # noqa: E402
    fit_scale,
    gamma_3d as gamma_interpolated,
    map_gpu_to_physical,
    read_mhd,
    write_mhd,
)


def load_topas_csv(path: Path, nx: int = 104, ny: int = 126, nz: int = 35) -> array.array:
    data = array.array("f", [0.0] * (nx * ny * nz))
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 4:
                continue
            try:
                ix, iy, iz = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                val = float(parts[3])
            except ValueError:
                continue
            if val == 0.0 or not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                continue
            data[iz * nx * ny + iy * nx + ix] = val
    return data


def idd_x(vals: list[float] | array.array, shape: tuple[int, int, int]) -> list[float]:
    nx, ny, nz = shape
    out = [0.0] * nx
    for iz in range(nz):
        for iy in range(ny):
            base = iz * nx * ny + iy * nx
            for ix in range(nx):
                out[ix] += vals[base + ix]
    return out


def lateral_sigma_mm(
    vals: list[float] | array.array,
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> list[dict]:
    """Dose-weighted lateral RMS radius/sqrt(2) per depth bin (patient X = ix)."""
    nx, ny, nz = shape
    sx, sy, sz = spacing_mm
    rows: list[dict] = []
    for ix in range(nx):
        sy_m = sz_m = w = 0.0
        for iz in range(nz):
            for iy in range(ny):
                v = vals[iz * nx * ny + iy * nx + ix]
                if v <= 0.0:
                    continue
                sy_m += iy * v
                sz_m += iz * v
                w += v
        if w <= 0.0:
            rows.append(
                {
                    "depth_bin": ix,
                    "depth_mm": (ix + 0.5) * sx,
                    "sigma_rms_mm": float("nan"),
                    "integral": 0.0,
                }
            )
            continue
        my, mz = sy_m / w, sz_m / w
        vy = vz = 0.0
        for iz in range(nz):
            for iy in range(ny):
                v = vals[iz * nx * ny + iy * nx + ix]
                if v <= 0.0:
                    continue
                vy += v * (iy - my) ** 2
                vz += v * (iz - mz) ** 2
        # convert voxel index variance to mm
        sig = math.sqrt(0.5 * (vy * sy * sy + vz * sz * sz) / w)
        rows.append(
            {
                "depth_bin": ix,
                "depth_mm": (ix + 0.5) * sx,
                "sigma_rms_mm": sig,
                "integral": w,
            }
        )
    return rows


def plane_com_yz(
    vals: list[float] | array.array,
    shape: tuple[int, int, int],
    depth_bin: int,
) -> tuple[float, float, float] | None:
    """Dose-weighted (COM_y, COM_z, integral) at fixed depth bin; bins not mm."""
    nx, ny, nz = shape
    sy = sz = w = 0.0
    for iz in range(nz):
        for iy in range(ny):
            v = vals[iz * nx * ny + iy * nx + depth_bin]
            if v <= 0:
                continue
            sy += iy * v
            sz += iz * v
            w += v
    if w <= 0:
        return None
    return sy / w, sz / w, w


def central_lateral_profile(
    vals: list[float] | array.array,
    shape: tuple[int, int, int],
    depth_bin: int,
    axis: str = "y",
) -> list[tuple[int, float]]:
    """1D lateral profile at fixed depth through COM of that plane."""
    nx, ny, nz = shape
    com = plane_com_yz(vals, shape, depth_bin)
    if com is None:
        return []
    cy = max(0, min(ny - 1, int(round(com[0]))))
    cz = max(0, min(nz - 1, int(round(com[1]))))
    prof: list[tuple[int, float]] = []
    if axis == "y":
        for iy in range(ny):
            prof.append((iy, vals[cz * nx * ny + iy * nx + depth_bin]))
    else:
        for iz in range(nz):
            prof.append((iz, vals[iz * nx * ny + cy * nx + depth_bin]))
    return prof


def fwhm_bins(prof: list[tuple[int, float]]) -> float | None:
    """Linear-interpolated FWHM in bin units from a 1D profile."""
    if not prof:
        return None
    peak = max(v for _, v in prof)
    if peak <= 0:
        return None
    half = 0.5 * peak
    xs = [float(i) for i, _ in prof]
    ys = [v for _, v in prof]
    # left crossing
    left = right = None
    for i in range(1, len(ys)):
        if ys[i - 1] < half <= ys[i] or ys[i - 1] > half >= ys[i]:
            if ys[i] == ys[i - 1]:
                x = xs[i]
            else:
                t = (half - ys[i - 1]) / (ys[i] - ys[i - 1])
                x = xs[i - 1] + t * (xs[i] - xs[i - 1])
            if left is None:
                left = x
            right = x
    if left is None or right is None or right <= left:
        return None
    return right - left


def profile_corr(
    a: list[tuple[int, float]], b: list[tuple[int, float]]
) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    va = [x[1] for x in a]
    vb = [x[1] for x in b]
    ma = sum(va) / len(va)
    mb = sum(vb) / len(vb)
    num = den_a = den_b = 0.0
    for x, y in zip(va, vb):
        da, db = x - ma, y - mb
        num += da * db
        den_a += da * da
        den_b += db * db
    if den_a <= 0 or den_b <= 0:
        return None
    return num / math.sqrt(den_a * den_b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-mhd", type=Path, required=True)
    ap.add_argument("--topas-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="prelim")
    ap.add_argument("--gamma-points", type=int, default=12000)
    ap.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    ap.add_argument(
        "--flip-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip patient X when mapping GPU (default True for TOPAS Patient CSV)",
    )
    ap.add_argument(
        "--flip-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flip patient Y when mapping GPU (default False for TOPAS Patient CSV)",
    )
    ap.add_argument(
        "--skip-global-gamma",
        action="store_true",
        help="Skip global gamma for a quick diagnostic run",
    )
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = 104, 126, 35
    spacing = (2.0, 2.0, 2.0)
    sx, sy, sz = spacing
    topas = load_topas_csv(args.topas_csv, nx, ny, nz)
    if max(topas) <= 0:
        raise SystemExit(f"TOPAS dose empty: {args.topas_csv}")

    gmeta, gpu = read_mhd(args.gpu_mhd)
    gdim = tuple(int(x) for x in gmeta["DimSize"].split())
    flip_x = bool(args.flip_x)
    flip_y = bool(args.flip_y)
    mapped = map_gpu_to_physical(
        gpu, gdim, (417, 505, 35), (nx, ny, nz), flip_x, flip_y
    )
    thr_frac = 0.1
    scale = fit_scale(mapped, topas, thr_frac)
    scaled = [scale * v for v in mapped]

    sab = saa = sbb = ss = 0.0
    n = 0
    thr = thr_frac * max(topas)
    for a, b in zip(scaled, topas):
        sab += a * b
        saa += a * a
        sbb += b * b
        if b >= thr:
            ss += (a - b) ** 2
            n += 1
    cos = sab / math.sqrt(saa * sbb) if saa and sbb else -1.0
    nrmse = math.sqrt(ss / n) / max(topas) if n else None

    idd_t = idd_x(topas, (nx, ny, nz))
    idd_g = idd_x(scaled, (nx, ny, nz))
    num = den_t = den_g = 0.0
    for a, b in zip(idd_g, idd_t):
        num += a * b
        den_t += b * b
        den_g += a * a
    idd_corr = num / math.sqrt(den_t * den_g) if den_t and den_g else 0.0
    peak_t = idd_t.index(max(idd_t))
    peak_g = idd_g.index(max(idd_g))

    sig_t = lateral_sigma_mm(topas, (nx, ny, nz), spacing)
    sig_g = lateral_sigma_mm(scaled, (nx, ny, nz), spacing)

    # sigma + COM lateral rows where both integrals are significant
    max_int_t = max((r["integral"] for r in sig_t), default=0.0)
    sig_rows = []
    com_dy = []
    com_dz = []
    for st, sg in zip(sig_t, sig_g):
        if st["integral"] <= 0 or sg["integral"] <= 0:
            continue
        if not (st["sigma_rms_mm"] == st["sigma_rms_mm"] and sg["sigma_rms_mm"] == sg["sigma_rms_mm"]):
            continue
        if st["integral"] < 0.01 * max_int_t:
            continue
        ix = st["depth_bin"]
        com_t = plane_com_yz(topas, (nx, ny, nz), ix)
        com_g = plane_com_yz(scaled, (nx, ny, nz), ix)
        dy_mm = dz_mm = None
        if com_t and com_g:
            dy_mm = (com_g[0] - com_t[0]) * sy
            dz_mm = (com_g[1] - com_t[1]) * sz
            com_dy.append(dy_mm)
            com_dz.append(dz_mm)
        ratio = sg["sigma_rms_mm"] / st["sigma_rms_mm"] if st["sigma_rms_mm"] > 1e-9 else None
        sig_rows.append(
            {
                "depth_bin": ix,
                "depth_mm": st["depth_mm"],
                "sigma_topas_mm": st["sigma_rms_mm"],
                "sigma_gpu_mm": sg["sigma_rms_mm"],
                "ratio_gpu_over_topas": ratio,
                "com_y_topas_bin": com_t[0] if com_t else None,
                "com_y_gpu_bin": com_g[0] if com_g else None,
                "com_z_topas_bin": com_t[1] if com_t else None,
                "com_z_gpu_bin": com_g[1] if com_g else None,
                "com_dy_mm": dy_mm,
                "com_dz_mm": dz_mm,
                "integral_topas": st["integral"],
                "integral_gpu": sg["integral"],
            }
        )

    # Global gamma is the acceptance metric; local gamma is a stricter
    # diagnostic whose per-voxel denominator strongly amplifies MC noise.
    g_local = gamma_interpolated(
        topas, scaled, (nx, ny, nz), spacing, 3.0, 3.0, 10.0,
        args.gamma_points, 0, local_dose=True,
        interpolation_step_mm=args.gamma_resolution_mm,
    )
    g_local_5 = gamma_interpolated(
        topas, scaled, (nx, ny, nz), spacing, 3.0, 5.0, 10.0,
        args.gamma_points, 0, local_dose=True,
        interpolation_step_mm=args.gamma_resolution_mm,
    )
    g_global = None
    if not args.skip_global_gamma:
        g_global = gamma_interpolated(
            topas, scaled, (nx, ny, nz), spacing, 3.0, 3.0, 10.0,
            args.gamma_points, 0,
            interpolation_step_mm=args.gamma_resolution_mm,
        )

    # Lateral profiles at TOPAS IDD peak depth (same plane for both)
    prof_y_t = central_lateral_profile(topas, (nx, ny, nz), peak_t, "y")
    prof_y_g = central_lateral_profile(scaled, (nx, ny, nz), peak_t, "y")
    prof_z_t = central_lateral_profile(topas, (nx, ny, nz), peak_t, "z")
    prof_z_g = central_lateral_profile(scaled, (nx, ny, nz), peak_t, "z")
    # Also profiles at each field's own peak (morphology without depth shift)
    prof_y_g_own = central_lateral_profile(scaled, (nx, ny, nz), peak_g, "y")
    prof_z_g_own = central_lateral_profile(scaled, (nx, ny, nz), peak_g, "z")

    fwhm_y_t = fwhm_bins(prof_y_t)
    fwhm_y_g = fwhm_bins(prof_y_g)
    fwhm_z_t = fwhm_bins(prof_z_t)
    fwhm_z_g = fwhm_bins(prof_z_g)
    fwhm_y_g_own = fwhm_bins(prof_y_g_own)
    fwhm_z_g_own = fwhm_bins(prof_z_g_own)

    # write CSVs
    with (out / f"{args.tag}_idd.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_bin", "depth_mm", "idd_topas", "idd_gpu_scaled", "rel_diff"])
        for i, (a, b) in enumerate(zip(idd_t, idd_g)):
            rel = (b - a) / a if a > 0 else float("nan")
            w.writerow([i, (i + 0.5) * spacing[0], f"{a:.6g}", f"{b:.6g}", f"{rel:.6g}"])

    with (out / f"{args.tag}_lateral_sigma.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "depth_bin",
                "depth_mm",
                "sigma_topas_mm",
                "sigma_gpu_mm",
                "ratio_gpu_over_topas",
                "com_dy_mm",
                "com_dz_mm",
                "integral_topas",
                "integral_gpu",
            ]
        )
        for r in sig_rows:
            w.writerow(
                [
                    r["depth_bin"],
                    f"{r['depth_mm']:.4g}",
                    f"{r['sigma_topas_mm']:.6g}",
                    f"{r['sigma_gpu_mm']:.6g}",
                    f"{r['ratio_gpu_over_topas']:.6g}" if r["ratio_gpu_over_topas"] else "",
                    f"{r['com_dy_mm']:.4g}" if r["com_dy_mm"] is not None else "",
                    f"{r['com_dz_mm']:.4g}" if r["com_dz_mm"] is not None else "",
                    f"{r['integral_topas']:.6g}",
                    f"{r['integral_gpu']:.6g}",
                ]
            )

    with (out / f"{args.tag}_lateral_profile_peak_y.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["y_bin", "y_mm", "dose_topas", "dose_gpu_scaled"])
        for (iy, dt), (_, dg) in zip(prof_y_t, prof_y_g):
            w.writerow([iy, (iy + 0.5) * sy, f"{dt:.6g}", f"{dg:.6g}"])

    with (out / f"{args.tag}_lateral_profile_peak_z.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["z_bin", "z_mm", "dose_topas", "dose_gpu_scaled"])
        for (iz, dt), (_, dg) in zip(prof_z_t, prof_z_g):
            w.writerow([iz, (iz + 0.5) * sz, f"{dt:.6g}", f"{dg:.6g}"])

    write_mhd(
        out / f"{args.tag}_topas.mhd",
        topas,
        (nx, ny, nz),
        spacing,
        (-102.75, -43.55, -814.19),
        "Gy",
    )
    write_mhd(
        out / f"{args.tag}_gpu_scaled.mhd",
        array.array("f", scaled),
        (nx, ny, nz),
        spacing,
        (-102.75, -43.55, -814.19),
        "Gy",
    )

    ratios = [r["ratio_gpu_over_topas"] for r in sig_rows if r["ratio_gpu_over_topas"]]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    # Near-peak lateral ratios (more robust than whole-track mean for pencil)
    near = [
        r["ratio_gpu_over_topas"]
        for r in sig_rows
        if r["ratio_gpu_over_topas"]
        and abs(r["depth_bin"] - peak_t) <= 5
    ]
    mean_ratio_near_peak = sum(near) / len(near) if near else None

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _rms(xs: list[float]) -> float | None:
        return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None

    report = {
        "tag": args.tag,
        "gpu_mhd": str(args.gpu_mhd),
        "topas_csv": str(args.topas_csv),
        "flip_x": flip_x,
        "flip_y": flip_y,
        "scale_gpu_to_topas": scale,
        "cosine_3d": cos,
        "nrmse_high_dose": nrmse,
        "idd_correlation": idd_corr,
        "idd_peak_bin_topas": peak_t,
        "idd_peak_bin_gpu": peak_g,
        "idd_peak_shift_bins": (peak_t - peak_g) if flip_x else (peak_g - peak_t),
        "idd_peak_shift_mm": ((peak_t - peak_g) if flip_x else (peak_g - peak_t)) * sx,
        "patient_x_peak_shift_bins": peak_g - peak_t,
        "patient_x_peak_shift_mm": (peak_g - peak_t) * sx,
        "lateral": {
            "sigma_mean_ratio_gpu_over_topas": mean_ratio,
            "sigma_mean_ratio_near_topas_peak_pm5bins": mean_ratio_near_peak,
            "sigma_n_depth_bins": len(sig_rows),
            "com_dy_mean_mm": _mean(com_dy),
            "com_dz_mean_mm": _mean(com_dz),
            "com_dy_rms_mm": _rms(com_dy),
            "com_dz_rms_mm": _rms(com_dz),
            "profile_at_topas_peak_depth_bin": peak_t,
            "fwhm_y_topas_mm": fwhm_y_t * sy if fwhm_y_t is not None else None,
            "fwhm_y_gpu_mm": fwhm_y_g * sy if fwhm_y_g is not None else None,
            "fwhm_z_topas_mm": fwhm_z_t * sz if fwhm_z_t is not None else None,
            "fwhm_z_gpu_mm": fwhm_z_g * sz if fwhm_z_g is not None else None,
            "fwhm_y_gpu_at_own_peak_mm": (
                fwhm_y_g_own * sy if fwhm_y_g_own is not None else None
            ),
            "fwhm_z_gpu_at_own_peak_mm": (
                fwhm_z_g_own * sz if fwhm_z_g_own is not None else None
            ),
            "profile_y_corr_at_topas_peak": profile_corr(prof_y_t, prof_y_g),
            "profile_z_corr_at_topas_peak": profile_corr(prof_z_t, prof_z_g),
        },
        "gamma_primary_global_3pct_3mm_thr10": g_global,
        "gamma_local_3pct_3mm_thr10_diagnostic": g_local,
        "gamma_local_3pct_5mm_thr10_diagnostic": g_local_5,
        "note": (
            "Primary acceptance metric is global gamma on reference voxels above "
            "10% peak. Local gamma is retained as a stricter MC-noise/physics "
            "diagnostic (dose criterion = % of each local reference value). "
            "Lateral metrics: sigma ratio, COM offset, FWHM Y/Z, profile corr."
        ),
    }
    (out / f"{args.tag}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
