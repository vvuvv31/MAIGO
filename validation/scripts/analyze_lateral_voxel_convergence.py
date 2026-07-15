#!/usr/bin/env python3
"""Analyze lateral voxel-size convergence from sparse GPU voxel CSVs.

Fixed FOV is assumed (bins * size constant). Absolute MeV/primary; no scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data["energy_deposition_MeV_per_primary"], dtype=float)),
    )


def load_sparse_voxels(
    path: Path, bins_x: int, bins_y: int, bins_z: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6), ndmin=2)
    dose = np.zeros((bins_x, bins_y, bins_z), dtype=np.float64)
    if rows.size == 0:
        x_mm = np.zeros(bins_x)
        y_mm = np.zeros(bins_y)
        z_mm = np.zeros(bins_z)
        return dose, x_mm, y_mm, z_mm
    ix = rows[:, 0].astype(np.int64)
    iy = rows[:, 1].astype(np.int64)
    iz = rows[:, 2].astype(np.int64)
    if (
        np.any(ix < 0)
        or np.any(iy < 0)
        or np.any(iz < 0)
        or np.any(ix >= bins_x)
        or np.any(iy >= bins_y)
        or np.any(iz >= bins_z)
    ):
        raise SystemExit(f"Voxel index out of range in {path}")
    dose[ix, iy, iz] = rows[:, 6]
    # Reconstruct centers from observed sparse samples when available.
    x_centers = np.full(bins_x, np.nan)
    y_centers = np.full(bins_y, np.nan)
    z_centers = np.full(bins_z, np.nan)
    for i, x, e in zip(ix, rows[:, 3], rows[:, 6]):
        if e > 0.0 and not np.isfinite(x_centers[i]):
            x_centers[i] = x
    for i, y, e in zip(iy, rows[:, 4], rows[:, 6]):
        if e > 0.0 and not np.isfinite(y_centers[i]):
            y_centers[i] = y
    for i, z, e in zip(iz, rows[:, 5], rows[:, 6]):
        if e > 0.0 and not np.isfinite(z_centers[i]):
            z_centers[i] = z
    # Fill missing centers by linear extrapolation of spacing.
    def complete(centers: np.ndarray, size_hint: float) -> np.ndarray:
        known = np.flatnonzero(np.isfinite(centers))
        if known.size == 0:
            return (np.arange(centers.size) + 0.5) * size_hint - 0.5 * centers.size * size_hint
        if known.size == 1:
            i0 = int(known[0])
            return centers[i0] + (np.arange(centers.size) - i0) * size_hint
        # median spacing from known pairs
        spacings = np.diff(centers[known]) / np.diff(known)
        spacing = float(np.median(spacings))
        i0 = int(known[0])
        return centers[i0] + (np.arange(centers.size) - i0) * spacing

    # size_hint from first two filled if possible
    def spacing_hint(centers: np.ndarray, n: int, fallback: float) -> float:
        known = np.flatnonzero(np.isfinite(centers))
        if known.size >= 2:
            return float(np.median(np.diff(centers[known]) / np.diff(known)))
        return fallback

    sx = spacing_hint(x_centers, bins_x, 5.0)
    sy = spacing_hint(y_centers, bins_y, 5.0)
    sz = spacing_hint(z_centers, bins_z, 0.5)
    x_mm = complete(x_centers, sx)
    y_mm = complete(y_centers, sy)
    z_mm = complete(z_centers, sz)
    return dose, x_mm, y_mm, z_mm


def weighted_sigma(coordinate: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(coordinate * weights) / total)
    variance = float(np.sum((coordinate - mean) ** 2 * weights) / total)
    return float(np.sqrt(max(0.0, variance)))


def lateral_at_depths(
    dose: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    depths: list[float],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for requested in depths:
        iz = int(np.argmin(np.abs(z_mm - requested)))
        plane = dose[:, :, iz]
        out[f"{z_mm[iz]:.2f}"] = {
            "requested_depth_mm": requested,
            "sigma_x_mm": weighted_sigma(x_mm, np.sum(plane, axis=1)),
            "sigma_y_mm": weighted_sigma(y_mm, np.sum(plane, axis=0)),
            "energy_MeV_per_primary": float(np.sum(plane)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help=(
            "voxel_size_mm:bins_x:bins_y:bins_z:idd_csv:voxel_csv "
            "e.g. 5:60:60:800:out/idd.csv:out/vox.csv"
        ),
    )
    parser.add_argument(
        "--depths-mm",
        nargs="+",
        type=float,
        default=[20.0, 50.0, 86.75],
    )
    parser.add_argument(
        "--reference-voxel-mm",
        type=float,
        default=None,
        help="Reference voxel size (default: finest/smallest)",
    )
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=None)
    args = parser.parse_args()

    cases: list[dict[str, object]] = []
    for item in args.cases:
        parts = item.split(":")
        if len(parts) != 6:
            raise SystemExit(f"Bad case spec: {item}")
        size = float(parts[0])
        bx, by, bz = int(parts[1]), int(parts[2]), int(parts[3])
        idd_path = Path(parts[4])
        voxel_path = Path(parts[5])
        depth, idd = load_idd(idd_path)
        dose, x_mm, y_mm, z_mm = load_sparse_voxels(voxel_path, bx, by, bz)
        # voxel->idd closure
        plane = bx * by
        reconstructed = dose.reshape(bx, by, bz).sum(axis=(0, 1))
        # dose array is (x,y,z); IDD should match
        if reconstructed.size != idd.size:
            # align by z centers if needed
            reconstructed = np.interp(depth, z_mm, dose.sum(axis=(0, 1)))
        closure = float(np.max(np.abs(reconstructed - idd)))
        widths = lateral_at_depths(dose, x_mm, y_mm, z_mm, args.depths_mm)
        cases.append(
            {
                "voxel_size_mm": size,
                "bins_xyz": [bx, by, bz],
                "fov_mm": [bx * size, by * size],
                "idd_path": idd_path.as_posix(),
                "voxel_path": voxel_path.as_posix(),
                "integral_MeV_per_primary": float(np.trapz(idd, depth)),
                "voxel_idd_closure_max_abs_MeV_per_primary_per_bin": closure,
                "lateral_widths": widths,
                "idd_depth_mm": depth.tolist(),
                "idd": idd.tolist(),
            }
        )

    cases.sort(key=lambda c: float(c["voxel_size_mm"]), reverse=True)
    sizes = [float(c["voxel_size_mm"]) for c in cases]
    ref_size = args.reference_voxel_mm if args.reference_voxel_mm is not None else min(sizes)
    reference = next(c for c in cases if abs(float(c["voxel_size_mm"]) - ref_size) < 1e-12)
    ref_widths: dict = reference["lateral_widths"]  # type: ignore[assignment]
    ref_integral = float(reference["integral_MeV_per_primary"])  # type: ignore[arg-type]
    ref_idd = np.asarray(reference["idd"], dtype=float)
    ref_depth = np.asarray(reference["idd_depth_mm"], dtype=float)

    for case in cases:
        idd = np.asarray(case["idd"], dtype=float)
        depth = np.asarray(case["idd_depth_mm"], dtype=float)
        eval_on_ref = np.interp(ref_depth, depth, idd)
        diff = eval_on_ref - ref_idd
        nrmse = float(np.sqrt(np.mean(diff * diff)) / np.max(ref_idd))
        width_delta: dict[str, dict[str, float]] = {}
        for key, ref_w in ref_widths.items():
            # match by nearest requested depth key in this case
            case_widths: dict = case["lateral_widths"]  # type: ignore[assignment]
            # find same requested depth
            match = None
            for ck, cw in case_widths.items():
                if abs(float(cw["requested_depth_mm"]) - float(ref_w["requested_depth_mm"])) < 1e-6:
                    match = cw
                    break
            if match is None:
                continue
            width_delta[key] = {
                "delta_sigma_x_mm": float(match["sigma_x_mm"] - ref_w["sigma_x_mm"]),
                "delta_sigma_y_mm": float(match["sigma_y_mm"] - ref_w["sigma_y_mm"]),
                "sigma_x_mm": float(match["sigma_x_mm"]),
                "sigma_y_mm": float(match["sigma_y_mm"]),
            }
        case["vs_reference"] = {
            "nrmse_idd_to_reference_max": nrmse,
            "integral_signed_percent": 100.0
            * (float(case["integral_MeV_per_primary"]) - ref_integral)
            / ref_integral,
            "lateral_width_deltas": width_delta,
        }
        case.pop("idd")
        case.pop("idd_depth_mm")

    report = {
        "normalization": "absolute MeV/primary; no global scale",
        "reference_voxel_size_mm": ref_size,
        "depths_mm": args.depths_mm,
        "neutron_deferred": True,
        "cases": cases,
        "recommendation": (
            "Production 5 mm lateral voxels remain acceptable if sigma differences "
            "versus 2.5 mm stay << 0.1 mm at peak; promote with 100k histories before "
            "paper claim."
        ),
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # left: sigma_x at each depth vs voxel size
        depth_keys = list(ref_widths.keys())
        for dkey in depth_keys:
            xs = []
            sigs = []
            for case in cases:
                size = float(case["voxel_size_mm"])
                deltas = case["vs_reference"]["lateral_width_deltas"]  # type: ignore[index]
                if dkey not in deltas:
                    continue
                xs.append(size)
                sigs.append(deltas[dkey]["sigma_x_mm"])
            axes[0].plot(xs, sigs, "o-", label=f"z≈{dkey} mm")
        axes[0].set_xlabel("lateral voxel size (mm)")
        axes[0].set_ylabel(r"$\sigma_x$ (mm)")
        axes[0].set_title("Lateral width vs voxel size")
        axes[0].invert_xaxis()
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        nrmse = [
            case["vs_reference"]["nrmse_idd_to_reference_max"]  # type: ignore[index]
            for case in cases
        ]
        sizes_plot = [float(c["voxel_size_mm"]) for c in cases]
        axes[1].plot(sizes_plot, nrmse, "s-", color="#d62728")
        axes[1].set_xlabel("lateral voxel size (mm)")
        axes[1].set_ylabel("IDD NRMSE to finest")
        axes[1].set_title("IDD sensitivity")
        axes[1].invert_xaxis()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_plot, dpi=140)
        plt.close(fig)

    print(f"Wrote {args.output_metrics}")
    if args.output_plot:
        print(f"Wrote {args.output_plot}")
    for case in cases:
        size = case["voxel_size_mm"]
        vs = case["vs_reference"]
        print(
            f"voxel={size} mm  IDD-NRMSE={vs['nrmse_idd_to_reference_max']:.4e}  "
            f"integralΔ={vs['integral_signed_percent']:+.3f}%  "
            f"closure={case['voxel_idd_closure_max_abs_MeV_per_primary_per_bin']:.3e}"
        )
        for key, delta in vs["lateral_width_deltas"].items():
            print(
                f"  z={key}: σx={delta['sigma_x_mm']:.3f} "
                f"(Δ{delta['delta_sigma_x_mm']:+.3f})  "
                f"σy={delta['sigma_y_mm']:.3f} "
                f"(Δ{delta['delta_sigma_y_mm']:+.3f})"
            )


if __name__ == "__main__":
    main()
