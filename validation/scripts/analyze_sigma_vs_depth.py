#!/usr/bin/env python3
"""Compute dose-weighted lateral sigma(z) from TOPAS/GPU voxel dose and compare."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sparse_voxels(
    path: Path,
    x_col: str = "x_mm",
    y_col: str = "y_mm",
    z_col: str = "z_mm",
    w_col: str = "energy_deposition_MeV_per_primary",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sparse voxel CSV (headered) with x,y,z,weight columns."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ws: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty CSV: {path}")
        # TOPAS raw DoseToMedium often has no header; detect numeric-only.
        first = next(reader, None)
        if first is None:
            raise SystemExit(f"No data rows: {path}")
        # If keys look like TOPAS raw (ix,iy,iz,...) handled elsewhere.
        if x_col not in first and w_col not in first:
            # rewind via generic loader for TOPAS raw
            return load_topas_raw_dose3d(path)
        # process first + rest
        def push(row: dict[str, str]) -> None:
            w = float(row[w_col])
            if w <= 0.0:
                return
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
            zs.append(float(row[z_col]))
            ws.append(w)

        push(first)
        for row in reader:
            push(row)
    if not ws:
        raise SystemExit(f"No positive weights in {path}")
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(ws, dtype=float),
    )


def load_topas_raw_dose3d(
    path: Path,
    histories: int,
    nx: int,
    ny: int,
    nz: int,
    half_xy_mm: float,
    half_z_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TOPAS CSV: ix, iy, iz, sum[, std...] over BeamScorer component."""
    dx = 2.0 * half_xy_mm / nx
    dy = 2.0 * half_xy_mm / ny
    dz = 2.0 * half_z_mm / nz
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ws: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p.strip() for p in stripped.split(",")]
            try:
                ix, iy, iz = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                value = float(parts[3])
            except (ValueError, IndexError):
                continue
            if value <= 0.0:
                continue
            # TOPAS bin centers in component local coords (origin at component center).
            x = -half_xy_mm + (ix + 0.5) * dx
            y = -half_xy_mm + (iy + 0.5) * dy
            z = -half_z_mm + (iz + 0.5) * dz
            # Convert z to depth from entrance (component HLZ=200 mm, entrance at -200).
            depth = z + half_z_mm
            xs.append(x)
            ys.append(y)
            zs.append(depth)
            ws.append(value / float(histories))
    if not ws:
        raise SystemExit(f"No positive TOPAS dose voxels in {path}")
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(ws, dtype=float),
    )


def sigma_vs_depth(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    depth_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    n = len(centers)
    sigma_x = np.full(n, np.nan)
    sigma_y = np.full(n, np.nan)
    sigma_r = np.full(n, np.nan)
    mean_x = np.full(n, np.nan)
    mean_y = np.full(n, np.nan)
    integral = np.zeros(n)
    iz = np.digitize(z, depth_edges) - 1
    for i in range(n):
        mask = iz == i
        if not np.any(mask):
            continue
        wi = w[mask]
        total = float(np.sum(wi))
        if total <= 0.0:
            continue
        integral[i] = total
        mx = float(np.sum(wi * x[mask]) / total)
        my = float(np.sum(wi * y[mask]) / total)
        mean_x[i] = mx
        mean_y[i] = my
        var_x = float(np.sum(wi * (x[mask] - mx) ** 2) / total)
        var_y = float(np.sum(wi * (y[mask] - my) ** 2) / total)
        sigma_x[i] = np.sqrt(max(var_x, 0.0))
        sigma_y[i] = np.sqrt(max(var_y, 0.0))
        # 2D RMS radius / sqrt(2) ~ sigma for circular Gaussian
        sigma_r[i] = np.sqrt(max(0.5 * (var_x + var_y), 0.0))
    return {
        "depth_mm": centers,
        "sigma_x_mm": sigma_x,
        "sigma_y_mm": sigma_y,
        "sigma_rms_mm": sigma_r,
        "mean_x_mm": mean_x,
        "mean_y_mm": mean_y,
        "integral_MeV_per_primary": integral,
    }


def write_sigma_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "depth_mm",
                "sigma_x_mm",
                "sigma_y_mm",
                "sigma_rms_mm",
                "mean_x_mm",
                "mean_y_mm",
                "integral_MeV_per_primary",
            ]
        )
        for i in range(len(data["depth_mm"])):
            writer.writerow(
                [
                    f"{data['depth_mm'][i]:.6g}",
                    f"{data['sigma_x_mm'][i]:.6g}",
                    f"{data['sigma_y_mm'][i]:.6g}",
                    f"{data['sigma_rms_mm'][i]:.6g}",
                    f"{data['mean_x_mm'][i]:.6g}",
                    f"{data['mean_y_mm'][i]:.6g}",
                    f"{data['integral_MeV_per_primary'][i]:.6g}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-voxels", type=Path, required=True)
    parser.add_argument("--topas-dose3d", type=Path, required=True)
    parser.add_argument("--histories-topas", type=int, default=100000)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument("--nz", type=int, default=400)
    parser.add_argument("--half-xy-mm", type=float, default=30.0)
    parser.add_argument("--half-z-mm", type=float, default=200.0)
    parser.add_argument("--depth-bin-mm", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/results/emittance_200_sigma"),
    )
    args = parser.parse_args()

    gx, gy, gz, gw = load_sparse_voxels(args.gpu_voxels)
    tx, ty, tz, tw = load_topas_raw_dose3d(
        args.topas_dose3d,
        histories=args.histories_topas,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        half_xy_mm=args.half_xy_mm,
        half_z_mm=args.half_z_mm,
    )

    depth_max = args.half_z_mm * 2.0
    edges = np.arange(0.0, depth_max + 0.5 * args.depth_bin_mm, args.depth_bin_mm)
    gpu = sigma_vs_depth(gx, gy, gz, gw, edges)
    topas = sigma_vs_depth(tx, ty, tz, tw, edges)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sigma_csv(args.output_dir / "gpu_sigma_vs_depth.csv", gpu)
    write_sigma_csv(args.output_dir / "topas_sigma_vs_depth.csv", topas)

    # Compare on common finite depths
    mask = (
        np.isfinite(gpu["sigma_rms_mm"])
        & np.isfinite(topas["sigma_rms_mm"])
        & (gpu["integral_MeV_per_primary"] > 0)
        & (topas["integral_MeV_per_primary"] > 0)
    )
    depth = gpu["depth_mm"][mask]
    g_sig = gpu["sigma_rms_mm"][mask]
    t_sig = topas["sigma_rms_mm"][mask]
    rel = 100.0 * (g_sig - t_sig) / np.maximum(t_sig, 1e-12)

    metrics = {
        "histories_topas": args.histories_topas,
        "source": {
            "type": "emittance BiGaussian",
            "SigmaX_mm": 0.2,
            "SigmaXprime": 0.032,
            "CorrelationX": -0.9411,
            "SigmaY_mm": 0.2,
            "SigmaYprime": 0.032,
            "CorrelationY": 0.9411,
        },
        "entrance_sigma_rms_mm": {
            "gpu": float(g_sig[0]) if len(g_sig) else None,
            "topas": float(t_sig[0]) if len(t_sig) else None,
        },
        "mean_abs_sigma_diff_mm": float(np.mean(np.abs(g_sig - t_sig))) if len(g_sig) else None,
        "mean_rel_sigma_percent": float(np.mean(rel)) if len(rel) else None,
        "rms_rel_sigma_percent": float(np.sqrt(np.mean(rel * rel))) if len(rel) else None,
        "peak_depth_region": {},
    }
    # around Bragg peak ~86 mm for 200 MeV/u
    peak_mask = (depth >= 70.0) & (depth <= 100.0)
    if np.any(peak_mask):
        metrics["peak_depth_region"] = {
            "mean_rel_percent": float(np.mean(rel[peak_mask])),
            "mean_abs_diff_mm": float(np.mean(np.abs(g_sig[peak_mask] - t_sig[peak_mask]))),
        }

    (args.output_dir / "sigma_vs_depth_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(topas["depth_mm"], topas["sigma_rms_mm"], lw=2, label="TOPAS σ_rms")
    ax.plot(gpu["depth_mm"], gpu["sigma_rms_mm"], lw=1.5, label="GPU σ_rms")
    ax.plot(topas["depth_mm"], topas["sigma_x_mm"], lw=1.0, ls="--", alpha=0.7, label="TOPAS σx")
    ax.plot(gpu["depth_mm"], gpu["sigma_x_mm"], lw=1.0, ls=":", alpha=0.7, label="GPU σx")
    ax.set(xlabel="Depth (mm)", ylabel="Dose-weighted sigma (mm)", title="200 MeV/u emittance: σ(z)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(depth, rel, lw=1.4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set(
        xlabel="Depth (mm)",
        ylabel="(GPU-TOPAS)/TOPAS %",
        title="Relative σ_rms residual",
        ylim=(-30, 30),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plot_path = args.output_dir / "sigma_vs_depth.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output_dir / 'gpu_sigma_vs_depth.csv'}")
    print(f"Wrote {args.output_dir / 'topas_sigma_vs_depth.csv'}")
    print(f"Wrote {args.output_dir / 'sigma_vs_depth_metrics.json'}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
