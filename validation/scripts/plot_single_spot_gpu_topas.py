#!/usr/bin/env python3
"""Plot the fixed single-spot GPU/TOPAS comparison without spatial fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_mhd(path: Path) -> tuple[dict[str, str], np.ndarray]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    nx, ny, nz = (int(v) for v in meta["DimSize"].split())
    raw = path.parent / meta["ElementDataFile"]
    values = np.fromfile(raw, dtype="<f4")
    if values.size != nx * ny * nz:
        raise ValueError(f"{raw}: expected {nx * ny * nz} floats, got {values.size}")
    return meta, values.reshape((nz, ny, nx))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Wrote {path}")


def dose_triptych(
    ref: np.ndarray,
    gpu: np.ndarray,
    extent: tuple[float, float, float, float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    ref_max = float(np.max(ref))
    if ref_max <= 0.0:
        raise ValueError(f"Empty reference plane for {title}")
    ref_pct = 100.0 * ref / ref_max
    gpu_pct = 100.0 * gpu / ref_max
    diff = gpu_pct - ref_pct
    diff_limit = max(5.0, float(np.percentile(np.abs(diff), 99.0)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for ax, values, panel in zip(
        axes[:2], (ref_pct, gpu_pct), ("TOPAS", "GPU (scaled)"), strict=True
    ):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=100.0,
            interpolation="nearest",
        )
        ax.set_title(panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(image, ax=ax, label="Dose (% of TOPAS panel maximum)")

    image = axes[2].imshow(
        diff,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
        interpolation="nearest",
    )
    axes[2].set_title("GPU - TOPAS")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    fig.colorbar(image, ax=axes[2], label="Dose difference (percentage points)")
    fig.suptitle(title, fontsize=14)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("out/ct/prelim_single/topas_compare_fixed"),
    )
    parser.add_argument(
        "--tag",
        default="single_fixed",
        help="Input filename prefix produced by compare_gpu_topas_prelim.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/ct/prelim_single/plots"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topas_meta, topas = read_mhd(args.comparison_dir / f"{args.tag}_topas.mhd")
    gpu_meta, gpu = read_mhd(args.comparison_dir / f"{args.tag}_gpu_scaled.mhd")
    if topas.shape != gpu.shape:
        raise ValueError(f"Shape mismatch: TOPAS={topas.shape}, GPU={gpu.shape}")
    if topas_meta["ElementSpacing"] != gpu_meta["ElementSpacing"]:
        raise ValueError("GPU/TOPAS spacing mismatch")

    metrics = json.loads(
        (args.comparison_dir / f"{args.tag}_metrics.json").read_text(encoding="utf-8")
    )
    sx, sy, sz = (float(v) for v in topas_meta["ElementSpacing"].split())
    nz, ny, nx = topas.shape

    # Patient X index increases opposite to beam travel. Reverse it so the plot
    # coordinate is physical depth from the +patient-X entrance face.
    depth_mm = (np.arange(nx, dtype=np.float64) + 0.5) * sx
    idd_topas = np.sum(topas, axis=(0, 1))[::-1]
    idd_gpu = np.sum(gpu, axis=(0, 1))[::-1]
    topas_peak_x = int(np.argmax(np.sum(topas, axis=(0, 1))))
    gpu_peak_x = int(np.argmax(np.sum(gpu, axis=(0, 1))))
    topas_peak_depth = (nx - topas_peak_x - 0.5) * sx
    gpu_peak_depth = (nx - gpu_peak_x - 0.5) * sx

    y_mm = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * sy
    z_mm = (np.arange(nz, dtype=np.float64) - 0.5 * (nz - 1)) * sz
    yz_topas = topas[:, :, topas_peak_x]
    yz_gpu = gpu[:, :, topas_peak_x]

    # Transverse YZ dose at the TOPAS Bragg-peak depth.
    fig = dose_triptych(
        yz_topas,
        yz_gpu,
        (y_mm[0] - sy / 2, y_mm[-1] + sy / 2, z_mm[0] - sz / 2, z_mm[-1] + sz / 2),
        "Patient Y (mm)",
        "Patient Z (mm)",
        f"Single spot transverse YZ dose at TOPAS peak depth ({topas_peak_depth:.1f} mm)",
    )
    save_figure(fig, args.output_dir, "single_spot_transverse_yz_peak")
    plt.close(fig)

    # Patient XY projection: sum over patient Z, then reverse X into beam depth.
    xy_topas = np.sum(topas, axis=0)[:, ::-1]
    xy_gpu = np.sum(gpu, axis=0)[:, ::-1]
    fig = dose_triptych(
        xy_topas,
        xy_gpu,
        (0.0, nx * sx, y_mm[0] - sy / 2, y_mm[-1] + sy / 2),
        "Beam depth from +patient-X entrance (mm)",
        "Patient Y (mm)",
        "Single spot patient-XY dose projection (summed over patient Z)",
    )
    for ax in fig.axes:
        if ax.get_xlabel().startswith("Beam depth"):
            ax.axvline(topas_peak_depth, color="cyan", lw=0.8, ls="--", alpha=0.8)
    save_figure(fig, args.output_dir, "single_spot_patient_xy_projection")
    plt.close(fig)

    # IDD and integrated transverse profiles at the same TOPAS peak plane.
    idd_norm = float(np.max(idd_topas))
    y_topas = np.sum(yz_topas, axis=0)
    y_gpu = np.sum(yz_gpu, axis=0)
    z_topas = np.sum(yz_topas, axis=1)
    z_gpu = np.sum(yz_gpu, axis=1)
    y_norm = float(np.max(y_topas))
    z_norm = float(np.max(z_topas))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].plot(depth_mm, 100.0 * idd_topas / idd_norm, lw=2.2, label="TOPAS")
    axes[0].plot(depth_mm, 100.0 * idd_gpu / idd_norm, lw=1.8, label="GPU (scaled)")
    axes[0].axvline(topas_peak_depth, color="0.4", ls="--", lw=0.9)
    axes[0].set(
        xlabel="Beam depth from +patient-X entrance (mm)",
        ylabel="Integrated depth dose (% of TOPAS peak)",
        title=f"IDD: peaks {topas_peak_depth:.1f}/{gpu_peak_depth:.1f} mm",
        xlim=(0.0, nx * sx),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(y_mm, 100.0 * y_topas / y_norm, lw=2.2, label="TOPAS")
    axes[1].plot(y_mm, 100.0 * y_gpu / y_norm, lw=1.8, label="GPU (scaled)")
    axes[1].set(
        xlabel="Patient Y (mm)",
        ylabel="Integrated lateral dose (%)",
        title="Y profile at TOPAS peak depth",
        xlim=(float(y_mm[0]), float(y_mm[-1])),
    )
    axes[1].grid(alpha=0.25)

    axes[2].plot(z_mm, 100.0 * z_topas / z_norm, lw=2.2, label="TOPAS")
    axes[2].plot(z_mm, 100.0 * z_gpu / z_norm, lw=1.8, label="GPU (scaled)")
    axes[2].set(
        xlabel="Patient Z (mm)",
        ylabel="Integrated lateral dose (%)",
        title="Z profile at TOPAS peak depth",
        xlim=(float(z_mm[0]), float(z_mm[-1])),
    )
    axes[2].grid(alpha=0.25)

    global_gamma = metrics["gamma_primary_global_3pct_3mm_thr10"]["pass_percent"]
    local_gamma = metrics["gamma_local_3pct_3mm_thr10_diagnostic"]["pass_percent"]
    fig.suptitle(
        "Single spot GPU vs TOPAS | "
        f"global gamma 3%/3mm = {global_gamma:.2f}% | local gamma = {local_gamma:.2f}%",
        fontsize=14,
    )
    save_figure(fig, args.output_dir, "single_spot_idd_lateral_profiles")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
