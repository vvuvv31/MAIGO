#!/usr/bin/env python3
"""Plot mapped full-plan GPU dose against the matRad physical-dose cube."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_mhd(path: Path) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    meta = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    nx, ny, nz = (int(value) for value in meta["DimSize"].split())
    spacing = tuple(float(value) for value in meta["ElementSpacing"].split())
    offset = tuple(float(value) for value in meta.get("Offset", "0 0 0").split())
    raw = np.fromfile(path.parent / meta["ElementDataFile"], dtype="<f4")
    if raw.size != nx * ny * nz:
        raise ValueError(f"{path}: inconsistent RAW size")
    return raw.reshape(nz, ny, nx), spacing, offset


def axis_centers(count: int, spacing: float, offset: float) -> np.ndarray:
    return offset + np.arange(count) * spacing


def extent(centers_a: np.ndarray, centers_b: np.ndarray) -> list[float]:
    da = centers_a[1] - centers_a[0]
    db = centers_b[1] - centers_b[0]
    return [
        centers_a[0] - da / 2,
        centers_a[-1] + da / 2,
        centers_b[0] - db / 2,
        centers_b[-1] + db / 2,
    ]


def save_both(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    print(f"Wrote {base.with_suffix('.png')}")
    print(f"Wrote {base.with_suffix('.svg')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--gpu-calibrated", type=Path, required=True)
    parser.add_argument("--gpu-weighted-equivalent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    physical, spacing, offset = read_mhd(args.physical)
    gpu, gpu_spacing, gpu_offset = read_mhd(args.gpu_calibrated)
    equivalent, eq_spacing, eq_offset = read_mhd(args.gpu_weighted_equivalent)
    if physical.shape != gpu.shape or physical.shape != equivalent.shape:
        raise ValueError("all mapped dose cubes must have the same shape")
    if spacing != gpu_spacing or spacing != eq_spacing or offset != gpu_offset or offset != eq_offset:
        raise ValueError("all mapped dose cubes must have identical geometry")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = physical.shape
    x = axis_centers(nx, spacing[0], offset[0])
    y = axis_centers(ny, spacing[1], offset[1])
    z = axis_centers(nz, spacing[2], offset[2])
    iz, iy, ix = np.unravel_index(int(np.argmax(physical)), physical.shape)

    vmax = float(np.max(physical))
    diff_limit = max(float(np.percentile(np.abs(gpu - physical), 99.5)), 0.01 * vmax)
    slices = [
        (lambda a: a[iz, :, :], extent(x, y), "Patient XY", "X (mm)", "Y (mm)"),
        (lambda a: a[:, iy, :], extent(x, z), "Patient XZ", "X (mm)", "Z (mm)"),
        (lambda a: a[:, :, ix], extent(y, z), "Patient YZ", "Y (mm)", "Z (mm)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    last_dose = last_diff = None
    for column, (select, image_extent, plane, xlabel, ylabel) in enumerate(slices):
        for row, (data, label) in enumerate(
            [(physical, "matRad physical dose"), (gpu, "GPU, LS dose-calibrated")]
        ):
            last_dose = axes[row, column].imshow(
                select(data),
                origin="lower",
                extent=image_extent,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
                aspect="auto",
            )
            axes[row, column].set_title(f"{label} — {plane}")
            axes[row, column].set_xlabel(xlabel)
            axes[row, column].set_ylabel(ylabel)
        last_diff = axes[2, column].imshow(
            select(gpu - physical),
            origin="lower",
            extent=image_extent,
            vmin=-diff_limit,
            vmax=diff_limit,
            cmap="coolwarm",
            aspect="auto",
        )
        axes[2, column].set_title(f"GPU − matRad — {plane}")
        axes[2, column].set_xlabel(xlabel)
        axes[2, column].set_ylabel(ylabel)
    fig.colorbar(last_dose, ax=axes[:2, :], label="Dose (Gy)", shrink=0.75)
    fig.colorbar(last_diff, ax=axes[2, :], label="Dose difference (Gy)", shrink=0.75)
    fig.suptitle(
        f"Full-plan orthogonal slices through reference peak (x,y,z index={ix},{iy},{iz})"
    )
    save_both(fig, args.output_dir / "full_plan_orthogonal_slices")
    plt.close(fig)

    idd_physical = np.sum(physical, axis=(0, 1))
    idd_gpu = np.sum(gpu, axis=(0, 1))
    idd_equivalent = np.sum(equivalent, axis=(0, 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(x, idd_physical, label="matRad physical", lw=2)
    axes[0].plot(x, idd_gpu, label="GPU calibrated to physical", lw=1.7)
    axes[0].plot(x, idd_equivalent, label="GPU weighted-history equivalent", lw=1.3)
    axes[0].set_xlabel("Patient X (mm)")
    axes[0].set_ylabel("Integrated dose (Gy, summed over YZ)")
    axes[0].set_title("Absolute IDD / depth profile")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for values, label in [
        (idd_physical, "matRad physical"),
        (idd_gpu, "GPU calibrated"),
        (idd_equivalent, "GPU weighted-history equivalent"),
    ]:
        axes[1].plot(x, values / np.max(values), label=label)
    axes[1].set_xlabel("Patient X (mm)")
    axes[1].set_ylabel("Normalized IDD")
    axes[1].set_title("IDD shape")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    save_both(fig, args.output_dir / "full_plan_idd")
    plt.close(fig)


if __name__ == "__main__":
    main()
