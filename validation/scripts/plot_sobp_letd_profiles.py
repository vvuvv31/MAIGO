#!/usr/bin/env python3
"""Plot matched-range 1D and 2D GPU/TOPAS SOBP all-hadron LET_d profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def read_topas(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", delimiter=",")
    volume = np.zeros(shape, dtype=np.float64)
    volume[
        rows[:, 2].astype(int),
        rows[:, 1].astype(int),
        rows[:, 0].astype(int),
    ] = rows[:, 3]
    return volume


def read_mhd(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    nx, ny, nz = map(int, fields["DimSize"].split())
    spacing = tuple(map(float, fields["ElementSpacing"].split()))
    raw = np.fromfile(path.parent / fields["ElementDataFile"], dtype="<f4")
    return raw.reshape(nz, ny, nx).astype(np.float64), spacing


def nearest_index(coordinates: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coordinates - value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path(
            "out/letd_sobp/3d_lowenergy/"
            "gpu_sobp_letd_3d_all_hadron.mhd"
        ),
    )
    parser.add_argument(
        "--topas",
        type=Path,
        default=Path(
            "validation/topas/output/"
            "sobp_letd_3d_100k_all_hadron.csv"
        ),
    )
    parser.add_argument(
        "--gpu-dose",
        type=Path,
        default=Path("out/letd_sobp/3d_lowenergy/gpu_sobp_dose_3d.mhd"),
    )
    parser.add_argument(
        "--topas-dose",
        type=Path,
        default=Path(
            "validation/topas/output/"
            "sobp_water_3cm_5_10cm_3mm_dose3d.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/letd_sobp/3d_lowenergy/profiles"),
    )
    args = parser.parse_args()

    gpu, spacing = read_mhd(args.gpu)
    topas = read_topas(args.topas, gpu.shape)
    gpu_dose, dose_spacing = read_mhd(args.gpu_dose)
    if dose_spacing != spacing or gpu_dose.shape != gpu.shape:
        raise RuntimeError("GPU dose and LET grids differ")
    topas_dose = read_topas(args.topas_dose, gpu.shape)
    dx, dy, dz = spacing
    nz, ny, nx = gpu.shape
    x = -0.5 * nx * dx + (np.arange(nx) + 0.5) * dx
    y = -0.5 * ny * dy + (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = (
        (gpu_dose >= 0.01 * np.max(gpu_dose))
        | (topas_dose >= 0.01 * np.max(topas_dose))
    )

    # 1D: average a 12 x 12 mm central column for a stable depth profile, then
    # show unsmoothed lateral profiles at three SOBP depths.
    central_x = np.abs(x) <= 6.0
    central_y = np.abs(y) <= 6.0
    gpu_depth = gpu[:, central_y][:, :, central_x].mean(axis=(1, 2))
    topas_depth = topas[:, central_y][:, :, central_x].mean(axis=(1, 2))
    depths_mm = (60.0, 75.0, 90.0)

    figure, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(z, topas_depth, color="black", linewidth=2.0, label="TOPAS")
    axes[0].plot(z, gpu_depth, color="#d62728", linewidth=1.8, label="GPU")
    axes[0].axvspan(50.0, 100.0, color="0.8", alpha=0.25, label="SOBP")
    axes[0].set(
        xlabel="Depth z [mm]",
        ylabel=r"All-hadron LET$_d$ [MeV/mm/(g/cm$^3$)]",
        title=r"Central 12$\times$12 mm mean depth profile",
        xlim=(0.0, 150.0),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for depth, color in zip(depths_mm, colors):
        iz = nearest_index(z, depth)
        gpu_lateral = np.nanmean(
            np.where(valid[iz, central_y], gpu[iz, central_y], np.nan),
            axis=0,
        )
        topas_lateral = np.nanmean(
            np.where(valid[iz, central_y], topas[iz, central_y], np.nan),
            axis=0,
        )
        axes[1].plot(
            x,
            topas_lateral,
            linestyle="--",
            color=color,
            linewidth=1.8,
            label=f"TOPAS z={z[iz]:.1f} mm",
        )
        axes[1].plot(
            x,
            gpu_lateral,
            linestyle="-",
            color=color,
            linewidth=1.6,
            label=f"GPU z={z[iz]:.1f} mm",
        )
    axes[1].set(
        xlabel="Lateral x [mm]",
        ylabel=r"All-hadron LET$_d$ [MeV/mm/(g/cm$^3$)]",
        title="Central lateral profiles",
        xlim=(-30.0, 30.0),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=9)
    figure.suptitle(r"C-12 SOBP all-hadron LET$_d$: 1D profiles")
    figure.savefig(output_dir / "letd_1d_profiles.png", dpi=200)
    plt.close(figure)

    # 2D: XY at 75 mm and central XZ. Both TOPAS/GPU panels share one global
    # physical LET range; both difference panels share one symmetric range.
    iz = nearest_index(z, 75.0)
    iy = nearest_index(y, 0.0)
    topas_xy = np.where(valid[iz], topas[iz], np.nan)
    gpu_xy = np.where(valid[iz], gpu[iz], np.nan)
    topas_xz = np.where(valid[:, iy], topas[:, iy], np.nan)
    gpu_xz = np.where(valid[:, iy], gpu[:, iy], np.nan)
    physical_samples = np.concatenate(
        [
            topas_xy[np.isfinite(topas_xy)],
            gpu_xy[np.isfinite(gpu_xy)],
            topas_xz[np.isfinite(topas_xz)],
            gpu_xz[np.isfinite(gpu_xz)],
        ]
    )
    vmax = float(np.percentile(physical_samples[physical_samples > 0.0], 99.5))
    differences = np.concatenate(
        [(gpu_xy - topas_xy).ravel(), (gpu_xz - topas_xz).ravel()]
    )
    difference_limit = float(
        np.nanpercentile(np.abs(differences), 99.5)
    )
    physical_norm = Normalize(vmin=0.0, vmax=vmax)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )
    physical_cmap = plt.colormaps["magma"].copy()
    physical_cmap.set_bad("#e6e6e6")
    difference_cmap = plt.colormaps["coolwarm"].copy()
    difference_cmap.set_bad("#e6e6e6")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    xy_extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    xz_extent = (x[0] - dx / 2, x[-1] + dx / 2, z[0] - dz / 2, z[-1] + dz / 2)
    physical_images = []
    for axis, image, title, extent in (
        (axes[0, 0], topas_xy, f"TOPAS XY, z={z[iz]:.1f} mm", xy_extent),
        (axes[0, 1], gpu_xy, f"GPU XY, z={z[iz]:.1f} mm", xy_extent),
        (axes[1, 0], topas_xz, f"TOPAS XZ, y={y[iy]:.1f} mm", xz_extent),
        (axes[1, 1], gpu_xz, f"GPU XZ, y={y[iy]:.1f} mm", xz_extent),
    ):
        rendered = axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=physical_cmap,
            norm=physical_norm,
            aspect="equal",
        )
        physical_images.append(rendered)
        axis.set_title(title)

    difference_images = []
    for axis, image, title, extent in (
        (axes[0, 2], gpu_xy - topas_xy, "GPU - TOPAS", xy_extent),
        (axes[1, 2], gpu_xz - topas_xz, "GPU - TOPAS", xz_extent),
    ):
        rendered = axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=difference_cmap,
            norm=difference_norm,
            aspect="equal",
        )
        difference_images.append(rendered)
        axis.set_title(title)

    for axis in axes[0]:
        axis.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(-45.0, 45.0), ylim=(-45.0, 45.0))
    for axis in axes[1]:
        axis.set(xlabel="x [mm]", ylabel="Depth z [mm]", xlim=(-45.0, 45.0), ylim=(0.0, 150.0))
    figure.colorbar(
        physical_images[0],
        ax=axes[:, :2],
        label=r"LET$_d$ [MeV/mm/(g/cm$^3$)]",
        shrink=0.9,
    )
    figure.colorbar(
        difference_images[0],
        ax=axes[:, 2],
        label=r"GPU - TOPAS LET$_d$ [MeV/mm/(g/cm$^3$)]",
        shrink=0.9,
    )
    figure.suptitle(
        "C-12 SOBP all-hadron LET$_d$: matched 2D ranges\n"
        f"1% dose-union mask; TOPAS/GPU range 0–{vmax:.2f}; "
        f"difference ±{difference_limit:.2f}"
    )
    figure.savefig(output_dir / "letd_2d_profiles_common_range.png", dpi=200)
    plt.close(figure)

    print(output_dir / "letd_1d_profiles.png")
    print(output_dir / "letd_2d_profiles_common_range.png")
    print(f"physical_range=0,{vmax:.8g}")
    print(f"difference_range={-difference_limit:.8g},{difference_limit:.8g}")


if __name__ == "__main__":
    main()
