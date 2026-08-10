#!/usr/bin/env python3
"""Compare 3 mm isotropic GPU and TOPAS SOBP LET_d voxel maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_topas(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    data = np.loadtxt(path, comments="#", delimiter=",")
    volume = np.zeros(shape, dtype=np.float64)
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    z = data[:, 2].astype(int)
    volume[z, y, x] = data[:, 3]
    return volume


def read_mhd(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    nx, ny, nz = map(int, fields["DimSize"].split())
    spacing = tuple(map(float, fields["ElementSpacing"].split()))
    raw = np.fromfile(path.parent / fields["ElementDataFile"], dtype="<f4")
    return raw.reshape(nz, ny, nx).astype(np.float64), spacing


def metrics(gpu: np.ndarray, topas: np.ndarray, mask: np.ndarray) -> dict:
    difference = gpu[mask] - topas[mask]
    relative = 100.0 * difference / np.maximum(np.abs(topas[mask]), 1.0e-12)
    return {
        "voxels": int(mask.sum()),
        "mean_bias": float(np.mean(difference)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_relative_percent": float(np.mean(relative)),
        "median_absolute_relative_percent": float(np.median(np.abs(relative))),
        "p95_absolute_relative_percent": float(np.percentile(np.abs(relative), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-prefix",
        type=Path,
        default=Path("out/letd_sobp/3d/gpu_sobp_letd_3d"),
    )
    parser.add_argument(
        "--topas-primary",
        type=Path,
        default=Path(
            "validation/topas/output/sobp_letd_3d_100k_primary_c12.csv"
        ),
    )
    parser.add_argument(
        "--topas-all",
        type=Path,
        default=Path(
            "validation/topas/output/sobp_letd_3d_100k_all_hadron.csv"
        ),
    )
    parser.add_argument(
        "--topas-dose",
        type=Path,
        default=Path(
            "validation/topas/output/sobp_water_3cm_5_10cm_3mm_dose3d.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/letd_sobp/3d/comparison")
    )
    args = parser.parse_args()

    primary_mhd = args.gpu_prefix.with_name(
        args.gpu_prefix.name + "_primary_c12.mhd"
    )
    all_mhd = args.gpu_prefix.with_name(
        args.gpu_prefix.name + "_all_hadron.mhd"
    )
    gpu_primary, spacing = read_mhd(primary_mhd)
    gpu_all, spacing_all = read_mhd(all_mhd)
    if spacing != spacing_all:
        raise RuntimeError("GPU LET maps have different spacing")
    shape = gpu_primary.shape
    topas_primary = read_topas(args.topas_primary, shape)
    topas_all = read_topas(args.topas_all, shape)
    topas_dose = read_topas(args.topas_dose, shape)
    x = -0.5 * shape[2] * spacing[0] + (np.arange(shape[2]) + 0.5) * spacing[0]
    y = -0.5 * shape[1] * spacing[1] + (np.arange(shape[1]) + 0.5) * spacing[1]
    z_mm = (np.arange(shape[0]) + 0.5) * spacing[2]
    field_xy = (np.abs(y[:, None]) <= 15.0) & (np.abs(x[None, :]) <= 15.0)
    field_mask = np.broadcast_to(field_xy, shape)
    sobp_mask = field_mask & (z_mm[:, None, None] >= 50.0) & (
        z_mm[:, None, None] <= 100.0
    )
    # The parent SOBP file deliberately disables the TOPAS 3D dose scorer for
    # the first LET check, so its all-zero placeholder cannot define a mask.
    dose_available = bool(np.max(topas_dose) > 0.0)
    dose_mask = (
        topas_dose >= 0.01 * np.max(topas_dose)
        if dose_available
        else field_mask
    )

    result = {
        "definition": "dose-weighted electronic LET, MeV/mm/(g/cm3)",
        "shape_zyx": list(shape),
        "spacing_xyz_mm": list(spacing),
        "dose_mask_available": dose_available,
        "irradiated_field_all_depths": {
            "mask": "|x|,|y| <= 15 mm",
            "primary_c12": metrics(
                gpu_primary, topas_primary, field_mask & (topas_primary > 0.0)
            ),
            "all_hadron": metrics(
                gpu_all, topas_all, field_mask & (topas_all > 0.0)
            ),
        },
        "sobp_high_dose_volume": {
            "mask": "|x|,|y| <= 15 mm and 50 <= z <= 100 mm",
            "primary_c12": metrics(
                gpu_primary, topas_primary, sobp_mask & (topas_primary > 0.0)
            ),
            "all_hadron": metrics(
                gpu_all, topas_all, sobp_mask & (topas_all > 0.0)
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    z = int(round(75.0 / spacing[2] - 0.5))
    y = shape[1] // 2
    panels = [
        (topas_all[z], "TOPAS all-hadron, z=75 mm"),
        (gpu_all[z], "GPU all-hadron, z=75 mm"),
        (gpu_all[z] - topas_all[z], "GPU - TOPAS"),
        (topas_all[:, y, :], "TOPAS all-hadron, central XZ"),
        (gpu_all[:, y, :], "GPU all-hadron, central XZ"),
        (gpu_all[:, y, :] - topas_all[:, y, :], "GPU - TOPAS"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, (axis, (image, title)) in enumerate(zip(axes.flat, panels)):
        is_difference = index % 3 == 2
        if is_difference:
            limit = np.percentile(
                np.abs(
                    image[
                        field_mask[z]
                        if index == 2
                        else field_mask[:, y, :]
                    ]
                ),
                99,
            )
            rendered = axis.imshow(
                image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
            )
        else:
            rendered = axis.imshow(image, origin="lower", cmap="magma")
        axis.set_title(title)
        figure.colorbar(rendered, ax=axis, label=r"LET$_d$ [MeV/mm/(g/cm$^3$)]")
    figure.suptitle("3D C-12 SOBP all-hadron LET$_d$: GPU vs TOPAS")
    figure.savefig(args.output_dir / "gpu_vs_topas_sobp_letd_3d.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
