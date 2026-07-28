#!/usr/bin/env python3
"""Plot aligned TOPAS/GPU minibeam transverse planes with common dose ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_minibeam_dose import read_mhd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-scale", type=float, default=1.0)
    parser.add_argument(
        "--depths-mm", type=float, nargs="+", default=[0.5, 35.0, 70.25]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_mhd(args.reference)
    evaluation = read_mhd(args.evaluation)
    for field in ("shape_xyz", "spacing_xyz_mm", "offset_xyz_mm"):
        if getattr(reference, field) != getattr(evaluation, field):
            raise ValueError(f"Grid mismatch for {field}")

    spacing_x, spacing_y, spacing_z = reference.spacing_xyz_mm
    offset_x, offset_y, offset_z = reference.offset_xyz_mm
    nx, ny, _ = reference.shape_xyz
    extent = (
        offset_x - 0.5 * spacing_x,
        offset_x + (nx - 0.5) * spacing_x,
        offset_y - 0.5 * spacing_y,
        offset_y + (ny - 0.5) * spacing_y,
    )
    figure, axes = plt.subplots(
        3,
        len(args.depths_mm),
        figsize=(5.0 * len(args.depths_mm), 11.0),
        constrained_layout=True,
        squeeze=False,
    )
    for column, requested_depth in enumerate(args.depths_mm):
        iz = int(round((requested_depth - offset_z) / spacing_z))
        iz = max(0, min(reference.data_zyx.shape[0] - 1, iz))
        depth = offset_z + iz * spacing_z
        ref = np.asarray(reference.data_zyx[iz], dtype=np.float64)
        eva = (
            np.asarray(evaluation.data_zyx[iz], dtype=np.float64)
            * args.evaluation_scale
        )
        maximum = max(float(np.max(ref)), float(np.max(eva)))
        difference = eva - ref
        difference_limit = float(np.max(np.abs(difference)))
        dose_images = []
        for row, (values, label) in enumerate(((ref, "TOPAS"), (eva, "GPU"))):
            image = axes[row, column].imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="turbo",
                vmin=0.0,
                vmax=maximum,
                interpolation="nearest",
            )
            dose_images.append(image)
            axes[row, column].set_title(f"{label}, depth {depth:.2f} mm")
        diff_image = axes[2, column].imshow(
            difference,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-difference_limit,
            vmax=difference_limit,
            interpolation="nearest",
        )
        axes[2, column].set_title(f"GPU − TOPAS, depth {depth:.2f} mm")
        figure.colorbar(dose_images[0], ax=axes[:2, column], label="Dose (Gy)")
        figure.colorbar(diff_image, ax=axes[2, column], label="Dose difference (Gy)")
        for row in range(3):
            axes[row, column].set_xlabel("X across slits (mm)")
            axes[row, column].set_ylabel("Y along slits (mm)")

    figure.suptitle(
        "Minibeam transverse dose comparison; GPU scaled by prescribed "
        f"history ratio {args.evaluation_scale:.7g}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
