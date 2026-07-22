#!/usr/bin/env python3
"""Plot multiple axial, coronal, and sagittal CT-plan dose slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import read_mhd


def load_volume(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, values = read_mhd(path)
    nx, ny, nz = (int(value) for value in metadata["DimSize"].split())
    return metadata, np.asarray(values, dtype=np.float64).reshape(nz, ny, nx)


def slice_indices(mask: np.ndarray, axis: int) -> list[int]:
    occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
    if occupied.size == 0:
        raise ValueError("reference high-dose mask is empty")
    low, high = int(occupied[0]), int(occupied[-1])
    return [int(round(low + fraction * (high - low))) for fraction in (0.2, 0.5, 0.8)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Original GPU")
    parser.add_argument("--calibrated-label", default="Calibrated GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--difference-range-percent", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    # Import lazily so numerical validation does not require matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata, reference = load_volume(args.reference)
    baseline_metadata, baseline = load_volume(args.baseline)
    calibrated_metadata, calibrated = load_volume(args.calibrated)
    shape = reference.shape
    if baseline.shape != shape or calibrated.shape != shape:
        raise SystemExit(
            f"dose shapes differ: reference={shape}, baseline={baseline.shape}, "
            f"calibrated={calibrated.shape}"
        )
    if baseline_metadata["ElementSpacing"] != metadata["ElementSpacing"]:
        raise SystemExit("baseline/reference spacing differs")
    if calibrated_metadata["ElementSpacing"] != metadata["ElementSpacing"]:
        raise SystemExit("calibrated/reference spacing differs")

    sx, sy, sz = (float(value) for value in metadata["ElementSpacing"].split())
    ox, oy, oz = (float(value) for value in metadata["Offset"].split())
    nz, ny, nx = shape
    peak = float(reference.max())
    high_dose = reference >= args.threshold_percent / 100.0 * peak

    # Array axes are [Z, Y, X].  Three levels are selected within the high-dose
    # bounding box for each anatomical plane.
    plane_specs = [
        ("Axial", 0, slice_indices(high_dose, 0), "X (mm)", "Y (mm)"),
        ("Coronal", 1, slice_indices(high_dose, 1), "X (mm)", "Z (mm)"),
        ("Sagittal", 2, slice_indices(high_dose, 2), "Y (mm)", "Z (mm)"),
    ]

    rows = sum(len(spec[2]) for spec in plane_specs)
    fig, axes = plt.subplots(rows, 4, figsize=(18, 3.25 * rows), constrained_layout=True)
    dose_cmap = "inferno"
    diff_cmap = "RdBu_r"
    diff_limit = args.difference_range_percent
    dose_image = diff_image = None
    row = 0
    for plane_name, axis, indices, xlabel, ylabel in plane_specs:
        for level, index in enumerate(indices):
            if axis == 0:
                slices = [volume[index, :, :] for volume in (reference, baseline, calibrated)]
                extent = (ox, ox + sx * (nx - 1), oy, oy + sy * (ny - 1))
                coordinate = oz + index * sz
                coordinate_name = "Z"
            elif axis == 1:
                slices = [volume[:, index, :] for volume in (reference, baseline, calibrated)]
                extent = (ox, ox + sx * (nx - 1), oz, oz + sz * (nz - 1))
                coordinate = oy + index * sy
                coordinate_name = "Y"
            else:
                slices = [volume[:, :, index] for volume in (reference, baseline, calibrated)]
                extent = (oy, oy + sy * (ny - 1), oz, oz + sz * (nz - 1))
                coordinate = ox + index * sx
                coordinate_name = "X"

            difference = 100.0 * (slices[2] - slices[0]) / peak
            for column, dose_slice in enumerate(slices):
                dose_image = axes[row, column].imshow(
                    dose_slice,
                    origin="lower",
                    extent=extent,
                    cmap=dose_cmap,
                    vmin=0.0,
                    vmax=peak,
                    interpolation="nearest",
                    aspect="equal",
                )
                axes[row, column].contour(
                    dose_slice,
                    levels=[0.1 * peak, 0.5 * peak, 0.9 * peak],
                    colors=["white", "cyan", "lime"],
                    linewidths=0.45,
                    origin="lower",
                    extent=extent,
                )
            diff_image = axes[row, 3].imshow(
                difference,
                origin="lower",
                extent=extent,
                cmap=diff_cmap,
                vmin=-diff_limit,
                vmax=diff_limit,
                interpolation="nearest",
                aspect="equal",
            )
            axes[row, 0].set_ylabel(
                f"{plane_name} {level + 1}/3\n{coordinate_name}={coordinate:.2f} mm\n{ylabel}"
            )
            for panel in axes[row, :]:
                panel.set_xlabel(xlabel)
                panel.tick_params(labelsize=7)
            row += 1

    titles = [
        "Reference physical dose",
        args.baseline_label.replace("\\n", "\n"),
        args.calibrated_label.replace("\\n", "\n"),
        "Calibrated − reference\n(% of reference maximum)",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12)
    assert dose_image is not None and diff_image is not None
    fig.colorbar(dose_image, ax=axes[:, :3], location="bottom", shrink=0.55, label="Dose (Gy)")
    fig.colorbar(
        diff_image,
        ax=axes[:, 3],
        location="bottom",
        shrink=0.8,
        label="Dose difference (% of reference maximum)",
    )
    fig.suptitle(
        "CT full-plan dose comparison across anatomical planes and slice levels",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
