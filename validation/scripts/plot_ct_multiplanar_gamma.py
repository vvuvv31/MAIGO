#!/usr/bin/env python3
"""Plot scaled GPU/reference dose, difference, and planar gamma on CT slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import read_mhd


def load_volume(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, values = read_mhd(path)
    nx, ny, nz = (int(value) for value in metadata["DimSize"].split())
    return metadata, np.asarray(values, dtype=np.float64).reshape(nz, ny, nx)


def slice_indices(mask: np.ndarray, axis: int) -> list[int]:
    other_axes = tuple(index for index in range(3) if index != axis)
    occupied = np.flatnonzero(np.any(mask, axis=other_axes))
    if occupied.size == 0:
        raise ValueError("reference high-dose mask is empty")
    low, high = int(occupied[0]), int(occupied[-1])
    return [
        int(round(low + fraction * (high - low)))
        for fraction in (0.2, 0.5, 0.8)
    ]


def bilinear_sample(
    values: np.ndarray, row: np.ndarray, column: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = values.shape
    valid = (
        (row >= 0.0)
        & (row <= rows - 1)
        & (column >= 0.0)
        & (column <= columns - 1)
    )
    sampled = np.zeros(row.shape, dtype=np.float64)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return sampled, valid
    r0 = np.floor(row[ids]).astype(np.int32)
    c0 = np.floor(column[ids]).astype(np.int32)
    r1 = np.minimum(r0 + 1, rows - 1)
    c1 = np.minimum(c0 + 1, columns - 1)
    fr = row[ids] - r0
    fc = column[ids] - c0
    sampled[ids] = (
        (1.0 - fr) * (1.0 - fc) * values[r0, c0]
        + (1.0 - fr) * fc * values[r0, c1]
        + fr * (1.0 - fc) * values[r1, c0]
        + fr * fc * values[r1, c1]
    )
    return sampled, valid


def gamma_2d_map(
    reference: np.ndarray,
    evaluated: np.ndarray,
    spacing_column_mm: float,
    spacing_row_mm: float,
    reference_peak: float,
    dose_percent: float,
    distance_mm: float,
    threshold_percent: float,
    interpolation_step_mm: float,
    local_dose: bool,
) -> tuple[np.ndarray, float, int]:
    threshold = threshold_percent / 100.0 * reference_peak
    selected_mask = reference >= threshold
    selected = np.flatnonzero(selected_mask)
    gamma_map = np.full(reference.shape, np.nan, dtype=np.float64)
    if selected.size == 0:
        return gamma_map, float("nan"), 0

    rows, columns = reference.shape
    row = selected // columns
    column = selected % columns
    reference_values = reference.reshape(-1)[selected]
    if local_dose:
        dose_criterion = np.maximum(
            dose_percent / 100.0 * reference_values, 1.0e-30
        )
    else:
        dose_criterion = np.full(
            reference_values.shape,
            dose_percent / 100.0 * reference_peak,
            dtype=np.float64,
        )
    best_squared = np.full(selected.size, np.inf, dtype=np.float64)
    offsets = np.arange(
        -distance_mm,
        distance_mm + 0.25 * interpolation_step_mm,
        interpolation_step_mm,
    )
    for row_offset_mm in offsets:
        for column_offset_mm in offsets:
            distance_squared = (
                row_offset_mm * row_offset_mm
                + column_offset_mm * column_offset_mm
            )
            if distance_squared > distance_mm * distance_mm + 1.0e-9:
                continue
            query_row = row + row_offset_mm / spacing_row_mm
            query_column = column + column_offset_mm / spacing_column_mm
            candidate, valid = bilinear_sample(
                evaluated, query_row, query_column
            )
            ids = np.flatnonzero(valid)
            if ids.size == 0:
                continue
            gamma_squared = distance_squared / (distance_mm * distance_mm) + (
                (candidate[ids] - reference_values[ids]) / dose_criterion[ids]
            ) ** 2
            best_squared[ids] = np.minimum(best_squared[ids], gamma_squared)

    gamma_values = np.sqrt(best_squared)
    gamma_map.reshape(-1)[selected] = gamma_values
    pass_percent = 100.0 * np.count_nonzero(gamma_values <= 1.0) / selected.size
    return gamma_map, float(pass_percent), int(selected.size)


def plane_slice(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="TOPAS/Dij reference")
    parser.add_argument("--gpu-label", default="Scaled GPU (10M)")
    parser.add_argument("--dose-percent", type=float, default=2.0)
    parser.add_argument("--distance-mm", type=float, default=2.0)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    parser.add_argument("--difference-range-percent", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_metadata, reference = load_volume(args.reference)
    gpu_metadata, gpu = load_volume(args.gpu)
    if reference.shape != gpu.shape:
        raise SystemExit(
            f"dose shapes differ: reference={reference.shape}, gpu={gpu.shape}"
        )
    for key in ("DimSize", "ElementSpacing", "Offset"):
        if reference_metadata[key] != gpu_metadata[key]:
            raise SystemExit(
                f"dose metadata differs for {key}: "
                f"{reference_metadata[key]} != {gpu_metadata[key]}"
            )

    sx, sy, sz = (
        float(value) for value in reference_metadata["ElementSpacing"].split()
    )
    ox, oy, oz = (
        float(value) for value in reference_metadata["Offset"].split()
    )
    nz, ny, nx = reference.shape
    peak = float(reference.max())
    high_dose = reference >= args.threshold_percent / 100.0 * peak
    plane_specs = [
        {
            "name": "axial",
            "label": "Axial",
            "axis": 0,
            "indices": slice_indices(high_dose, 0),
            "extent": (ox, ox + sx * (nx - 1), oy, oy + sy * (ny - 1)),
            "spacing_column": sx,
            "spacing_row": sy,
            "xlabel": "X (mm)",
            "ylabel": "Y (mm)",
            "coordinate_name": "Z",
            "coordinate_origin": oz,
            "coordinate_spacing": sz,
        },
        {
            "name": "coronal",
            "label": "Coronal",
            "axis": 1,
            "indices": slice_indices(high_dose, 1),
            "extent": (ox, ox + sx * (nx - 1), oz, oz + sz * (nz - 1)),
            "spacing_column": sx,
            "spacing_row": sz,
            "xlabel": "X (mm)",
            "ylabel": "Z (mm)",
            "coordinate_name": "Y",
            "coordinate_origin": oy,
            "coordinate_spacing": sy,
        },
        {
            "name": "sagittal",
            "label": "Sagittal",
            "axis": 2,
            "indices": slice_indices(high_dose, 2),
            "extent": (oy, oy + sy * (ny - 1), oz, oz + sz * (nz - 1)),
            "spacing_column": sy,
            "spacing_row": sz,
            "xlabel": "Y (mm)",
            "ylabel": "Z (mm)",
            "coordinate_name": "X",
            "coordinate_origin": ox,
            "coordinate_spacing": sx,
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "reference": args.reference.as_posix(),
        "gpu": args.gpu.as_posix(),
        "criterion": {
            "dose_percent": args.dose_percent,
            "distance_mm": args.distance_mm,
            "threshold_percent_of_reference_peak": args.threshold_percent,
            "interpolation_step_mm": args.gamma_resolution_mm,
            "gamma_dimension": "2D in-plane",
        },
        "planes": {},
    }
    dose_cmap = "inferno"
    difference_cmap = "RdBu_r"
    gamma_cmap = plt.get_cmap("viridis").copy()
    gamma_cmap.set_bad("#d9d9d9")

    for specification in plane_specs:
        rows = len(specification["indices"])
        figure, axes = plt.subplots(
            rows,
            5,
            figsize=(23, 4.4 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        plane_metrics: list[dict[str, object]] = []
        dose_image = difference_image = gamma_image = None
        for row_index, slice_index in enumerate(specification["indices"]):
            reference_slice = plane_slice(
                reference, specification["axis"], slice_index
            )
            gpu_slice = plane_slice(gpu, specification["axis"], slice_index)
            difference_percent = 100.0 * (gpu_slice - reference_slice) / peak
            global_gamma, global_pass, points = gamma_2d_map(
                reference_slice,
                gpu_slice,
                specification["spacing_column"],
                specification["spacing_row"],
                peak,
                args.dose_percent,
                args.distance_mm,
                args.threshold_percent,
                args.gamma_resolution_mm,
                local_dose=False,
            )
            local_gamma, local_pass, _ = gamma_2d_map(
                reference_slice,
                gpu_slice,
                specification["spacing_column"],
                specification["spacing_row"],
                peak,
                args.dose_percent,
                args.distance_mm,
                args.threshold_percent,
                args.gamma_resolution_mm,
                local_dose=True,
            )
            coordinate = (
                specification["coordinate_origin"]
                + slice_index * specification["coordinate_spacing"]
            )
            plane_metrics.append(
                {
                    "slice_index": int(slice_index),
                    "coordinate_mm": float(coordinate),
                    "threshold_points": points,
                    "global_gamma_pass_percent": global_pass,
                    "local_gamma_pass_percent": local_pass,
                }
            )

            for column, dose_slice in enumerate(
                (reference_slice, gpu_slice)
            ):
                dose_image = axes[row_index, column].imshow(
                    dose_slice,
                    origin="lower",
                    extent=specification["extent"],
                    cmap=dose_cmap,
                    vmin=0.0,
                    vmax=peak,
                    interpolation="nearest",
                    aspect="equal",
                )
                axes[row_index, column].contour(
                    dose_slice,
                    levels=[0.1 * peak, 0.5 * peak, 0.9 * peak],
                    colors=["white", "cyan", "lime"],
                    linewidths=0.55,
                    origin="lower",
                    extent=specification["extent"],
                )
            difference_image = axes[row_index, 2].imshow(
                difference_percent,
                origin="lower",
                extent=specification["extent"],
                cmap=difference_cmap,
                vmin=-args.difference_range_percent,
                vmax=args.difference_range_percent,
                interpolation="nearest",
                aspect="equal",
            )
            for column, (gamma_map, pass_percent, title) in enumerate(
                (
                    (global_gamma, global_pass, "Global"),
                    (local_gamma, local_pass, "Local"),
                ),
                start=3,
            ):
                gamma_image = axes[row_index, column].imshow(
                    gamma_map,
                    origin="lower",
                    extent=specification["extent"],
                    cmap=gamma_cmap,
                    vmin=0.0,
                    vmax=2.0,
                    interpolation="nearest",
                    aspect="equal",
                )
                axes[row_index, column].contour(
                    gamma_map,
                    levels=[1.0],
                    colors=["red"],
                    linewidths=0.7,
                    origin="lower",
                    extent=specification["extent"],
                )
                axes[row_index, column].text(
                    0.02,
                    0.98,
                    f"{title} pass: {pass_percent:.2f}%\n(n={points})",
                    transform=axes[row_index, column].transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 3,
                    },
                )

            axes[row_index, 0].set_ylabel(
                f"{specification['coordinate_name']}={coordinate:.1f} mm\n"
                f"{specification['ylabel']}"
            )
            for panel in axes[row_index, :]:
                panel.set_xlabel(specification["xlabel"])
                panel.tick_params(labelsize=8)

        column_titles = [
            args.reference_label,
            args.gpu_label,
            "GPU − reference\n(% of reference maximum)",
            (
                f"Global gamma\n{args.dose_percent:g}%/"
                f"{args.distance_mm:g} mm"
            ),
            (
                f"Local gamma\n{args.dose_percent:g}%/"
                f"{args.distance_mm:g} mm"
            ),
        ]
        for column, title in enumerate(column_titles):
            axes[0, column].set_title(title, fontsize=12)
        assert dose_image is not None
        assert difference_image is not None
        assert gamma_image is not None
        figure.colorbar(
            dose_image,
            ax=axes[:, :2],
            location="right",
            shrink=0.78,
            pad=0.015,
            label="Dose (Gy)",
        )
        figure.colorbar(
            difference_image,
            ax=axes[:, 2],
            location="right",
            shrink=0.78,
            pad=0.02,
            label="Dose difference (% of reference maximum)",
        )
        gamma_colorbar = figure.colorbar(
            gamma_image,
            ax=axes[:, 3:],
            location="right",
            shrink=0.78,
            pad=0.015,
            label="Gamma index",
        )
        gamma_colorbar.ax.axhline(1.0, color="red", linewidth=1.0)
        figure.suptitle(
            f"{specification['label']} dose and planar gamma comparison "
            f"(threshold {args.threshold_percent:g}% of reference peak)",
            fontsize=15,
        )
        output_path = args.output_dir / f"{specification['name']}_comparison.png"
        figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        metrics["planes"][specification["name"]] = plane_metrics
        print(f"Wrote {output_path}")

    metrics_path = args.output_dir / "planar_gamma_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
