#!/usr/bin/env python3
"""Plot absolute 1D depth and transverse minibeam dose profiles."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEY_VALUE_RE = re.compile(r"^\s*([^#=]+?)\s*=\s*(.*?)\s*$")


def load_mhd(path: Path) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = KEY_VALUE_RE.match(line)
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    dims = tuple(int(value) for value in fields["DimSize"].split())
    spacing = tuple(float(value) for value in fields["ElementSpacing"].split())
    offset = tuple(float(value) for value in fields["Offset"].split())
    if len(dims) != 3 or len(spacing) != 3 or len(offset) != 3:
        raise ValueError(f"{path}: expected three-dimensional MHD geometry")
    element_type = fields.get("ElementType")
    if element_type != "MET_FLOAT":
        raise ValueError(f"{path}: expected MET_FLOAT, found {element_type}")
    raw_path = path.parent / fields["ElementDataFile"]
    # MHD DimSize is x,y,z; the raw project layout is z,y,x.
    dose = np.memmap(
        raw_path, dtype="<f4", mode="r", shape=(dims[2], dims[1], dims[0])
    )
    x = offset[0] + np.arange(dims[0], dtype=np.float64) * spacing[0]
    y = offset[1] + np.arange(dims[1], dtype=np.float64) * spacing[1]
    z = offset[2] + np.arange(dims[2], dtype=np.float64) * spacing[2]
    return dose, x, y, z


def relative_difference_percent(
    evaluation: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    threshold = 0.01 * float(np.max(reference))
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    selected = reference >= threshold
    result[selected] = (
        100.0 * (evaluation[selected] - reference[selected]) / reference[selected]
    )
    return result


def depth_profiles(
    reference: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.sum(reference, axis=(1, 2), dtype=np.float64),
        np.sum(evaluation, axis=(1, 2), dtype=np.float64),
    )

def distal_crossing(
    z: np.ndarray, dose: np.ndarray, fraction: float = 0.8
) -> float:
    peak = int(np.argmax(dose))
    target = fraction * float(dose[peak])
    for index in range(peak + 1, dose.size):
        if dose[index] <= target < dose[index - 1]:
            denominator = float(dose[index] - dose[index - 1])
            if denominator == 0.0:
                return float(z[index - 1])
            weight = (target - float(dose[index - 1])) / denominator
            return float(z[index - 1] + weight * (z[index] - z[index - 1]))
    return float("nan")


def transverse_profiles(
    dose: np.ndarray,
    z: np.ndarray,
    center_mm: float,
    slab_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    selected = np.flatnonzero(np.abs(z - center_mm) < 0.5 * slab_width_mm)
    if selected.size == 0:
        selected = np.asarray([int(np.argmin(np.abs(z - center_mm)))])
    slab = dose[int(selected[0]) : int(selected[-1]) + 1]
    profile_x = np.sum(slab, axis=(0, 1), dtype=np.float64)
    profile_y = np.sum(slab, axis=(0, 2), dtype=np.float64)
    return profile_x, profile_y, (float(z[selected[0]]), float(z[selected[-1]]))


def plot_depth(
    output: Path,
    z: np.ndarray,
    reference: np.ndarray,
    evaluation: np.ndarray,
    histories_label: str,
) -> None:
    difference = relative_difference_percent(evaluation, reference)
    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    axes[0].plot(z, reference, color="#1967B3", linewidth=1.8, label="TOPAS")
    axes[0].plot(
        z, evaluation, color="#D55E00", linewidth=1.5,
        linestyle="--", label="GPU",
    )
    reference_r80 = distal_crossing(z, reference)
    evaluation_r80 = distal_crossing(z, evaluation)
    axes[0].axvline(
        reference_r80, color="#1967B3", linewidth=0.9, alpha=0.7
    )
    axes[0].axvline(
        evaluation_r80, color="#D55E00", linewidth=0.9,
        linestyle="--", alpha=0.7,
    )
    axes[0].set_ylabel("Integrated voxel dose (Gy)")
    axes[0].set_title(
        f"Minibeam depth-dose profile — {histories_label} incident C-12\n"
        f"R80 TOPAS={reference_r80:.3f} mm, GPU={evaluation_r80:.3f} mm, "
        f"Δ={evaluation_r80-reference_r80:+.3f} mm"
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].plot(z, difference, color="#6A3D9A", linewidth=1.0)
    axes[1].set_ylim(-15.0, 15.0)
    axes[1].set_xlabel("Depth in water (mm)")
    axes[1].set_ylabel("GPU − TOPAS (%)")
    axes[1].grid(alpha=0.22)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transverse(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    reference: np.ndarray,
    evaluation: np.ndarray,
    depths_mm: list[float],
    slab_width_mm: float,
    histories_label: str,
    evaluation_scale: float = 1.0,
) -> None:
    fig, axes = plt.subplots(
        2, len(depths_mm), figsize=(5.0 * len(depths_mm), 8.0),
        squeeze=False,
    )
    x_profiles: list[tuple[np.ndarray, np.ndarray, tuple[float, float]]] = []
    y_profiles: list[tuple[np.ndarray, np.ndarray, tuple[float, float]]] = []
    for depth in depths_mm:
        ref_x, ref_y, span = transverse_profiles(
            reference, z, depth, slab_width_mm
        )
        gpu_x, gpu_y, _ = transverse_profiles(
            evaluation, z, depth, slab_width_mm
        )
        gpu_x *= evaluation_scale
        gpu_y *= evaluation_scale
        x_profiles.append((ref_x, gpu_x, span))
        y_profiles.append((ref_y, gpu_y, span))

    x_limit = 1.05 * max(float(np.max(p)) for pair in x_profiles for p in pair[:2])
    y_limit = 1.05 * max(float(np.max(p)) for pair in y_profiles for p in pair[:2])
    for column, depth in enumerate(depths_mm):
        for row, (coordinate, profiles, axis_name, limit) in enumerate(
            (
                (x, x_profiles[column], "X across slits", x_limit),
                (y, y_profiles[column], "Y along slits", y_limit),
            )
        ):
            reference_profile, evaluation_profile, span = profiles
            axis = axes[row, column]
            axis.plot(
                coordinate, reference_profile, color="#1967B3",
                linewidth=1.6, label="TOPAS",
            )
            axis.plot(
                coordinate, evaluation_profile, color="#D55E00",
                linewidth=1.4, linestyle="--", label="GPU",
            )
            axis.set_xlim(
                (-30.0, 30.0) if row == 0 else (float(y[0]), float(y[-1]))
            )
            axis.set_ylim(0.0, limit)
            axis.set_title(
                f"{axis_name}, depth {depth:g} mm\n"
                f"integrated over z={span[0]:.2f}–{span[1]:.2f} mm"
            )
            axis.set_xlabel(f"{'X' if row == 0 else 'Y'} position (mm)")
            axis.set_ylabel("Integrated voxel dose (Gy)")
            axis.grid(alpha=0.22)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle(
        f"Minibeam transverse dose profiles — {histories_label} incident C-12",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--depths-mm", type=float, nargs="+", default=[0.5, 35.0, 70.45]
    )
    parser.add_argument("--slab-width-mm", type=float, default=1.0)
    parser.add_argument("--histories-label", default="100k")
    parser.add_argument(
        "--evaluation-scale",
        type=float,
        default=1.0,
        help="Prescribed history-count scale applied to GPU profiles",
    )
    args = parser.parse_args()

    reference, x_ref, y_ref, z_ref = load_mhd(args.reference)
    evaluation, x_eval, y_eval, z_eval = load_mhd(args.evaluation)
    if reference.shape != evaluation.shape:
        raise ValueError("Reference and evaluation dose shapes differ")
    for name, first, second in (
        ("x", x_ref, x_eval), ("y", y_ref, y_eval), ("z", z_ref, z_eval)
    ):
        if not np.allclose(first, second, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Reference and evaluation {name} coordinates differ")

    ref_depth, gpu_depth = depth_profiles(reference, evaluation)
    gpu_depth *= args.evaluation_scale
    filename_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.histories_label)
    depth_output = args.output_dir / f"depth_dose_1d_{filename_label}.png"
    transverse_output = (
        args.output_dir / f"transverse_dose_1d_{filename_label}.png"
    )
    plot_depth(
        depth_output, z_ref, ref_depth, gpu_depth, args.histories_label
    )
    plot_transverse(
        transverse_output,
        x_ref, y_ref, z_ref, reference, evaluation,
        args.depths_mm, args.slab_width_mm, args.histories_label,
        args.evaluation_scale,
    )
    print(depth_output)
    print(transverse_output)


if __name__ == "__main__":
    main()
