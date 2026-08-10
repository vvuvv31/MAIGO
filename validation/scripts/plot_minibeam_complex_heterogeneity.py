#!/usr/bin/env python3
"""Plot and quantify matched complex-heterogeneity minibeam MHD doses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from compare_minibeam_dose import read_mhd
from compare_minibeam_heterogeneous import distal_r80, profile_metrics, smooth
from compare_minibeam_planar_gamma import gamma_line


DEFAULT_DEPTHS = {
    "lateral_multimaterial": [10.0, 30.0, 60.0, 105.0],
    "longitudinal_multimaterial": [10.0, 28.0, 58.0, 86.0],
    "combined_multimaterial": [10.0, 24.0, 44.0, 60.0, 78.0],
}
MATERIAL_ORDER = ("water", "lung", "bone")
MATERIAL_COLORS = ("#4c78a8", "#72b7b2", "#b279a2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--depths-mm", type=float, nargs="+")
    parser.add_argument("--depth-smoothing-mm", type=float, default=1.5)
    parser.add_argument("--lateral-slab-mm", type=float, default=5.0)
    args = parser.parse_args()

    reference = read_mhd(args.reference)
    evaluation = read_mhd(args.evaluation)
    for field in ("shape_xyz", "spacing_xyz_mm", "offset_xyz_mm"):
        if getattr(reference, field) != getattr(evaluation, field):
            raise ValueError(f"grid mismatch for {field}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    case_name = metadata["case"]
    requested_depths = args.depths_mm or DEFAULT_DEPTHS[case_name]

    sx, _, sz = reference.spacing_xyz_mm
    ox, _, oz = reference.offset_xyz_mm
    nx, _, nz = reference.shape_xyz
    x = ox + np.arange(nx, dtype=np.float64) * sx
    depth = oz + np.arange(nz, dtype=np.float64) * sz
    topas_depth = np.sum(reference.data_zyx, axis=(1, 2), dtype=np.float64)
    gpu_depth = np.sum(evaluation.data_zyx, axis=(1, 2), dtype=np.float64)
    smoothing_bins = max(1, int(round(args.depth_smoothing_mm / sz)))
    topas_depth_s = smooth(topas_depth, smoothing_bins)
    gpu_depth_s = smooth(gpu_depth, smoothing_bins)

    material_map = np.zeros((nz, nx), dtype=np.uint8)
    for region in metadata["regions"]:
        x_mask = (x >= region["x_min_mm"]) & (x < region["x_max_mm"])
        z_mask = (
            (depth >= region["depth_min_mm"])
            & (depth < region["depth_max_mm"])
        )
        material_map[np.ix_(z_mask, x_mask)] = MATERIAL_ORDER.index(
            region["material"]
        )

    depth_result = profile_metrics(topas_depth_s, gpu_depth_s)
    depth_result.update(
        {
            "topas_R80_mm": distal_r80(depth, topas_depth_s),
            "gpu_R80_mm": distal_r80(depth, gpu_depth_s),
            "global_gamma_3pct_1mm_thr10": gamma_line(
                topas_depth_s, gpu_depth_s, sz, 0.10, 3.0, 1.0, False
            )["pass_percent"],
            "local_gamma_3pct_1mm_thr10": gamma_line(
                topas_depth_s, gpu_depth_s, sz, 0.10, 3.0, 1.0, True
            )["pass_percent"],
        }
    )
    depth_result["delta_R80_mm"] = (
        depth_result["gpu_R80_mm"] - depth_result["topas_R80_mm"]
    )

    lateral_profiles = []
    for requested_depth in requested_depths:
        selected = (
            np.abs(depth - requested_depth) <= 0.5 * args.lateral_slab_mm
        )
        if not np.any(selected):
            raise ValueError(f"depth outside scorer: {requested_depth}")
        topas = np.sum(
            reference.data_zyx[selected, 0, :], axis=0, dtype=np.float64
        )
        gpu = np.sum(
            evaluation.data_zyx[selected, 0, :], axis=0, dtype=np.float64
        )
        metrics = profile_metrics(topas, gpu)
        metrics.update(
            {
                "depth_mm": requested_depth,
                "global_gamma_3pct_0p4mm_thr5": gamma_line(
                    topas, gpu, sx, 0.05, 3.0, 0.4, False
                )["pass_percent"],
                "local_gamma_3pct_0p4mm_thr5": gamma_line(
                    topas, gpu, sx, 0.05, 3.0, 0.4, True
                )["pass_percent"],
            }
        )
        lateral_profiles.append((requested_depth, topas, gpu, metrics))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{args.output_prefix}_metrics.json")
    result = {
        "case": case_name,
        "normalization": "absolute DoseToMedium; no fitted scale",
        "depth": depth_result,
        "lateral": [item[3] for item in lateral_profiles],
    }
    metrics_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.0),
        sharex=False,
        gridspec_kw={"height_ratios": [1.25, 2.5, 1.0]},
        constrained_layout=True,
    )
    axes[0].imshow(
        material_map,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            x[0] - 0.5 * sx,
            x[-1] + 0.5 * sx,
            depth[0] - 0.5 * sz,
            depth[-1] + 0.5 * sz,
        ],
        cmap=ListedColormap(MATERIAL_COLORS),
        vmin=0,
        vmax=len(MATERIAL_ORDER) - 1,
    )
    axes[0].set_xlim(-20.0, 20.0)
    axes[0].set_ylabel("Depth (mm)")
    axes[0].set_xlabel("Position across slits (mm)")
    axes[0].set_title(f"{case_name}: material map")
    axes[0].legend(
        handles=[
            Patch(color=color, label=name)
            for name, color in zip(
                MATERIAL_ORDER, MATERIAL_COLORS, strict=True
            )
        ],
        ncols=3,
        loc="upper right",
    )

    axes[1].plot(depth, topas_depth_s, label="TOPAS", linewidth=1.8)
    axes[1].plot(depth, gpu_depth_s, label="GPU", linewidth=1.5)
    axes[1].set_ylabel("Integrated transverse dose (Gy)")
    axes[1].set_title(
        "Depth profile; "
        f"global/local γ 3%/1mm = "
        f"{depth_result['global_gamma_3pct_1mm_thr10']:.1f}%/"
        f"{depth_result['local_gamma_3pct_1mm_thr10']:.1f}%"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    ratio = np.divide(
        gpu_depth_s,
        topas_depth_s,
        out=np.full_like(gpu_depth_s, np.nan),
        where=topas_depth_s > 0.02 * np.max(topas_depth_s),
    )
    axes[2].plot(depth, ratio, color="tab:purple", linewidth=1.2)
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].axhspan(0.97, 1.03, color="tab:green", alpha=0.15)
    axes[2].set_ylim(0.6, 1.4)
    axes[2].set_xlabel("Depth (mm)")
    axes[2].set_ylabel("GPU / TOPAS")
    axes[2].grid(alpha=0.2)
    depth_path = Path(f"{args.output_prefix}_depth.png")
    figure.savefig(depth_path, dpi=180)
    plt.close(figure)

    columns = 2
    rows = int(np.ceil(len(lateral_profiles) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 3.6 * rows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (requested_depth, topas, gpu, metrics) in zip(
        axes.flat, lateral_profiles, strict=False
    ):
        axis.plot(x, topas, label="TOPAS", linewidth=1.5)
        axis.plot(x, gpu, label="GPU", linewidth=1.3)
        axis.set_xlim(-20.0, 20.0)
        axis.set_title(
            f"{requested_depth:g} mm; global γ 3%/0.4mm "
            f"{metrics['global_gamma_3pct_0p4mm_thr5']:.1f}%"
        )
        axis.set_ylabel(f"Dose in {args.lateral_slab_mm:g} mm slab (Gy)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(lateral_profiles) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Position across slits (mm)")
    axes[0, 0].legend()
    figure.suptitle(f"{case_name}: lateral profiles")
    lateral_path = Path(f"{args.output_prefix}_lateral.png")
    figure.savefig(lateral_path, dpi=180)
    plt.close(figure)

    print(metrics_path)
    print(depth_path)
    print(lateral_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
