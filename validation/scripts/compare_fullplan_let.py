#!/usr/bin/env python3
"""Compare patient-axis TOPAS and beam-axis GPU full-plan LET_d maps."""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from match_gpu_to_physical_dose import (
    dose_only_pass_rate,
    gamma_3d,
    map_gpu_to_physical,
    read_mhd,
)


def mapped_gpu(
    path: Path,
    reference_shape: tuple[int, int, int],
    flip_x: bool,
    mapping: str,
) -> np.ndarray:
    metadata, values = read_mhd(path)
    gpu_shape = tuple(int(v) for v in metadata["DimSize"].split())
    mapped = map_gpu_to_physical(
        values,
        gpu_shape,
        reference_shape,
        reference_shape,
        flip_x,
        False,
        mapping,
    )
    nx, ny, nz = reference_shape
    return np.asarray(mapped, dtype=np.float64).reshape(nz, ny, nx)


def native(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    metadata, values = read_mhd(path)
    nx, ny, nz = (int(v) for v in metadata["DimSize"].split())
    spacing = tuple(float(v) for v in metadata["ElementSpacing"].split())
    return np.asarray(values, dtype=np.float64).reshape(nz, ny, nx), spacing


def score(gpu: np.ndarray, topas: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    if not np.any(mask):
        return {
            "available": False,
            "voxels": 0,
            "reason": "no common positive LET voxels inside the TOPAS dose mask",
        }
    g = gpu[mask]
    t = topas[mask]
    delta = g - t
    relative = 100.0 * delta / np.maximum(np.abs(t), 1.0e-12)
    centered_g = g - np.mean(g)
    centered_t = t - np.mean(t)
    correlation = float(
        np.dot(centered_g, centered_t)
        / max(np.linalg.norm(centered_g) * np.linalg.norm(centered_t), 1.0e-30)
    )
    return {
        "available": True,
        "voxels": int(mask.sum()),
        "topas_mean": float(np.mean(t)),
        "gpu_mean": float(np.mean(g)),
        "mean_bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_relative_percent": float(np.mean(relative)),
        "median_absolute_relative_percent": float(np.median(np.abs(relative))),
        "p90_absolute_relative_percent": float(np.percentile(np.abs(relative), 90)),
        "p95_absolute_relative_percent": float(np.percentile(np.abs(relative), 95)),
        "pearson_r": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas-dose", type=Path, required=True)
    parser.add_argument("--topas-primary", type=Path, required=True)
    parser.add_argument("--topas-all", type=Path, required=True)
    parser.add_argument("--gpu-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument(
        "--mapping",
        choices=("tps_x", "beam_y", "identity"),
        default="tps_x",
    )
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--gamma-points", type=int, default=1_000_000)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    parser.add_argument(
        "--all-hadron-gamma",
        action="store_true",
        help="Calculate global/local 3%%/3 mm, 2%%/2 mm, 1%%/1 mm, and "
        "3%%/0 mm LET gamma using the TOPAS-dose threshold mask",
    )
    parser.add_argument(
        "--body-mask",
        type=Path,
        help="Patient-axis float32 MHD mask; LET and dose statistics exclude mask=0",
    )
    args = parser.parse_args()

    dose, spacing = native(args.topas_dose)
    topas_primary, _ = native(args.topas_primary)
    topas_all, _ = native(args.topas_all)
    shape = (dose.shape[2], dose.shape[1], dose.shape[0])
    gpu_primary = mapped_gpu(
        args.gpu_prefix.with_name(args.gpu_prefix.name + "_primary_c12.mhd"),
        shape,
        args.flip_x,
        args.mapping,
    )
    gpu_all = mapped_gpu(
        args.gpu_prefix.with_name(args.gpu_prefix.name + "_all_hadron.mhd"),
        shape,
        args.flip_x,
        args.mapping,
    )
    if args.body_mask:
        body, _ = native(args.body_mask)
        if body.shape != dose.shape:
            raise ValueError(
                f"body mask shape {body.shape} does not match dose {dose.shape}"
            )
        body_mask = body > 0.5
        topas_primary = np.where(body_mask, topas_primary, 0.0)
        topas_all = np.where(body_mask, topas_all, 0.0)
        gpu_primary = np.where(body_mask, gpu_primary, 0.0)
        gpu_all = np.where(body_mask, gpu_all, 0.0)
    else:
        body_mask = np.ones(dose.shape, dtype=bool)
    dose = np.where(body_mask, dose, 0.0)
    dose_mask = body_mask & (
        dose >= args.threshold_percent / 100.0 * float(np.max(dose))
    )

    report: dict[str, object] = {
        "definition": "dose-averaged electronic LET, MeV/mm/(g/cm3)",
        "mask": f"TOPAS dose >= {args.threshold_percent:g}% of maximum",
        "body_mask": str(args.body_mask) if args.body_mask else None,
        "body_mask_voxels": int(body_mask.sum()),
        "shape_zyx": list(dose.shape),
        "spacing_xyz_mm": list(spacing),
        "dose_mask_voxels": int(dose_mask.sum()),
        "primary_c12": score(
            gpu_primary,
            topas_primary,
            dose_mask & (topas_primary > 0.0) & (gpu_primary > 0.0),
        ),
        "all_hadron": score(
            gpu_all,
            topas_all,
            dose_mask & (topas_all > 0.0) & (gpu_all > 0.0),
        ),
        "topas_zero_voxels_inside_dose_mask": {
            "primary_c12": int(np.count_nonzero(dose_mask & (topas_primary <= 0.0))),
            "all_hadron": int(np.count_nonzero(dose_mask & (topas_all <= 0.0))),
        },
    }
    if args.all_hadron_gamma:
        topas_flat = array.array("f", topas_all.reshape(-1))
        gpu_flat = gpu_all.reshape(-1).tolist()
        gamma_mask = dose_mask.reshape(-1)
        gamma_report: dict[str, object] = {
            "quantity": "all-hadron LET_d",
            "selection_mask": (
                f"RTSTRUCT BODY and TOPAS dose >= {args.threshold_percent:g}% "
                "of BODY dose maximum; no LET threshold"
            ),
            "global_normalization": (
                "maximum TOPAS all-hadron LET_d inside the dose selection mask"
            ),
            "selected_voxels": int(gamma_mask.sum()),
        }
        for dose_percent, distance_mm in (
            (3.0, 3.0),
            (2.0, 2.0),
            (1.0, 1.0),
        ):
            label = f"{dose_percent:g}pct_{distance_mm:g}mm"
            gamma_report[f"global_{label}"] = gamma_3d(
                topas_flat,
                gpu_flat,
                shape,
                spacing,
                dose_percent=dose_percent,
                distance_mm=distance_mm,
                thr_percent=args.threshold_percent,
                max_points=args.gamma_points,
                seed=0,
                interpolation_step_mm=args.gamma_resolution_mm,
                selection_mask=gamma_mask,
            )
            gamma_report[f"local_{label}"] = gamma_3d(
                topas_flat,
                gpu_flat,
                shape,
                spacing,
                dose_percent=dose_percent,
                distance_mm=distance_mm,
                thr_percent=args.threshold_percent,
                max_points=args.gamma_points,
                seed=0,
                local_dose=True,
                interpolation_step_mm=args.gamma_resolution_mm,
                selection_mask=gamma_mask,
            )
        gamma_report["global_3pct_0mm"] = dose_only_pass_rate(
            topas_flat,
            gpu_flat,
            dose_percent=3.0,
            thr_percent=args.threshold_percent,
            selection_mask=gamma_mask,
        )
        gamma_report["local_3pct_0mm"] = dose_only_pass_rate(
            topas_flat,
            gpu_flat,
            dose_percent=3.0,
            thr_percent=args.threshold_percent,
            local_dose=True,
            selection_mask=gamma_mask,
        )
        report["all_hadron_gamma"] = gamma_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "let_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    peak = np.unravel_index(int(np.argmax(dose)), dose.shape)
    slices = [
        (
            topas_all[peak[0]],
            gpu_all[peak[0]],
            f"axial z index {peak[0]}",
            spacing[0],
            spacing[1],
        ),
        (
            topas_all[:, peak[1], :],
            gpu_all[:, peak[1], :],
            f"coronal y index {peak[1]}",
            spacing[0],
            spacing[2],
        ),
        (
            topas_all[:, :, peak[2]],
            gpu_all[:, :, peak[2]],
            f"sagittal x index {peak[2]}",
            spacing[1],
            spacing[2],
        ),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
    for row, (topas, gpu, title, sx, sy) in enumerate(slices):
        valid = dose_mask[
            peak[0] if row == 0 else slice(None),
            peak[1] if row == 1 else slice(None),
            peak[2] if row == 2 else slice(None),
        ]
        vmax = float(np.percentile(topas[valid], 99)) if np.any(valid) else float(np.max(topas))
        difference = gpu - topas
        limit = float(np.percentile(np.abs(difference[valid]), 99)) if np.any(valid) else 1.0
        panels = [
            (topas, "TOPAS", "magma", 0.0, vmax),
            (gpu, "GPU", "magma", 0.0, vmax),
            (difference, "GPU - TOPAS", "coolwarm", -limit, limit),
        ]
        for col, (image, label, cmap, vmin, vmax_panel) in enumerate(panels):
            rendered = axes[row, col].imshow(
                image,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_panel,
                aspect=sy / sx,
            )
            axes[row, col].set_title(f"{title}: {label}")
            figure.colorbar(
                rendered,
                ax=axes[row, col],
                label=r"LET$_d$ [MeV/mm/(g/cm$^3$)]",
            )
    figure.suptitle("Full-plan all-hadron LET$_d$: TOPAS vs GPU")
    figure.savefig(args.output_dir / "let_all_hadron_multiplanar.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
