#!/usr/bin/env python3
"""Compare ancestor-attributed TOPAS IDD with the current GPU species IDD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_MAP = (
    ("primary_c12", "primary_c12_MeV_per_primary", "primary_c12_MeV_per_primary"),
    ("secondary_carbon", "secondary_carbon_MeV_per_primary", "secondary_carbon_MeV_per_primary"),
    ("boron", "boron_MeV_per_primary", "boron_MeV_per_primary"),
    ("beryllium", "beryllium_MeV_per_primary", "beryllium_MeV_per_primary"),
    ("lithium", "lithium_MeV_per_primary", "lithium_MeV_per_primary"),
    ("helium", "helium_MeV_per_primary", "helium_MeV_per_primary"),
    ("proton", "proton_MeV_per_primary", "proton_MeV_per_primary"),
    ("other_charged", "other_charged_MeV_per_primary", "other_MeV_per_primary"),
)


def load(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.ndim != 1 or len(data) < 3:
        raise ValueError(f"Expected a one-dimensional IDD table in {path}")
    if "depth_mm" not in (data.dtype.names or ()) or np.any(np.diff(data["depth_mm"]) <= 0):
        raise ValueError(f"Depth grid in {path} must be strictly increasing")
    return data


def require_columns(data: np.ndarray, path: Path, columns: set[str]) -> None:
    missing = sorted(columns - set(data.dtype.names or ()))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def interval_metrics(reference: np.ndarray, gpu: np.ndarray) -> dict[str, float | None]:
    reference_sum = float(np.sum(reference))
    gpu_sum = float(np.sum(gpu))
    reference_max = float(np.max(reference))
    return {
        "topas_deposited_MeV_per_primary": reference_sum,
        "gpu_deposited_MeV_per_primary": gpu_sum,
        "gpu_minus_topas_MeV_per_primary": gpu_sum - reference_sum,
        "relative_difference_percent":
            (gpu_sum / reference_sum - 1.0) * 100.0 if reference_sum != 0.0 else None,
        "nrmse_normalized_to_topas_maximum":
            float(np.sqrt(np.mean((gpu - reference) ** 2)) / reference_max)
            if reference_max > 0.0
            else None,
    }


def metrics_for_mask(
    topas: np.ndarray,
    gpu: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    charged_topas = np.sum(
        [np.asarray(topas[topas_column]) for _, topas_column, _ in CATEGORY_MAP], axis=0
    )
    neutral_topas = (
        np.asarray(topas["neutron_MeV_per_primary"])
        + np.asarray(topas["gamma_MeV_per_primary"])
        + np.asarray(topas["neutral_other_MeV_per_primary"])
    )
    unresolved = np.asarray(topas["unresolved_MeV_per_primary"])
    raw_total = np.asarray(topas["total_from_3d_MeV_per_primary"])
    gpu_total = np.asarray(gpu["total_MeV_per_primary"])

    result: dict[str, object] = {
        "gpu_vs_topas_raw_total": interval_metrics(raw_total[mask], gpu_total[mask]),
        "gpu_vs_topas_charged_origin_total": interval_metrics(
            charged_topas[mask], gpu_total[mask]
        ),
        "topas_neutral_origin_deposited_MeV_per_primary": float(np.sum(neutral_topas[mask])),
        "topas_neutral_origin_fraction_of_total_percent": float(
            np.sum(neutral_topas[mask]) / np.sum(raw_total[mask]) * 100.0
        ),
        "topas_unresolved_MeV_per_primary": float(np.sum(unresolved[mask])),
        "categories": {},
    }
    for label, topas_column, gpu_column in CATEGORY_MAP:
        result["categories"][label] = interval_metrics(
            np.asarray(topas[topas_column])[mask], np.asarray(gpu[gpu_column])[mask]
        )
    return result


def write_plot(path: Path, topas: np.ndarray, gpu: np.ndarray, tail_start_mm: float) -> None:
    depth = np.asarray(topas["depth_mm"])
    charged_topas = np.sum(
        [np.asarray(topas[topas_column]) for _, topas_column, _ in CATEGORY_MAP], axis=0
    )
    raw_total = np.asarray(topas["total_from_3d_MeV_per_primary"])
    gpu_total = np.asarray(gpu["total_MeV_per_primary"])
    neutral = raw_total - charged_topas
    tail = depth >= tail_start_mm

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(depth, raw_total, color="black", label="TOPAS total")
    axes[0, 0].plot(depth, charged_topas, color="#1f77b4", linestyle="--", label="TOPAS charged-origin total")
    axes[0, 0].plot(depth, gpu_total, color="#d62728", alpha=0.85, label="GPU total")
    axes[0, 0].axvline(tail_start_mm, color="gray", linestyle=":")
    axes[0, 0].set(title="Attribution-aligned IDD", xlabel="Depth (mm)", ylabel="MeV / primary / 0.5 mm")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(depth, gpu_total - raw_total, label="GPU - TOPAS total", color="#d62728")
    axes[0, 1].plot(depth, gpu_total - charged_topas, label="GPU - TOPAS charged", color="#1f77b4")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].axvline(tail_start_mm, color="gray", linestyle=":")
    axes[0, 1].set(title="Absolute per-bin difference", xlabel="Depth (mm)", ylabel="MeV / primary / 0.5 mm")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORY_MAP)))
    for color, (label, topas_column, gpu_column) in zip(colors, CATEGORY_MAP):
        axes[1, 0].plot(depth, topas[topas_column], color=color, linewidth=1.1, label=f"TOPAS {label}")
        axes[1, 0].plot(depth, gpu[gpu_column], color=color, linestyle="--", linewidth=0.9, label=f"GPU {label}")
    axes[1, 0].set(title="Charged-origin categories", xlabel="Depth (mm)", ylabel="MeV / primary / 0.5 mm")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=6, ncol=2)

    positive = np.maximum(float(np.max(raw_total)) * 1.0e-10, 1.0e-14)
    axes[1, 1].semilogy(depth[tail], np.maximum(raw_total[tail], positive), color="black", label="TOPAS total")
    axes[1, 1].semilogy(depth[tail], np.maximum(charged_topas[tail], positive), color="#1f77b4", label="TOPAS charged")
    axes[1, 1].semilogy(depth[tail], np.maximum(neutral[tail], positive), color="#2ca02c", label="TOPAS neutral origin")
    axes[1, 1].semilogy(depth[tail], np.maximum(gpu_total[tail], positive), color="#d62728", label="GPU total")
    axes[1, 1].set(title=f"Tail from {tail_start_mm:g} mm", xlabel="Depth (mm)", ylabel="MeV / primary / 0.5 mm")
    axes[1, 1].grid(alpha=0.2, which="both")
    axes[1, 1].legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topas", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("--tail-start-mm", type=float, default=90.0)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    topas = load(args.topas)
    gpu = load(args.gpu)
    topas_columns = {
        "total_from_3d_MeV_per_primary",
        "neutron_MeV_per_primary",
        "gamma_MeV_per_primary",
        "neutral_other_MeV_per_primary",
        "unresolved_MeV_per_primary",
        *(topas_column for _, topas_column, _ in CATEGORY_MAP),
    }
    gpu_columns = {"total_MeV_per_primary", *(gpu_column for _, _, gpu_column in CATEGORY_MAP)}
    require_columns(topas, args.topas, topas_columns)
    require_columns(gpu, args.gpu, gpu_columns)
    if len(topas) != len(gpu) or not np.allclose(
        topas["depth_mm"], gpu["depth_mm"], rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("TOPAS and GPU depth grids do not match")

    depth = np.asarray(topas["depth_mm"])
    tail_mask = depth >= args.tail_start_mm
    if np.count_nonzero(tail_mask) < 2:
        raise ValueError("Tail interval must include at least two bins")
    metrics = {
        "reference": args.topas.as_posix(),
        "evaluation": args.gpu.as_posix(),
        "semantics": (
            "TOPAS electrons/positrons inherit charged-parent ancestry; neutron/gamma descendants "
            "remain neutral-source lineages. GPU other is compared with TOPAS other_charged."
        ),
        "global_scale_applied": False,
        "tail_start_mm": args.tail_start_mm,
        "all_depth": metrics_for_mask(topas, gpu, np.ones(len(depth), dtype=bool)),
        "tail": metrics_for_mask(topas, gpu, tail_mask),
        "gpu_spatial_3d_comparison_available": False,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_plot(args.plot, topas, gpu, args.tail_start_mm)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
