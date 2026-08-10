#!/usr/bin/env python3
"""Dependency-free TOPAS/GPU ancestor-attributed IDD comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


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


def load(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        rows = [{name: float(value) for name, value in row.items()} for row in reader]
    if len(rows) < 3 or "depth_mm" not in rows[0]:
        raise ValueError(f"Expected a one-dimensional IDD table in {path}")
    if any(rows[index]["depth_mm"] <= rows[index - 1]["depth_mm"] for index in range(1, len(rows))):
        raise ValueError(f"Depth grid in {path} must be strictly increasing")
    return rows


def require_columns(rows: list[dict[str, float]], path: Path, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def values(rows: list[dict[str, float]], column: str, indices: list[int]) -> list[float]:
    return [rows[index][column] for index in indices]


def interval_metrics(reference: list[float], gpu: list[float]) -> dict[str, float | None]:
    reference_sum = math.fsum(reference)
    gpu_sum = math.fsum(gpu)
    reference_max = max(reference)
    return {
        "topas_deposited_MeV_per_primary": reference_sum,
        "gpu_deposited_MeV_per_primary": gpu_sum,
        "gpu_minus_topas_MeV_per_primary": gpu_sum - reference_sum,
        "relative_difference_percent":
            (gpu_sum / reference_sum - 1.0) * 100.0 if reference_sum != 0.0 else None,
        "nrmse_normalized_to_topas_maximum":
            math.sqrt(math.fsum((actual - expected) ** 2 for actual, expected in zip(gpu, reference))
                      / len(reference)) / reference_max
            if reference_max > 0.0
            else None,
    }


def charged_topas(rows: list[dict[str, float]], index: int) -> float:
    return math.fsum(rows[index][topas_column] for _, topas_column, _ in CATEGORY_MAP)


def metrics_for_indices(
    topas: list[dict[str, float]], gpu: list[dict[str, float]], indices: list[int]
) -> dict[str, object]:
    raw_total = values(topas, "total_from_3d_MeV_per_primary", indices)
    gpu_total = values(gpu, "total_MeV_per_primary", indices)
    charged = [charged_topas(topas, index) for index in indices]
    neutral = [
        topas[index]["neutron_MeV_per_primary"]
        + topas[index]["gamma_MeV_per_primary"]
        + topas[index]["neutral_other_MeV_per_primary"]
        for index in indices
    ]
    unresolved = values(topas, "unresolved_MeV_per_primary", indices)
    categories = {
        label: interval_metrics(
            values(topas, topas_column, indices), values(gpu, gpu_column, indices)
        )
        for label, topas_column, gpu_column in CATEGORY_MAP
    }
    return {
        "gpu_vs_topas_raw_total": interval_metrics(raw_total, gpu_total),
        "gpu_vs_topas_charged_origin_total": interval_metrics(charged, gpu_total),
        "topas_neutral_origin_deposited_MeV_per_primary": math.fsum(neutral),
        "topas_neutral_origin_fraction_of_total_percent":
            math.fsum(neutral) / math.fsum(raw_total) * 100.0,
        "topas_unresolved_MeV_per_primary": math.fsum(unresolved),
        "categories": categories,
    }


def svg_path(depth: list[float], dose: list[float], x: float, y: float,
             width: float, height: float, maximum: float) -> str:
    x0, x1 = depth[0], depth[-1]
    points = []
    for depth_value, dose_value in zip(depth, dose):
        px = x + (depth_value - x0) / (x1 - x0) * width
        py = y + height - dose_value / maximum * height
        points.append(f"{px:.2f},{py:.2f}")
    return "M" + " L".join(points)


def write_svg(path: Path, topas: list[dict[str, float]], gpu: list[dict[str, float]],
              metrics: dict[str, object], tail_start_mm: float) -> None:
    depth = [row["depth_mm"] for row in topas]
    panels = (
        (
            "Total IDD",
            [row["total_from_3d_MeV_per_primary"] for row in topas],
            [row["total_MeV_per_primary"] for row in gpu],
            metrics["all_depth"]["gpu_vs_topas_raw_total"]["relative_difference_percent"],
            metrics["tail"]["gpu_vs_topas_raw_total"]["relative_difference_percent"],
        ),
        (
            "Helium-attributed IDD",
            [row["helium_MeV_per_primary"] for row in topas],
            [row["helium_MeV_per_primary"] for row in gpu],
            metrics["all_depth"]["categories"]["helium"]["relative_difference_percent"],
            metrics["tail"]["categories"]["helium"]["relative_difference_percent"],
        ),
        (
            "Proton-attributed IDD",
            [row["proton_MeV_per_primary"] for row in topas],
            [row["proton_MeV_per_primary"] for row in gpu],
            metrics["all_depth"]["categories"]["proton"]["relative_difference_percent"],
            metrics["tail"]["categories"]["proton"]["relative_difference_percent"],
        ),
    )
    width, height = 1200, 900
    plot_x, plot_width, plot_height = 90.0, 1040.0, 200.0
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#202124}.title{font-size:18px;font-weight:600}.axis{font-size:13px}.note{font-size:14px}.grid{stroke:#d8dbe0;stroke-width:1}.frame{stroke:#555;stroke-width:1;fill:none}.topas{stroke:#111;stroke-width:2;fill:none}.gpu{stroke:#d62728;stroke-width:2;stroke-dasharray:7 4;fill:none}.tail{stroke:#777;stroke-width:1;stroke-dasharray:3 4}</style>',
        '<text x="600" y="27" text-anchor="middle" class="title">TOPAS vs Intel Arc B580 — attribution-aligned 100k IDD</text>',
        '<text x="600" y="50" text-anchor="middle" class="axis">solid: TOPAS · dashed: GPU · no global scaling</text>',
    ]
    for panel_index, (title, reference, evaluation, all_difference, tail_difference) in enumerate(panels):
        plot_y = 82.0 + panel_index * 270.0
        maximum = max(max(reference), max(evaluation)) * 1.05
        tail_x = plot_x + (tail_start_mm - depth[0]) / (depth[-1] - depth[0]) * plot_width
        chunks.extend([
            f'<text x="{plot_x:.0f}" y="{plot_y - 15:.0f}" class="title">{title}</text>',
            f'<text x="{plot_x + plot_width:.0f}" y="{plot_y - 15:.0f}" text-anchor="end" class="note">all: {all_difference:+.2f}% · ≥{tail_start_mm:g} mm: {tail_difference:+.2f}%</text>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_height / 2}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height / 2}" class="grid"/>',
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" class="frame"/>',
            f'<line x1="{tail_x:.2f}" y1="{plot_y}" x2="{tail_x:.2f}" y2="{plot_y + plot_height}" class="tail"/>',
            f'<path d="{svg_path(depth, reference, plot_x, plot_y, plot_width, plot_height, maximum)}" class="topas"/>',
            f'<path d="{svg_path(depth, evaluation, plot_x, plot_y, plot_width, plot_height, maximum)}" class="gpu"/>',
            f'<text x="{plot_x - 10}" y="{plot_y + 5}" text-anchor="end" class="axis">{maximum:.3g}</text>',
            f'<text x="{plot_x - 10}" y="{plot_y + plot_height + 5}" text-anchor="end" class="axis">0</text>',
            f'<text x="{plot_x}" y="{plot_y + plot_height + 22}" text-anchor="middle" class="axis">{depth[0]:g}</text>',
            f'<text x="{plot_x + plot_width}" y="{plot_y + plot_height + 22}" text-anchor="middle" class="axis">{depth[-1]:g} mm</text>',
            f'<text x="{tail_x + 5:.2f}" y="{plot_y + 16}" class="axis">{tail_start_mm:g} mm</text>',
        ])
    chunks.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(chunks) + "\n")


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
    require_columns(topas, args.topas, {
        "total_from_3d_MeV_per_primary", "neutron_MeV_per_primary",
        "gamma_MeV_per_primary", "neutral_other_MeV_per_primary",
        "unresolved_MeV_per_primary", *(column for _, column, _ in CATEGORY_MAP),
    })
    require_columns(gpu, args.gpu, {
        "total_MeV_per_primary", *(column for _, _, column in CATEGORY_MAP),
    })
    if len(topas) != len(gpu) or any(
        abs(reference["depth_mm"] - evaluation["depth_mm"]) > 1.0e-9
        for reference, evaluation in zip(topas, gpu)
    ):
        raise ValueError("TOPAS and GPU depth grids do not match")

    all_indices = list(range(len(topas)))
    tail_indices = [index for index, row in enumerate(topas)
                    if row["depth_mm"] >= args.tail_start_mm]
    if len(tail_indices) < 2:
        raise ValueError("Tail interval must include at least two bins")
    metrics = {
        "reference": args.topas.as_posix(),
        "evaluation": args.gpu.as_posix(),
        "semantics": (
            "TOPAS electrons/positrons inherit their charged parent category; charged nuclear "
            "products are classified by their own Z/A; neutral-source lineages remain separate. "
            "GPU other is compared with TOPAS other_charged."
        ),
        "global_scale_applied": False,
        "tail_start_mm": args.tail_start_mm,
        "all_depth": metrics_for_indices(topas, gpu, all_indices),
        "tail": metrics_for_indices(topas, gpu, tail_indices),
        "gpu_species_max_bin_closure_MeV_per_primary": max(
            abs(
                row["total_MeV_per_primary"]
                - math.fsum(row[column] for _, _, column in CATEGORY_MAP)
            )
            for row in gpu
        ),
        "gpu_spatial_3d_comparison_available": False,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(metrics, indent=2) + "\n")
    write_svg(args.plot, topas, gpu, metrics, args.tail_start_mm)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
