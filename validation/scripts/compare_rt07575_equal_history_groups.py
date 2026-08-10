#!/usr/bin/env python3
"""Compare equal-history RT07575 GPU/GPU, GPU/TOPAS and TOPAS/TOPAS pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_topas_seed_gamma import load, quantity_report  # noqa: E402


FILES_TOPAS = {
    "dose": "dose.mhd",
    "primary_c12_letd": "let_primary_c12.mhd",
    "all_hadron_letd": "let_all_hadron.mhd",
}
FILES_GPU = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}


def load_topas(root: Path) -> dict[str, np.ndarray]:
    return {name: load(root / filename)[1] for name, filename in FILES_TOPAS.items()}


def load_gpu_mapped(root: Path) -> dict[str, np.ndarray]:
    mapped: dict[str, np.ndarray] = {}
    for name, filename in FILES_GPU.items():
        metadata, flat = load(root / filename)
        shape_xyz = tuple(int(v) for v in metadata["DimSize"].split())
        if shape_xyz != (505, 35, 417):
            raise ValueError(f"{root / filename}: unexpected GPU shape {shape_xyz}")
        gpu = flat.reshape(shape_xyz[2], shape_xyz[1], shape_xyz[0])
        # GPU [depth=patient-X, patient-Z, patient-Y] -> patient [Z,Y,X].
        mapped[name] = np.transpose(gpu, (1, 2, 0))[:, :, ::-1].reshape(-1)
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-seed1", type=Path, required=True)
    parser.add_argument("--gpu-seed2", type=Path, required=True)
    parser.add_argument("--topas-seed1", type=Path, required=True)
    parser.add_argument("--topas-seed2", type=Path, required=True)
    parser.add_argument("--body-mask", type=Path, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma-points", type=int, default=50_000)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    args = parser.parse_args()

    topas1 = load_topas(args.topas_seed1)
    topas2 = load_topas(args.topas_seed2)
    gpu1 = load_gpu_mapped(args.gpu_seed1)
    gpu2 = load_gpu_mapped(args.gpu_seed2)
    body_meta, body_values = load(args.body_mask)
    shape_xyz = tuple(int(v) for v in body_meta["DimSize"].split())
    spacing_xyz = tuple(float(v) for v in body_meta["ElementSpacing"].split())
    if shape_xyz != (417, 505, 35):
        raise ValueError(f"unexpected patient/BODY shape {shape_xyz}")
    body = body_values > 0.5
    topas1_peak = float(np.max(topas1["dose"][body]))
    selection = body & (topas1["dose"] >= 0.10 * topas1_peak)

    pair_inputs = {
        "gpu_seed2_vs_gpu_seed1": (gpu1, gpu2, True),
        "gpu_seed1_vs_topas_seed1": (topas1, gpu1, False),
        "topas_seed2_vs_topas_seed1": (topas1, topas2, True),
    }
    report: dict[str, object] = {
        "case": "RT07575",
        "histories_per_result": args.histories,
        "gpu_profile": "best",
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {
            "3d_points": min(args.gamma_points, int(np.count_nonzero(selection))),
            "available_points": int(np.count_nonzero(selection)),
            "interpolation_step_mm": args.gamma_resolution_mm,
            "3pct_0mm_points": "all selected voxels",
        },
        "pairs": {},
    }
    pairs = report["pairs"]
    assert isinstance(pairs, dict)
    for pair_name, (reference, evaluation, same_engine_seeds) in pair_inputs.items():
        print(f"Computing {pair_name}", flush=True)
        pair: dict[str, object] = {
            "reference": pair_name.split("_vs_")[1],
            "evaluation": pair_name.split("_vs_")[0],
            "same_engine_independent_seeds": same_engine_seeds,
            "quantities": {},
        }
        quantities = pair["quantities"]
        assert isinstance(quantities, dict)
        for quantity in FILES_TOPAS:
            print(f"  {quantity}", flush=True)
            item = quantity_report(
                reference[quantity],
                evaluation[quantity],
                selection,
                body,
                shape_xyz,
                spacing_xyz,
                args.gamma_points,
                args.gamma_resolution_mm,
                args.histories,
            )
            if not same_engine_seeds:
                # 1/sqrt(N) projection is valid only for two realizations of
                # the same stochastic model, not for GPU/TOPAS model residuals.
                item.pop("particle_projection", None)
            quantities[quantity] = item
        pairs[pair_name] = pair

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RT07575 equal-history seed comparison",
        "",
        f"Every result uses {args.histories:,} histories. GPU profile: best. No fitted normalization.",
        "Mask: RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax; LET uses the same dose mask.",
        "",
    ]
    labels = {
        "gpu_seed2_vs_gpu_seed1": "GPU seed 2 / GPU seed 1",
        "gpu_seed1_vs_topas_seed1": "GPU seed 1 / TOPAS seed 1",
        "topas_seed2_vs_topas_seed1": "TOPAS seed 2 / TOPAS seed 1",
    }
    for pair_name, pair in pairs.items():
        assert isinstance(pair, dict)
        lines.extend(
            [
                f"## {labels[pair_name]}",
                "",
                "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        quantities = pair["quantities"]
        assert isinstance(quantities, dict)
        for quantity, item in quantities.items():
            assert isinstance(item, dict)
            gamma = item["gamma"]
            cells = [
                f"{gamma[label]['global']['pass_percent']:.3f} / "
                f"{gamma[label]['local']['pass_percent']:.3f}"
                for label in ("33", "22", "11", "30")
            ]
            lines.append(
                f"| {quantity} | "
                f"{item['evaluation_over_reference_body_integral']:.6f} | "
                f"{item['selected_mean_evaluation'] / item['selected_mean_reference']:.6f} | "
                f"{item['selected_nrmse_over_reference_max_percent']:.3f}% | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
