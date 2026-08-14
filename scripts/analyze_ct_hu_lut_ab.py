#!/usr/bin/env python3
"""Summarize CT density-SPR versus TOPAS HU-LUT dose diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from ct_fullplan_diagnostics import CASES, read_cctg, read_mhd


ORIGIN_CATEGORIES = (
    "primary_c12", "secondary_carbon", "boron", "beryllium",
    "lithium", "helium", "proton", "other_charged",
)
MODELS = (
    "topas_hu_lut",
    "topas_hu_lut_section07_primary",
    "topas_hu_lut_section07",
    "topas_hu_lut_section07_primary_100k",
    "topas_hu_lut_section07_100k",
    "topas_hu_lut_local_deposit_primary_probe",
    "topas_hu_lut_local_deposit_full_probe",
)


def summarize_model(repo: Path, case: str, model: str, profile: str,
                    topas: np.ndarray, inside: np.ndarray,
                    selected: np.ndarray, section: np.ndarray) -> dict[str, object]:
    if model == "production_density_spr":
        dose_path = repo / f"benchmark/ct/result/{case}/{profile}/dose.mhd"
        origin_root = None
    else:
        root = repo / f"benchmark/ct/result/diagnostics/hu_lut_ab/{case}/{model}/{profile}"
        dose_path = root / "dose.mhd"
        origin_root = root
    _, dose = read_mhd(dose_path)
    spec = CASES[case]
    axis_zyx = 2 - "xyz".index(str(spec["beam_axis"]))
    transverse = tuple(axis for axis in range(3) if axis != axis_zyx)
    reference_depth = (topas * inside).sum(axis=transverse, dtype=np.float64)
    dose_depth = (dose * inside).sum(axis=transverse, dtype=np.float64)
    depth_mask = reference_depth >= 0.1 * reference_depth.max()
    materials = []
    for label, material_mask in (
        ("lung", section == 1),
        ("soft_tissue", (section >= 2) & (section <= 8)),
        ("bone", section >= 9),
    ):
        mask = selected & material_mask
        reference_sum = float(topas[mask].sum(dtype=np.float64))
        evaluation_sum = float(dose[mask].sum(dtype=np.float64))
        materials.append({
            "material": label,
            "voxels": int(mask.sum()),
            "gpu_to_topas": evaluation_sum / reference_sum
            if reference_sum > 0.0 else None,
        })
    origin = []
    if origin_root is not None:
        total = float(dose[selected].sum(dtype=np.float64))
        for category in ORIGIN_CATEGORIES:
            path = origin_root / f"dose_origin_{category}.mhd"
            if path.exists():
                _, values = read_mhd(path)
                origin.append({
                    "category": category,
                    "fraction_of_selected_gpu_dose":
                        float(values[selected].sum(dtype=np.float64)) / total,
                })
    return {
        "model": model,
        "dose_file": str(dose_path.relative_to(repo)),
        "high_dose_integral_gpu_to_topas":
            float(dose[selected].sum(dtype=np.float64) /
                  topas[selected].sum(dtype=np.float64)),
        "beam_depth_gpu_to_topas":
            float(dose_depth[depth_mask].sum() / reference_depth[depth_mask].sum()),
        "gpu_peak_plane": int(np.argmax(dose_depth)),
        "topas_peak_plane": int(np.argmax(reference_depth)),
        "materials": materials,
        "charged_origin_fractions": origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/ct/result/diagnostics/hu_lut_ab.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = {"selection": "BODY and TOPAS dose >= 10% BODY Dmax", "cases": []}
    for case, spec in CASES.items():
        _, topas = read_mhd(repo / str(spec["topas"]))
        _, body = read_mhd(repo / str(spec["body"]))
        _, _, section = read_cctg(repo / str(spec["ct"]))
        inside = body > 0.5
        selected = inside & (topas >= 0.1 * topas[inside].max())
        models = [summarize_model(
            repo, case, "production_density_spr", args.profile,
            topas, inside, selected, section)]
        for model in MODELS:
            candidate = (repo / "benchmark/ct/result/diagnostics/hu_lut_ab" /
                         case / model / args.profile / "dose.mhd")
            if candidate.exists():
                models.append(summarize_model(
                    repo, case, model, args.profile,
                    topas, inside, selected, section))
        result["cases"].append({"case": case, "models": models})
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
