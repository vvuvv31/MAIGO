#!/usr/bin/env python3
"""Verify equal-history local 3%/0mm gain for residual-heat scale=0.9 on mfp=0.5.

Baseline production-best before this lever: nuclear_residual_heat_mfp_mm=0.5
(local ~61.15%). Improved arm adds nuclear_residual_heat_scale=0.9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/scripts"))

from analyze_rt07575_equal_history_dose_residual import (  # noqa: E402
    load_gpu_mapped,
    load_patient,
    pair_metrics,
)
from analyze_rt07575_residual_shape_energy import selection_metrics  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402

TOPAS = (
    ROOT
    / "out/ct/RT07575/cascade_secondary_ablation/topas_single_spot_1M/topas_patient_1M.mhd"
)
BASELINE = (
    ROOT
    / "out/ct/RT07575/cascade_secondary_ablation/single_spot_rh_mfp0p5_1M/dose.mhd"
)
IMPROVED = (
    ROOT
    / "out/ct/RT07575/cascade_secondary_ablation/single_spot_mfp0p5_rhs0p9_1M/dose.mhd"
)
PRODUCTION_YAML = ROOT / "config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"


def _gp(block: dict, key: str) -> float:
    entry = block[key]
    return float(entry["pass_percent"] if isinstance(entry, dict) else entry)


def eval_gpu(path: Path, topas: np.ndarray, shape_xyz, spacing) -> dict:
    _, gpu = load_gpu_mapped(path, topas.shape)
    dmax = float(np.max(topas))
    sel = topas >= 0.10 * dmax
    dmax_s = float(np.max(topas[sel]))
    metrics = selection_metrics(topas, gpu, sel, dmax_s)
    gamma = pair_metrics(topas, gpu, sel, shape_xyz, spacing, gamma=True)
    return {
        "path": str(path),
        "selection_voxels": int(np.count_nonzero(sel)),
        "mask": "topas >= 0.10 * topas_dmax",
        "nrmse_pct_dmax": metrics["nrmse_pct_dmax"],
        "mean_E_over_R": float(
            np.mean(gpu[sel]) / max(np.mean(topas[sel]), 1e-30)
        ),
        "gamma_3pct_0mm": {
            "global": _gp(gamma["gamma_3pct_0mm"], "global"),
            "local": _gp(gamma["gamma_3pct_0mm"], "local"),
        },
    }


def main() -> int:
    for path in (TOPAS, BASELINE, IMPROVED, PRODUCTION_YAML):
        if not path.exists():
            raise SystemExit(f"missing required artifact: {path}")

    yaml_text = PRODUCTION_YAML.read_text(encoding="utf-8")
    if "nuclear_residual_heat_mfp_mm: 0.5" not in yaml_text:
        raise SystemExit(f"production YAML missing mfp=0.5: {PRODUCTION_YAML}")
    if "nuclear_residual_heat_scale: 0.9" not in yaml_text:
        raise SystemExit(f"production YAML missing scale=0.9: {PRODUCTION_YAML}")

    _, topas = load_patient(TOPAS)
    meta, _ = load(TOPAS)
    spacing = tuple(float(v) for v in meta["ElementSpacing"].split())
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())

    baseline = eval_gpu(BASELINE, topas, shape_xyz, spacing)
    improved = eval_gpu(IMPROVED, topas, shape_xyz, spacing)
    delta = (
        improved["gamma_3pct_0mm"]["local"]
        - baseline["gamma_3pct_0mm"]["local"]
    )
    if delta <= 0.0:
        raise SystemExit(
            f"expected positive local gamma Δ, got {delta:.4f} pp "
            f"(baseline={baseline['gamma_3pct_0mm']['local']:.4f}, "
            f"improved={improved['gamma_3pct_0mm']['local']:.4f})"
        )

    report = {
        "protocol": {
            "case": "RT07575",
            "histories": 1000000,
            "topas": str(TOPAS),
            "mask": "topas >= 0.10 * topas_dmax",
            "baseline_lever": "nuclear_residual_heat_mfp_mm=0.5",
            "new_lever": "nuclear_residual_heat_scale=0.9",
            "production_yaml": str(PRODUCTION_YAML),
        },
        "baseline": baseline,
        "improved": improved,
        "delta_local_3pct_0mm_pp": delta,
        "topas_topas_reference_local_3pct_0mm": 83.0,
        "gap_to_topas_topas_pp": 83.0 - improved["gamma_3pct_0mm"]["local"],
    }
    print(json.dumps(report, indent=2))
    print(
        f"OK: local 3%/0mm {baseline['gamma_3pct_0mm']['local']:.2f}% → "
        f"{improved['gamma_3pct_0mm']['local']:.2f}% "
        f"(Δ {delta:+.2f} pp); gap to 83% = "
        f"{report['gap_to_topas_topas_pp']:.1f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
