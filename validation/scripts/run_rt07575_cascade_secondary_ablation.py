#!/usr/bin/env python3
"""Equal-history RT07575 ablations for cascade final states and residual heat.

Arms (all 12,963,817 histories, seed 20260801, no fitted scale):
  residual_heat_0p5mm  — nuclear_residual_heat_mfp_mm=0.5
  electronic_buildup   — electronic_buildup_fraction=0.06, lateral sigma 0.5 mm
  material_inclxx      — lung/soft/bone INCL++ material reaction+cascade packages
                         (requires diagnostic packages under
                          data/packages/ soft-tissue + validation/results/diagnostic_g4_11_3_2/ lung-bone)

Uses the production best-profile template. Resume-safe: completed arms keep
dose/LET MHD outputs with zero queue overflows.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_rt07575_equal_history_groups import (  # noqa: E402
    load_gpu_mapped,
    load_topas,
)
from match_gpu_to_physical_dose import dose_only_pass_rate, gamma_3d  # noqa: E402

HISTORIES = 12_963_817
SEED = 20_260_801
TEMPLATE = ROOT / "config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"
OUT_ROOT = ROOT / "out/ct/RT07575/cascade_secondary_ablation"
DIAG = ROOT / "validation/results/diagnostic_g4_11_3_2"
PKG = ROOT / "data/packages"
TOPAS_DOSE = ROOT / "ct/fullplan_result/RT07575/topas/dose.mhd"
if not TOPAS_DOSE.exists():
    # Alternate seed-1 path used by equal-history scripts
    TOPAS_DOSE = ROOT / "out/fullplan_result/RT07575/topas/dose.mhd"

FILES_GPU = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}


def material_inclxx_overrides() -> dict[str, str]:
    soft_rx = PKG / "topas_400MeVu_soft_tissue_inclxx_100k_primary_3d.bin"
    soft_cas = PKG / "topas_400MeVu_soft_tissue_inclxx_100k_cascade_3d.bin"
    lung_rx = DIAG / "topas_400MeVu_lung_inclxx_20k_primary_3d.bin"
    lung_cas = DIAG / "topas_400MeVu_lung_inclxx_20k_cascade_3d.bin"
    bone_rx = DIAG / "topas_400MeVu_bone_inclxx_20k_primary_3d.bin"
    bone_cas = DIAG / "topas_400MeVu_bone_inclxx_20k_cascade_3d.bin"
    missing = [p for p in (soft_rx, soft_cas, lung_rx, lung_cas, bone_rx, bone_cas) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "material_inclxx packages missing:\n  " + "\n  ".join(str(p) for p in missing)
        )
    return {
        "reaction_package_file": str(soft_rx),
        "cascade_package_file": str(soft_cas),
        "ct_soft_tissue_reaction_package_file": str(soft_rx),
        "ct_soft_tissue_cascade_package_file": str(soft_cas),
        "ct_lung_reaction_package_file": str(lung_rx),
        "ct_lung_cascade_package_file": str(lung_cas),
        "ct_bone_reaction_package_file": str(bone_rx),
        "ct_bone_cascade_package_file": str(bone_cas),
    }


VARIANTS: dict[str, dict[str, str]] = {
    "residual_heat_0p5mm": {"nuclear_residual_heat_mfp_mm": "0.5"},
    "electronic_buildup": {
        "electronic_buildup_fraction": "0.06",
        "electronic_buildup_mfp_mm": "0.5",
        "electronic_buildup_lateral_sigma_mm": "0.5",
    },
}


def replace_or_append(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count > 1:
        raise ValueError(f"expected at most one {key}, found {count}")
    return updated if count else updated.rstrip() + f"\n{key}: {value}\n"


def render_config(template: Path, output: Path, run_dir: Path, variant: str) -> None:
    text = template.read_text(encoding="utf-8")
    overrides: dict[str, str] = {
        "number_of_histories": str(HISTORIES),
        "random_seed": str(SEED),
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
    }
    if variant == "material_inclxx":
        overrides.update(material_inclxx_overrides())
    else:
        overrides.update(VARIANTS[variant])
    for key, value in overrides.items():
        text = replace_or_append(text, key, value)
    output.write_text(text, encoding="utf-8")


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def num(pat: str, default: float | None = None) -> float:
        m = re.search(pat, text, re.MULTILINE)
        if m is None:
            if default is None:
                raise ValueError(f"missing {pat} in {path}")
            return default
        return float(m.group(1))

    return {
        "histories": int(num(r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": num(r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": num(
            r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"
        ),
        "energy_balance_error": num(r"^Energy balance error:\s*([0-9.eE+-]+)$"),
        "secondary_queue_overflow": int(
            num(r"^Secondary queue overflow:\s*(\d+)$", 0)
        ),
        "cascade_queue_overflow": int(num(r"^Cascade queue overflow:\s*(\d+)$", 0)),
        "neutral_queue_overflow": int(num(r"^Neutral queue overflow:\s*(\d+)$", 0)),
        "untracked_nuclear_energy_MeV": num(
            r"^Untracked nuclear energy:\s*([0-9.eE+-]+)\s*MeV$", 0.0
        ),
    }


def arm_complete(run_dir: Path) -> bool:
    dose = run_dir / "dose.mhd"
    log = run_dir / "run.log"
    if not dose.exists() or not log.exists():
        return False
    try:
        info = parse_log(log)
    except Exception:
        return False
    return (
        info["histories"] == HISTORIES
        and info["secondary_queue_overflow"] == 0
        and info["cascade_queue_overflow"] == 0
    )


def run_arm(binary: Path, variant: str) -> Path:
    run_dir = OUT_ROOT / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    if arm_complete(run_dir):
        print(f"[skip] {variant} already complete")
        return run_dir
    cfg = run_dir / "config.yaml"
    render_config(TEMPLATE, cfg, run_dir, variant)
    log_path = run_dir / "run.log"
    print(f"[run] {variant}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(binary), str(cfg)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{variant} failed with code {proc.returncode}; see {log_path}")
    info = parse_log(log_path)
    if info["histories"] != HISTORIES:
        raise RuntimeError(f"{variant}: histories {info['histories']} != {HISTORIES}")
    if info["secondary_queue_overflow"] or info["cascade_queue_overflow"]:
        raise RuntimeError(f"{variant}: queue overflow in log")
    return run_dir


def compare_to_topas(run_dir: Path) -> dict[str, Any]:
    # Prefer equal-history TOPAS seed 1 if present.
    candidates = [
        ROOT / "out/ct/RT07575/equal_history_best_seeds/topas_seed1/dose.mhd",
        ROOT / "ct/fullplan_result/RT07575/dose.mhd",
        TOPAS_DOSE,
    ]
    topas_path = next((p for p in candidates if p.exists()), None)
    if topas_path is None:
        return {"error": "TOPAS dose reference not found"}
    # Reuse residual diagnostic path: map GPU and compute simple NRMSE/gamma.
    # Full gamma is expensive; keep the same functions as physics ablation.
    try:
        from compare_rt07575_equal_history_groups import compare_pair  # type: ignore
    except Exception:
        compare_pair = None
    meta_t, topas = load_topas(topas_path) if "load_topas" in dir() else (None, None)
    # Fallback: use compare_topas_seed_gamma.load
    if topas is None:
        from compare_topas_seed_gamma import load

        meta, values = load(topas_path)
        shape = tuple(int(v) for v in meta["DimSize"].split())
        topas = np.asarray(values, dtype=np.float32).reshape(shape[2], shape[1], shape[0])
        patient_shape = topas.shape
    else:
        patient_shape = topas.shape
    _, gpu = load_gpu_mapped(run_dir / "dose.mhd", patient_shape)
    selection = topas >= 0.10 * float(np.max(topas))
    # BODY mask optional
    ref = topas[selection]
    ev = gpu[selection]
    dmax = float(np.max(ref)) if ref.size else 1.0
    nrmse = float(np.sqrt(np.mean((ev - ref) ** 2)) / dmax * 100.0) if ref.size else None
    mean_ratio = float(np.mean(ev) / np.mean(ref)) if ref.size and np.mean(ref) != 0 else None
    return {
        "topas_path": str(topas_path),
        "selected_voxels": int(selection.sum()),
        "nrmse_over_reference_dmax_percent": nrmse,
        "mask_mean_eval_over_ref": mean_ratio,
        "log": parse_log(run_dir / "run.log"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "build/oneapi-nvidia-release/carbon_mc",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["residual_heat_0p5mm", "electronic_buildup", "material_inclxx"],
    )
    parser.add_argument("--skip-material", action="store_true")
    args = parser.parse_args()
    arms = list(args.arms)
    if args.skip_material:
        arms = [a for a in arms if a != "material_inclxx"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"arms": {}}
    for arm in arms:
        try:
            run_dir = run_arm(args.binary, arm)
            summary["arms"][arm] = {
                "status": "ok",
                "run_dir": str(run_dir),
                "compare": compare_to_topas(run_dir),
            }
        except Exception as exc:  # noqa: BLE001 - report each arm
            summary["arms"][arm] = {"status": "error", "error": str(exc)}
            print(f"[error] {arm}: {exc}")
    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(v.get("status") == "ok" for v in summary["arms"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
