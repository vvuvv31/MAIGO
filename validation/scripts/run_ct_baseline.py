#!/usr/bin/env python3
"""One-shot CT validation baseline: primary-only, secondary multimat, TOPAS compare.

Produces absolute MeV/primary IDDs and metrics JSON (no global dose scale).

Usage (from repo root, after oneAPI env + build):
  python validation/scripts/run_ct_baseline.py
  python validation/scripts/run_ct_baseline.py --skip-gpu   # metrics only from existing CSVs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "validation" / "scripts" / "compare_ct_patient.py"
RESULTS = ROOT / "validation" / "results" / "ct"
TOPAS_REF = (
    ROOT / "validation" / "topas" / "output" / "ct_patient_development_energy_deposit.csv"
)
PRIMARY_CFG = ROOT / "config" / "beam_200MeVu_ct_patient_multimat.yaml"
SECONDARY_CFG = ROOT / "config" / "beam_200MeVu_ct_patient_multimat_secondary.yaml"
PRIMARY_OUT = ROOT / "out" / "ct" / "e150_patient_multimat_idd.csv"
SECONDARY_OUT = ROOT / "out" / "ct" / "e150_patient_multimat_secondary_idd.csv"
MC = ROOT / "build" / "oneapi-windows-release-grok" / "carbon_mc.exe"


def _run_mc(config: Path) -> None:
    if not MC.is_file():
        raise SystemExit(f"Missing {MC}; build carbon_mc first")
    # Prefer calling through oneAPI env on Windows via cmd wrapper when needed.
    cmd = [str(MC), "--config", str(config)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _compare(gpu: Path, out_json: Path, out_topas_idd: Path, histories: int) -> dict:
    cmd = [
        sys.executable,
        str(COMPARE),
        "--gpu",
        str(gpu),
        "--topas",
        str(TOPAS_REF),
        "--histories",
        str(histories),
        "--output-json",
        str(out_json),
        "--output-topas-idd",
        str(out_topas_idd),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gpu", action="store_true", help="Reuse existing IDD CSVs")
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="Also copy baseline artifacts here",
    )
    parser.add_argument("--histories", type=int, default=20000)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if not TOPAS_REF.is_file():
        raise SystemExit(f"Missing TOPAS reference {TOPAS_REF}")

    if not args.skip_gpu:
        _run_mc(PRIMARY_CFG)
        _run_mc(SECONDARY_CFG)

    if not PRIMARY_OUT.is_file() or not SECONDARY_OUT.is_file():
        raise SystemExit("Missing GPU IDD CSVs; run without --skip-gpu")

    shutil.copy(PRIMARY_OUT, RESULTS / "gpu_e150_patient_multimat_idd.csv")
    shutil.copy(SECONDARY_OUT, RESULTS / "gpu_e150_patient_multimat_secondary_idd.csv")

    m_pri = _compare(
        PRIMARY_OUT,
        RESULTS / "compare_patient_primary.metrics.json",
        RESULTS / "topas_e150_patient_idd.csv",
        args.histories,
    )
    m_sec = _compare(
        SECONDARY_OUT,
        RESULTS / "compare_patient_secondary.metrics.json",
        RESULTS / "topas_e150_patient_idd.csv",
        args.histories,
    )

    summary = {
        "primary_only": {
            "idd": str(RESULTS / "gpu_e150_patient_multimat_idd.csv"),
            "delta_R80_mm": m_pri.get("delta_R80_mm"),
            "integral_rel_diff": m_pri.get("integral_rel_diff"),
            "nrmse": m_pri.get("nrmse"),
            "gpu_integral_MeV": m_pri.get("gpu_integral_MeV"),
        },
        "multimat_secondary": {
            "idd": str(RESULTS / "gpu_e150_patient_multimat_secondary_idd.csv"),
            "delta_R80_mm": m_sec.get("delta_R80_mm"),
            "integral_rel_diff": m_sec.get("integral_rel_diff"),
            "nrmse": m_sec.get("nrmse"),
            "gpu_integral_MeV": m_sec.get("gpu_integral_MeV"),
        },
        "topas_reference": str(TOPAS_REF),
        "topas_integral_MeV": m_sec.get("topas_integral_MeV"),
        "absolute_MeV_per_primary": True,
        "global_dose_scale": False,
        "commands": {
            "primary": f"carbon_mc --config {PRIMARY_CFG.relative_to(ROOT)}",
            "secondary": f"carbon_mc --config {SECONDARY_CFG.relative_to(ROOT)}",
            "baseline": "python validation/scripts/run_ct_baseline.py",
        },
    }
    summary_path = RESULTS / "ct_baseline_summary.metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.scratch is not None:
        dest = args.scratch / "ct_baseline"
        dest.mkdir(parents=True, exist_ok=True)
        for name in [
            "gpu_e150_patient_multimat_idd.csv",
            "gpu_e150_patient_multimat_secondary_idd.csv",
            "compare_patient_primary.metrics.json",
            "compare_patient_secondary.metrics.json",
            "ct_baseline_summary.metrics.json",
            "topas_e150_patient_idd.csv",
        ]:
            src = RESULTS / name
            if src.is_file():
                shutil.copy(src, dest / name)
        print("Copied baseline to", dest)

    for key, m in (("primary_only", m_pri), ("multimat_secondary", m_sec)):
        for field in ("delta_R80_mm", "integral_rel_diff", "nrmse"):
            if field not in m:
                raise SystemExit(f"{key} metrics missing {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
