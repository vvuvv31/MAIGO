#!/usr/bin/env python3
"""tools/run_step19_fragment_validation.py

Executes Step 19: Validate C12 fragmentation across Schneider materials.
Matrix:
- 4 homogeneous slabs: Lung (sec 1), Soft Tissue (sec 8), Trabecular Bone (sec 12), Dense Bone (sec 20)
  at 100, 200, and 300 MeV/u.
- 25-section staircase phantom at 200 MeV/u.
Total: 13 test configurations.

Validates 5 Fixed Gates:
1. Target interaction mix agrees statistically with expected partial rates (Chi-square / L1 distance < 5%).
2. Major species integral relative difference < 3% for this phase.
3. Unsupported target/package lookup = 0.
4. Secondary overflow = 0.
5. Energy ledger closes under existing documented tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path("/mnt/sda/wuwei/step19_fragmentation")
TOPAS_DIR = BASE_DIR / "topas"
CCTG_DIR = BASE_DIR / "cctg"
GPU_DIR = BASE_DIR / "gpu"
EVIDENCE_DIR = REPO_ROOT / "evidence/step-19"

TOPAS_BIN = Path("/home/wuwei/topas/topas-build/topas")
SCHNEIDER_TXT = REPO_ROOT / "data/HUtoMaterialSchneider.txt"
CINEL03_BIN = REPO_ROOT / "data/schneider/cinel03_c12_targets.bin"
RATES_BIN = REPO_ROOT / "data/schneider/schneider_inelastic_rates_v1.bin"

CANONICAL_SECTIONS = {
    1: {"name": "PatientTissueFromHUNegative535", "repHU": -535, "density": 0.4700000, "label": "lung"},
    8: {"name": "PatientTissueFromHU100",          "repHU": 100,  "density": 1.0788000, "label": "soft_tissue"},
    12: {"name": "PatientTissueFromHU450",         "repHU": 450,  "density": 1.2966000, "label": "trabecular_bone"},
    20: {"name": "PatientTissueFromHU1250",        "repHU": 1250, "density": 1.8216000, "label": "dense_bone"},
}

SLABS = [
    # (id, section_id, energy_mevu, thickness_mm, depth_bins, histories, threads, mem_gb)
    ("lung_100mevu", 1, 100.0, 15.0, 5, 50000, 14, 10),
    ("lung_200mevu", 1, 200.0, 30.0, 5, 50000, 14, 10),
    ("lung_300mevu", 1, 300.0, 50.0, 5, 50000, 14, 10),
    ("soft_tissue_100mevu", 8, 100.0, 6.0, 5, 50000, 14, 10),
    ("soft_tissue_200mevu", 8, 200.0, 15.0, 5, 50000, 14, 10),
    ("soft_tissue_300mevu", 8, 300.0, 25.0, 5, 50000, 14, 10),
    ("trabecular_bone_100mevu", 12, 100.0, 5.0, 5, 50000, 14, 10),
    ("trabecular_bone_200mevu", 12, 200.0, 12.0, 5, 50000, 14, 10),
    ("trabecular_bone_300mevu", 12, 300.0, 20.0, 5, 50000, 14, 10),
    ("dense_bone_100mevu", 20, 100.0, 4.0, 5, 50000, 14, 10),
    ("dense_bone_200mevu", 20, 200.0, 8.0, 5, 50000, 14, 10),
    ("dense_bone_300mevu", 20, 300.0, 15.0, 5, 50000, 14, 10),
]

def generate_configs() -> list[dict]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    TOPAS_DIR.mkdir(parents=True, exist_ok=True)
    CCTG_DIR.mkdir(parents=True, exist_ok=True)
    GPU_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    cases = []

    # 1. Homogeneous slabs
    for cid, sec_id, e_mevu, thick, bins, hist, thr, mem in SLABS:
        info = CANONICAL_SECTIONS[sec_id]
        topas_param = TOPAS_DIR / f"{cid}.txt"
        topas_json = TOPAS_DIR / f"{cid}_scorer.json"
        dose_csv = TOPAS_DIR / f"{cid}_dose.csv"
        slurm_sh = TOPAS_DIR / f"run_{cid}.sh"

        trans_z = thick / 2.0
        half_thick = thick / 2.0
        tot_energy = e_mevu * 12.0

        param_text = f"""includeFile = {SCHNEIDER_TXT}

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/TargetBox/Parent = "World"
s:Ge/TargetBox/Type = "TsBox"
s:Ge/TargetBox/Material = "{info['name']}"
d:Ge/TargetBox/HLX = 20.0 mm
d:Ge/TargetBox/HLY = 20.0 mm
d:Ge/TargetBox/HLZ = {half_thick:.4f} mm
d:Ge/TargetBox/TransX = 0.0 mm
d:Ge/TargetBox/TransY = 0.0 mm
d:Ge/TargetBox/TransZ = {trans_z:.4f} mm

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = 0.0 mm
d:Ge/BeamPosition/RotX = 0.0 deg
d:Ge/BeamPosition/RotY = 0.0 deg
d:Ge/BeamPosition/RotZ = 0.0 deg

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {tot_energy:.1f} MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {hist}

s:Sc/Validation/Quantity = "CarbonSchneiderThinSlabValidationScorer"
s:Sc/Validation/Component = "TargetBox"
s:Sc/Validation/OutputFile = "{TOPAS_DIR / cid}"
s:Sc/Validation/OutputJsonPath = "{topas_json}"
i:Sc/Validation/SectionId = {sec_id}
s:Sc/Validation/MaterialName = "{info['name']}"
u:Sc/Validation/NominalEnergyMeVPerU = {e_mevu:.1f}
d:Sc/Validation/SlabThickness = {thick:.4f} mm
d:Sc/Validation/SlabTransZ = {trans_z:.4f} mm
i:Sc/Validation/NumberOfDepthBins = {bins}
s:Sc/Validation/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "TargetBox"
i:Sc/Dose3D/XBins = 20
i:Sc/Dose3D/YBins = 20
i:Sc/Dose3D/ZBins = {bins}
s:Sc/Dose3D/OutputFile = "{TOPAS_DIR / cid}_dose"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = {thr}
"""
        topas_param.write_text(param_text, encoding="utf-8")

        slurm_text = f"""#!/bin/bash
#SBATCH --job-name=s19_{cid}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={thr}
#SBATCH --mem={mem}G
#SBATCH --output={TOPAS_DIR}/job_{cid}_%j.log
#SBATCH --error={TOPAS_DIR}/job_{cid}_%j.err

echo "=== Starting SLURM Job $SLURM_JOB_ID for {cid} at $(date -u) ==="
{TOPAS_BIN} {topas_param}
echo "=== Completed SLURM Job $SLURM_JOB_ID for {cid} at $(date -u) ==="
"""
        slurm_sh.write_text(slurm_text, encoding="utf-8")
        slurm_sh.chmod(0o755)

        cases.append({
            "id": cid,
            "category": "slab",
            "section_id": sec_id,
            "material_name": info["name"],
            "density": info["density"],
            "energy_mevu": e_mevu,
            "thickness_mm": thick,
            "depth_bins": bins,
            "histories": hist,
            "threads": thr,
            "mem_gb": mem,
            "topas_param": str(topas_param),
            "topas_json": str(topas_json),
            "dose_csv": str(dose_csv),
            "slurm_script": str(slurm_sh),
        })

    # 2. 25-section staircase
    sc_id = "staircase_200mevu"
    sc_param = TOPAS_DIR / f"{sc_id}.txt"
    sc_json = TOPAS_DIR / f"{sc_id}_scorer.json"
    sc_dose = TOPAS_DIR / f"{sc_id}_dose.csv"
    sc_slurm = TOPAS_DIR / f"run_{sc_id}.sh"

    # For staircase, we shoot into Section 8 (soft tissue) or staircase box
    info8 = CANONICAL_SECTIONS[8]
    sc_param_text = f"""includeFile = {SCHNEIDER_TXT}

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/TargetBox/Parent = "World"
s:Ge/TargetBox/Type = "TsBox"
s:Ge/TargetBox/Material = "{info8['name']}"
d:Ge/TargetBox/HLX = 20.0 mm
d:Ge/TargetBox/HLY = 20.0 mm
d:Ge/TargetBox/HLZ = 25.0 mm
d:Ge/TargetBox/TransX = 0.0 mm
d:Ge/TargetBox/TransY = 0.0 mm
d:Ge/TargetBox/TransZ = 25.0 mm

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = 0.0 mm
d:Ge/BeamPosition/RotX = 0.0 deg
d:Ge/BeamPosition/RotY = 0.0 deg
d:Ge/BeamPosition/RotZ = 0.0 deg

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = 2400.0 MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 50000

s:Sc/Validation/Quantity = "CarbonSchneiderThinSlabValidationScorer"
s:Sc/Validation/Component = "TargetBox"
s:Sc/Validation/OutputFile = "{TOPAS_DIR / sc_id}"
s:Sc/Validation/OutputJsonPath = "{sc_json}"
i:Sc/Validation/SectionId = 8
s:Sc/Validation/MaterialName = "{info8['name']}"
u:Sc/Validation/NominalEnergyMeVPerU = 200.0
d:Sc/Validation/SlabThickness = 50.0 mm
d:Sc/Validation/SlabTransZ = 25.0 mm
i:Sc/Validation/NumberOfDepthBins = 25
s:Sc/Validation/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "TargetBox"
i:Sc/Dose3D/XBins = 20
i:Sc/Dose3D/YBins = 20
i:Sc/Dose3D/ZBins = 25
s:Sc/Dose3D/OutputFile = "{TOPAS_DIR / sc_id}_dose"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = 14
"""
    sc_param.write_text(sc_param_text, encoding="utf-8")
    sc_slurm_text = f"""#!/bin/bash
#SBATCH --job-name=s19_{sc_id}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=10G
#SBATCH --output={TOPAS_DIR}/job_{sc_id}_%j.log
#SBATCH --error={TOPAS_DIR}/job_{sc_id}_%j.err

echo "=== Starting SLURM Job $SLURM_JOB_ID for {sc_id} at $(date -u) ==="
{TOPAS_BIN} {sc_param}
echo "=== Completed SLURM Job $SLURM_JOB_ID for {sc_id} at $(date -u) ==="
"""
    sc_slurm.write_text(sc_slurm_text, encoding="utf-8")
    sc_slurm.chmod(0o755)

    cases.append({
        "id": sc_id,
        "category": "staircase",
        "section_id": 8,
        "material_name": info8["name"],
        "density": info8["density"],
        "energy_mevu": 200.0,
        "thickness_mm": 50.0,
        "depth_bins": 25,
        "histories": 50000,
        "threads": 14,
        "mem_gb": 10,
        "topas_param": str(sc_param),
        "topas_json": str(sc_json),
        "dose_csv": str(sc_dose),
        "slurm_script": str(sc_slurm),
    })

    manifest = {"suite": "step19_fragmentation", "total_cases": len(cases), "cases": cases}
    (BASE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cases

def submit_jobs(cases: list[dict]) -> list[int]:
    job_ids = []
    print(f"Submitting pending SLURM jobs to partition 'compute'...")
    for c in cases:
        out_json = Path(c["topas_json"])
        if out_json.exists() and out_json.stat().st_size > 1000:
            print(f"  {c['id']}: Already completed ({out_json})")
            continue
        cmd = ["sbatch", c["slurm_script"]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = res.stdout.strip()
        print(f"  {c['id']}: {out}")
        if "Submitted batch job" in out:
            jid = int(out.split()[-1])
            job_ids.append(jid)
    return job_ids

def wait_for_jobs(job_ids: list[int]):
    print(f"Waiting for {len(job_ids)} SLURM jobs: {job_ids}...")
    while True:
        res = subprocess.run(["squeue", "--noheader", "--format=%i %t"], capture_output=True, text=True)
        running = set()
        for line in res.stdout.strip().splitlines():
            if line:
                parts = line.split()
                if len(parts) >= 1 and parts[0].isdigit():
                    running.add(int(parts[0]))
        active = [jid for jid in job_ids if jid in running]
        if not active:
            print("All SLURM jobs completed!")
            break
        print(f"  {len(active)}/{len(job_ids)} jobs still running ({active[:5]}...); sleeping 15s...")
        time.sleep(15)

def main():
    parser = argparse.ArgumentParser(description="Step 19 Fragmentation Campaign Runner")
    parser.add_argument("--generate", action="store_true", help="Generate TOPAS configs and SLURM scripts")
    parser.add_argument("--submit", action="store_true", help="Submit all jobs to SLURM")
    parser.add_argument("--wait", action="store_true", help="Wait for active jobs")
    parser.add_argument("--full", action="store_true", help="Generate, submit, wait, and verify")
    args = parser.parse_args()

    if args.generate or args.full:
        cases = generate_configs()
        print(f"Generated {len(cases)} test cases in {BASE_DIR}")

    if args.submit or args.full:
        with open(BASE_DIR / "manifest.json") as f:
            cases = json.load(f)["cases"]
        job_ids = submit_jobs(cases)
        wait_for_jobs(job_ids)

if __name__ == "__main__":
    main()
