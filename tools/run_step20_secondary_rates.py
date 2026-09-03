#!/usr/bin/env python3
"""tools/run_step20_secondary_rates.py

Extracts and compiles deterministic partial and total inelastic interaction rates
for all 13 transportable secondary projectiles across 25 Schneider sections and
13 target elements using TOPAS/Geant4 with exact process attachment verification.

Projectiles:
  Phase A: B11 (5,11), B10 (5,10), Be9 (4,9), Be7 (4,7), Be10 (4,10), Li7 (3,7), Li6 (3,6)
  Phase B: He4 (2,4), He3 (2,3), H1 (1,1), H2 (1,2), H3 (1,3)
  Phase C: C11 (6,11)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
TOPAS_BIN = Path("/home/wuwei/topas/topas-build/topas")
TOPAS_G4_DATA = "/software/geant4-11.3.2/share/Geant4/data"
TOPAS_LD_PATH = "/software/topas/lib:/software/geant4-11.3.2/lib"

SCHNEIDER_TXT = REPO_DIR / "data" / "HUtoMaterialSchneider.txt"
WORK_DIR = Path("/mnt/sda/wuwei/step20_secondary_rates")

SECONDARY_PROJECTILES = [
    # Phase A
    {"id": "b11",  "symbol": "B11",  "z": 5, "a": 11, "particle": "GenericIon(5,11)"},
    {"id": "b10",  "symbol": "B10",  "z": 5, "a": 10, "particle": "GenericIon(5,10)"},
    {"id": "be9",  "symbol": "Be9",  "z": 4, "a": 9,  "particle": "GenericIon(4,9)"},
    {"id": "be7",  "symbol": "Be7",  "z": 4, "a": 7,  "particle": "GenericIon(4,7)"},
    {"id": "be10", "symbol": "Be10", "z": 4, "a": 10, "particle": "GenericIon(4,10)"},
    {"id": "li7",  "symbol": "Li7",  "z": 3, "a": 7,  "particle": "GenericIon(3,7)"},
    {"id": "li6",  "symbol": "Li6",  "z": 3, "a": 6,  "particle": "GenericIon(3,6)"},
    # Phase B
    {"id": "he4",  "symbol": "He4",  "z": 2, "a": 4,  "particle": "alpha"},
    {"id": "he3",  "symbol": "He3",  "z": 2, "a": 3,  "particle": "He3"},
    {"id": "h1",   "symbol": "H1",   "z": 1, "a": 1,  "particle": "proton"},
    {"id": "h2",   "symbol": "H2",   "z": 1, "a": 2,  "particle": "deuteron"},
    {"id": "h3",   "symbol": "H3",   "z": 1, "a": 3,  "particle": "triton"},
    # Phase C
    {"id": "c11",  "symbol": "C11",  "z": 6, "a": 11, "particle": "GenericIon(6,11)"},
]

CANONICAL_TARGETS = [
    {"name": "Hydrogen",   "z": 1},
    {"name": "Carbon",     "z": 6},
    {"name": "Nitrogen",   "z": 7},
    {"name": "Oxygen",     "z": 8},
    {"name": "Magnesium",  "z": 12},
    {"name": "Phosphorus", "z": 15},
    {"name": "Sulfur",     "z": 16},
    {"name": "Chlorine",   "z": 17},
    {"name": "Argon",      "z": 18},
    {"name": "Calcium",    "z": 20},
    {"name": "Sodium",     "z": 11},
    {"name": "Potassium",  "z": 19},
    {"name": "Titanium",   "z": 22},
]
CANONICAL_Z_ORDER = [t["z"] for t in CANONICAL_TARGETS]

EXPECTED_SECTIONS = 25
EXPECTED_TARGETS = 13
EXPECTED_ENERGIES = 860
ENERGY_MIN = 0.5
ENERGY_MAX = 430.0
ENERGY_STEP = 0.5

def sha256_file(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_cases():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = WORK_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"projectiles": SECONDARY_PROJECTILES, "generated_at": time.time()}

    for proj in SECONDARY_PROJECTILES:
        pid = proj["id"]
        z = proj["z"]
        a = proj["a"]
        part_name = proj["particle"]

        param_file = WORK_DIR / f"{pid}_param.txt"
        json_out = raw_dir / f"{pid}_dump.json"
        csv_out = raw_dir / f"{pid}_dump.csv"
        slurm_file = WORK_DIR / f"run_{pid}.sh"

        param_content = f"""includeFile = {SCHNEIDER_TXT}

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/Box/Parent = "World"
s:Ge/Box/Type = "TsBox"
s:Ge/Box/Material = "G4_WATER"
d:Ge/Box/HLX = 10.0 mm
d:Ge/Box/HLY = 10.0 mm
d:Ge/Box/HLZ = 10.0 mm

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping"

s:So/Source/Type = "Beam"
s:So/Source/Component = "Box"
s:So/Source/BeamParticle = "{part_name}"
d:So/Source/BeamEnergy = {100.0 * a:.1f} MeV
u:So/Source/BeamEnergySpread = 0.0
s:So/Source/BeamPositionDistribution = "None"
s:So/Source/BeamAngularDistribution = "None"
i:So/Source/NumberOfHistoriesInRun = 1

s:Sc/XsDump/Quantity = "CarbonSchneiderInelasticXsDump"
s:Sc/XsDump/Component = "Box"
s:Sc/XsDump/OutputFile = "{json_out}"
s:Sc/XsDump/OutputCsvFile = "{csv_out}"
i:Sc/XsDump/ProjectileZ = {z}
i:Sc/XsDump/ProjectileA = {a}
d:Sc/XsDump/MinEnergyMeVu = {ENERGY_MIN} MeV
d:Sc/XsDump/MaxEnergyMeVu = {ENERGY_MAX} MeV
d:Sc/XsDump/EnergyStepMeVu = {ENERGY_STEP} MeV
"""
        param_file.write_text(param_content, encoding="utf-8")

        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=s20_{pid}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output={WORK_DIR}/job_{pid}_%j.log
#SBATCH --error={WORK_DIR}/job_{pid}_%j.err

export TOPAS_G4_DATA_DIR={TOPAS_G4_DATA}
export LD_LIBRARY_PATH={TOPAS_LD_PATH}:$LD_LIBRARY_PATH

{TOPAS_BIN} {param_file}
"""
        slurm_file.write_text(slurm_content, encoding="utf-8")
        slurm_file.chmod(0o755)

    (WORK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated 13 secondary rate parameter files in {WORK_DIR}")

def submit_jobs():
    job_ids = []
    print("Submitting 13 secondary rate extraction jobs to partition compute...")
    for proj in SECONDARY_PROJECTILES:
        pid = proj["id"]
        slurm_file = WORK_DIR / f"run_{pid}.sh"
        res = subprocess.run(["sbatch", str(slurm_file)], capture_output=True, text=True, check=True)
        # Output format: Submitted batch job 659
        jid = res.stdout.strip().split()[-1]
        job_ids.append(jid)
        print(f"  {pid:6s} (Z={proj['z']}, A={proj['a']}): Submitted batch job {jid}")

    print(f"Waiting for {len(job_ids)} jobs to complete: {job_ids}...")
    while True:
        res = subprocess.run(["squeue", "--noheader", "-o", "%i"], capture_output=True, text=True)
        running = set(res.stdout.split())
        active = [j for j in job_ids if j in running]
        if not active:
            break
        print(f"  {len(active)}/13 jobs still running ({active[:5]}...); sleeping 10s...")
        time.sleep(10)

    print("All SLURM rate extraction jobs completed!")

def compile_secondary_rates():
    print("\nCompiling unified secondary inelastic rates binary table...")
    raw_dir = WORK_DIR / "raw"
    out_dir = REPO_DIR / "data" / "schneider"
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_filename = "secondary_inelastic_rates_v1.bin"
    bin_path = out_dir / bin_filename
    meta_path = out_dir / "secondary_inelastic_rates_v1.metadata.json"
    bin_tmp = out_dir / (bin_filename + ".tmp")

    num_projectiles = len(SECONDARY_PROJECTILES)

    # 4D tensor: [projectile][section][target][energy]
    # 3D tensor: [projectile][section][energy]
    mass_partial_tensor = [
        [[[0.0 for _ in range(EXPECTED_ENERGIES)] for _ in range(EXPECTED_TARGETS)] for _ in range(EXPECTED_SECTIONS)]
        for _ in range(num_projectiles)
    ]
    mass_total_tensor = [
        [[0.0 for _ in range(EXPECTED_ENERGIES)] for _ in range(EXPECTED_SECTIONS)]
        for _ in range(num_projectiles)
    ]

    for p_idx, proj in enumerate(SECONDARY_PROJECTILES):
        pid = proj["id"]
        json_file = raw_dir / f"{pid}_dump.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Missing output file {json_file}")

        with open(json_file) as f:
            d = json.load(f)

        if d.get("projectile_z") != proj["z"] or d.get("projectile_a") != proj["a"]:
            raise ValueError(f"Identity mismatch in {json_file}")

        sections = d.get("sections", [])
        if len(sections) != EXPECTED_SECTIONS:
            raise ValueError(f"Section count mismatch in {json_file}: {len(sections)} != {EXPECTED_SECTIONS}")

        for s_idx, sec in enumerate(sections):
            grid = sec.get("grid", [])
            if len(grid) != EXPECTED_ENERGIES:
                raise ValueError(f"Grid length mismatch in {json_file} sec {s_idx}: {len(grid)} != {EXPECTED_ENERGIES}")

            for e_idx, pt in enumerate(grid):
                tot = pt["mass_total_per_mm_at_1g_cm3"]
                mass_total_tensor[p_idx][s_idx][e_idx] = tot

                elements = pt.get("elements", [])
                sum_part = 0.0
                for t_idx, el in enumerate(elements):
                    tz = el["target_z"]
                    if tz != CANONICAL_Z_ORDER[t_idx]:
                        raise ValueError(f"Target Z order error in {json_file}")
                    part = el["mass_partial_per_mm_at_1g_cm3"]
                    mass_partial_tensor[p_idx][s_idx][t_idx][e_idx] = part
                    sum_part += part

                diff = abs(sum_part - tot)
                if diff > 1e-12:
                    raise ValueError(f"Partial sum discrepancy {diff} at {json_file} s={s_idx} e={e_idx}")

    print("Writing binary package format: SCHN2RAT...")
    with open(bin_tmp, 'wb') as f:
        # Magic: 8 bytes
        f.write(b"SCHN2RAT")
        # Header: uint32 version(1), num_projectiles(13), num_sections(25), num_targets(13), num_energies(860)
        # double energy_min(0.5), double energy_max(430.0), double energy_step(0.5)
        # struct ProjectileKey { int32 z; int32 a; } projectiles[13]
        # int32 target_z[13]
        header = struct.pack(
            "<IIIIIddd",
            1,
            num_projectiles,
            EXPECTED_SECTIONS,
            EXPECTED_TARGETS,
            EXPECTED_ENERGIES,
            ENERGY_MIN,
            ENERGY_MAX,
            ENERGY_STEP
        )
        f.write(header)

        # Write projectiles
        for proj in SECONDARY_PROJECTILES:
            f.write(struct.pack("<ii", proj["z"], proj["a"]))

        # Write target Z order
        for tz in CANONICAL_Z_ORDER:
            f.write(struct.pack("<i", tz))

        # Payload 1: mass_partial_rates[13][25][13][860] (IEEE 754 double precision)
        for p in range(num_projectiles):
            for s in range(EXPECTED_SECTIONS):
                for t in range(EXPECTED_TARGETS):
                    for e in range(EXPECTED_ENERGIES):
                        f.write(struct.pack("<d", mass_partial_tensor[p][s][t][e]))

        # Payload 2: mass_total_rates[13][25][860] (IEEE 754 double precision)
        for p in range(num_projectiles):
            for s in range(EXPECTED_SECTIONS):
                for e in range(EXPECTED_ENERGIES):
                    f.write(struct.pack("<d", mass_total_tensor[p][s][e]))

    bin_tmp.replace(bin_path)
    bin_sha256 = sha256_file(bin_path)

    meta = {
        "schema_version": 1,
        "data_filename": bin_filename,
        "data_sha256": bin_sha256,
        "binary_magic": "SCHN2RAT",
        "binary_version": 1,
        "num_projectiles": num_projectiles,
        "projectiles": SECONDARY_PROJECTILES,
        "num_sections": EXPECTED_SECTIONS,
        "num_targets": EXPECTED_TARGETS,
        "canonical_targets": CANONICAL_TARGETS,
        "canonical_z_order": CANONICAL_Z_ORDER,
        "num_energies": EXPECTED_ENERGIES,
        "energy_min_mevu": ENERGY_MIN,
        "energy_max_mevu": ENERGY_MAX,
        "energy_step_mevu": ENERGY_STEP,
        "file_size_bytes": bin_path.stat().st_size
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Successfully compiled {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.2f} MB, SHA256: {bin_sha256})")

def main():
    parser = argparse.ArgumentParser(description="Extract and compile secondary inelastic rates")
    parser.add_argument("--generate", action="store_true", help="Generate configuration and slurm scripts")
    parser.add_argument("--submit", action="store_true", help="Submit slurm jobs and wait for completion")
    parser.add_argument("--compile", action="store_true", help="Compile raw files into official binary rate table")
    args = parser.parse_args()

    if not (args.generate or args.submit or args.compile):
        args.generate = True
        args.submit = True
        args.compile = True

    if args.generate:
        generate_cases()
    if args.submit:
        submit_jobs()
    if args.compile:
        compile_secondary_rates()

if __name__ == "__main__":
    main()
