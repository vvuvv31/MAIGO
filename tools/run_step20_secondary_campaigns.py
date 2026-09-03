#!/usr/bin/env python3
"""tools/run_step20_secondary_campaigns.py

Runs TOPAS elemental-target campaigns for all 13 secondary projectiles across
all 13 Schneider target elements at energies 25, 50, 100, 150, 200, 250, 300 MeV/u.
Extracts, validates, and packages correlated inelastic events into
data/schneider/cinel03_secondary_targets.bin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "startup" / "package_tools"))
import cinel02
import cinel03

TOPAS_BIN = Path("/home/wuwei/topas/topas-build/topas")
TOPAS_G4_DATA = "/software/geant4-11.3.2/share/Geant4/data"
TOPAS_LD_PATH = "/software/topas/lib:/software/geant4-11.3.2/lib"

WORK_DIR = Path("/mnt/sda/wuwei/step20_secondary_campaigns")
CAMPAIGN_UUID = "00000000-0000-4000-8000-000000000020"

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

ELEMENT_TARGETS = [
    {"symbol": "H",  "z": 1,  "name": "Target_H",  "element": "Hydrogen",   "density_g_cm3": 0.1,  "hlz_mm": 200.0},
    {"symbol": "C",  "z": 6,  "name": "Target_C",  "element": "Carbon",     "density_g_cm3": 2.0,  "hlz_mm": 100.0},
    {"symbol": "N",  "z": 7,  "name": "Target_N",  "element": "Nitrogen",   "density_g_cm3": 1.0,  "hlz_mm": 100.0},
    {"symbol": "O",  "z": 8,  "name": "Target_O",  "element": "Oxygen",     "density_g_cm3": 1.0,  "hlz_mm": 100.0},
    {"symbol": "Na", "z": 11, "name": "Target_Na", "element": "Sodium",     "density_g_cm3": 0.97, "hlz_mm": 100.0},
    {"symbol": "Mg", "z": 12, "name": "Target_Mg", "element": "Magnesium",  "density_g_cm3": 1.74, "hlz_mm": 100.0},
    {"symbol": "P",  "z": 15, "name": "Target_P",  "element": "Phosphorus", "density_g_cm3": 1.82, "hlz_mm": 100.0},
    {"symbol": "S",  "z": 16, "name": "Target_S",  "element": "Sulfur",     "density_g_cm3": 2.07, "hlz_mm": 100.0},
    {"symbol": "Cl", "z": 17, "name": "Target_Cl", "element": "Chlorine",   "density_g_cm3": 1.56, "hlz_mm": 100.0},
    {"symbol": "Ar", "z": 18, "name": "Target_Ar", "element": "Argon",      "density_g_cm3": 1.40, "hlz_mm": 100.0},
    {"symbol": "K",  "z": 19, "name": "Target_K",  "element": "Potassium",  "density_g_cm3": 0.86, "hlz_mm": 100.0},
    {"symbol": "Ca", "z": 20, "name": "Target_Ca", "element": "Calcium",    "density_g_cm3": 1.55, "hlz_mm": 100.0},
    {"symbol": "Ti", "z": 22, "name": "Target_Ti", "element": "Titanium",   "density_g_cm3": 4.54, "hlz_mm": 50.0},
]

CAMPAIGN_ENERGIES = [25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]

def sha256_file(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_cases():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    scripts_dir = WORK_DIR / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = WORK_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "projectiles": SECONDARY_PROJECTILES,
        "targets": ELEMENT_TARGETS,
        "energies": CAMPAIGN_ENERGIES,
        "uuid": CAMPAIGN_UUID
    }
    (WORK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate single-projectile worker runner
    worker_script = WORK_DIR / "worker_projectile.py"
    worker_script_content = f"""#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

TOPAS_BIN = Path("{TOPAS_BIN}")
TOPAS_G4_DATA = "{TOPAS_G4_DATA}"
TOPAS_LD_PATH = "{TOPAS_LD_PATH}"

ELEMENT_TARGETS = {json.dumps(ELEMENT_TARGETS)}
CAMPAIGN_ENERGIES = {json.dumps(CAMPAIGN_ENERGIES)}
CAMPAIGN_UUID = "{CAMPAIGN_UUID}"

def run_single(proj, elem, energy_mevu, out_dir, histories=1000):
    sym = elem["symbol"]
    tz = elem["z"]
    mat_name = elem["name"]
    elem_name = elem["element"]
    density = elem["density_g_cm3"]
    hlz = elem["hlz_mm"]

    pid = proj["id"]
    pz = proj["z"]
    pa = proj["a"]
    part_name = proj["particle"]
    beam_energy = energy_mevu * pa

    case_tag = f"{{pid}}_{{sym}}_E{{int(energy_mevu)}}"
    case_dir = out_dir / case_tag
    case_dir.mkdir(parents=True, exist_ok=True)

    param_file = case_dir / "run.txt"
    param_content = f\"\"\"
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1000.0 mm
d:Ge/World/HLY = 1000.0 mm
d:Ge/World/HLZ = 2000.0 mm
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/BeamPosition/Parent = "World"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = 1500.0 mm
d:Ge/BeamPosition/RotX = 180.0 deg

s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Material = "{{mat_name}}"
d:Ge/Phantom/HLX = 50.0 mm
d:Ge/Phantom/HLY = 50.0 mm
d:Ge/Phantom/HLZ = {{hlz:.1f}} mm
d:Ge/Phantom/TransX = 0.0 mm
d:Ge/Phantom/TransY = 0.0 mm
d:Ge/Phantom/TransZ = 0.0 mm

sv:Ma/{{mat_name}}/Components = 1 "{{elem_name}}"
uv:Ma/{{mat_name}}/Fractions = 1 1.0
d:Ma/{{mat_name}}/Density = {{density}} g/cm3

s:So/PrimaryBeam/Type = "Beam"
s:So/PrimaryBeam/Component = "BeamPosition"
s:So/PrimaryBeam/BeamParticle = "{{part_name}}"
d:So/PrimaryBeam/BeamEnergy = {{beam_energy:.2f}} MeV
u:So/PrimaryBeam/BeamEnergySpread = 0.0
s:So/PrimaryBeam/BeamPositionDistribution = "None"
s:So/PrimaryBeam/BeamAngularDistribution = "None"
i:So/PrimaryBeam/NumberOfHistoriesInRun = {{histories}}

s:Sc/CarbonInelasticExposure/Quantity = "CarbonInelasticExposureNtuple"
s:Sc/CarbonInelasticExposure/Component = "Phantom"
s:Sc/CarbonInelasticExposure/OutputType = "ASCII"
s:Sc/CarbonInelasticExposure/OutputFile = "cinel03_exposure"
s:Sc/CarbonInelasticExposure/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/CarbonInelasticExposure/ProjectileZ = {{pz}}
i:Sc/CarbonInelasticExposure/ProjectileA = {{pa}}
b:Sc/CarbonInelasticExposure/IncludeSecondaries = "FALSE"
d:Sc/CarbonInelasticExposure/EnergyBinMin = 0.0 MeV
d:Sc/CarbonInelasticExposure/EnergyBinWidth = 1.0 MeV
i:Sc/CarbonInelasticExposure/EnergyBinCount = 501
b:Sc/CarbonInelasticExposure/RequireAuthoritativeCollisionState = "TRUE"

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4ion-inclxx" "CarbonInelasticCapturePhysics" "g4h-elastic_HP" "g4stopping"
d:Ph/Default/CutForAllParticles = 0.05 mm
\"\"\"
    param_file.write_text(param_content, encoding="utf-8")

    env = os.environ.copy()
    env["TOPAS_G4_DATA_DIR"] = TOPAS_G4_DATA
    env["LD_LIBRARY_PATH"] = f"{{TOPAS_LD_PATH}}:{{env.get('LD_LIBRARY_PATH', '')}}"
    raw_dir = case_dir / "raw"
    env["CARBON_CINEL02_OUTPUT_DIR"] = str(raw_dir)
    env["CARBON_CINEL02_CAMPAIGN_UUID"] = CAMPAIGN_UUID
    env["CARBON_CINEL02_RUN_TAG"] = case_tag
    env["CARBON_CINEL02_PRIMARY_ONLY"] = "true"
    env["CARBON_CINEL02_OVERWRITE_CAMPAIGN"] = "true"

    res = subprocess.run([str(TOPAS_BIN), str(param_file)], cwd=case_dir, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED {{case_tag}}: code {{res.returncode}}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-id", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    proj = None
    all_proj = {json.dumps(SECONDARY_PROJECTILES)}
    for p in all_proj:
        if p["id"] == args.proj_id:
            proj = p
            break
    if not proj:
        print(f"Unknown projectile: {{args.proj_id}}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for elem in ELEMENT_TARGETS:
        for energy in CAMPAIGN_ENERGIES:
            tasks.append((proj, elem, energy, out_dir))

    print(f"Running {{len(tasks)}} exposure points for {{proj['symbol']}} with {{args.workers}} parallel workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single, *t) for t in tasks]
        success_count = sum(1 for f in concurrent.futures.as_completed(futures) if f.result())

    print(f"Finished {{proj['symbol']}}: {{success_count}}/{{len(tasks)}} tasks succeeded.")

if __name__ == "__main__":
    main()
"""
    worker_script.write_text(worker_script_content, encoding="utf-8")
    worker_script.chmod(0o755)

    # Generate SLURM sbatch scripts for each projectile
    for proj in SECONDARY_PROJECTILES:
        pid = proj["id"]
        slurm_file = scripts_dir / f"run_campaign_{pid}.sh"
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=c20_{pid}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output={WORK_DIR}/job_c20_{pid}_%j.log
#SBATCH --error={WORK_DIR}/job_c20_{pid}_%j.err

python3 {worker_script} --proj-id {pid} --workers 12 --out-dir {raw_dir / pid}
"""
        slurm_file.write_text(slurm_content, encoding="utf-8")
        slurm_file.chmod(0o755)

    print(f"Generated 13 projectile campaign scripts in {scripts_dir}")

def submit_jobs():
    scripts_dir = WORK_DIR / "scripts"
    job_ids = []
    print("Submitting 13 secondary projectile campaigns to SLURM partition compute...")
    for proj in SECONDARY_PROJECTILES:
        pid = proj["id"]
        slurm_file = scripts_dir / f"run_campaign_{pid}.sh"
        res = subprocess.run(["sbatch", str(slurm_file)], capture_output=True, text=True, check=True)
        jid = res.stdout.strip().split()[-1]
        job_ids.append(jid)
        print(f"  Campaign {pid:6s} ({proj['symbol']}): Submitted batch job {jid}")

    print(f"Waiting for {len(job_ids)} campaign jobs to complete: {job_ids}...")
    while True:
        res = subprocess.run(["squeue", "--noheader", "-o", "%i"], capture_output=True, text=True)
        running = set(res.stdout.split())
        active = [j for j in job_ids if j in running]
        if not active:
            break
        print(f"  {len(active)}/13 campaigns still running ({active[:5]}...); sleeping 15s...")
        time.sleep(15)

    print("All secondary projectile campaign jobs completed!")

def compile_secondary_package():
    print("\n" + "=" * 80)
    print("Auditing and compiling unified cinel03_secondary_targets.bin package")
    print("=" * 80)

    raw_dir = WORK_DIR / "raw"
    all_events = []
    rejection_audit = defaultdict(int)

    cinel_files = sorted(raw_dir.rglob("worker_*.cinel02"))
    print(f"Found {len(cinel_files)} worker raw files across all projectiles...")

    for fpath in cinel_files:
        records = cinel02.read_raw(fpath)
        file_events = 0
        for rec, prods in records:
            if file_events >= 30:
                break
            # Gate: Energy ledger closure under documented tolerance
            e_coll = float(rec["collision_energy_MeV"])
            e_parent = float(rec["parent_energy_MeV"])
            e_loc = float(rec["process_local_deposit_MeV"])
            e_unsupp = float(rec["unsupported_product_energy_MeV"])
            e_prods = sum(float(p["kinetic_energy_MeV"]) for p in prods)
            e_total = e_parent + e_loc + e_unsupp + e_prods
            e_bound = e_coll + max(200.0, 0.20 * e_coll)

            if e_total > e_bound:
                rejection_audit["energy_conservation_violation"] += 1
                continue

            # Role audit
            has_role_error = False
            for p in prods:
                is_ground_ion = (p["z"] > 0 and p["a"] >= p["z"] and (p["pdg"] % 10 == 0) and p["excitation"] <= 1.0e-4)
                is_ground_neutral = (p["pdg"] in (22, 2112) and p["excitation"] <= 1.0e-4)
                if p["role"] == 2 and (is_ground_ion or is_ground_neutral):
                    has_role_error = True
                    break
            if has_role_error:
                rejection_audit["role_error"] += 1
                continue

            all_events.append((rec, prods))
            file_events += 1

    print(f"Collected and audited {len(all_events)} valid inelastic secondary events!")
    print(f"Rejection audit: {dict(rejection_audit)}")

    if not all_events:
        raise RuntimeError("No valid events collected from secondary campaigns!")

    # Sort strictly by (projectile_z, projectile_a, target_z, collision_energy_MeV_per_u)
    print("Sorting events canonically by (Z_p, A_p, Z_t, E)...")
    all_events.sort(key=lambda item: (
        item[0]["projectile_z"],
        item[0]["projectile_a"],
        item[0]["target_z"],
        item[0]["collision_energy_MeV_per_u"]
    ))

    # Build CINPKG04 package
    out_dir = REPO_DIR / "data" / "schneider"
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "cinel03_secondary_targets.bin"
    meta_path = out_dir / "cinel03_secondary_targets.metadata.json"
    audit_path = out_dir / "cinel03_secondary_targets.rejection_audit.json"

    pkg = cinel03.Cinel03Package()
    pkg.minimum_energy_MeV_per_u = 0.5
    pkg.energy_bin_width_MeV_per_u = 1.0
    pkg.minimum_events_per_bin = 1
    pkg.campaign_uuid = CAMPAIGN_UUID

    packed_interactions = []
    packed_products = []
    cells = []

    cursor = 0
    while cursor < len(all_events):
        first_rec = all_events[cursor][0]
        cur_pz = first_rec["projectile_z"]
        cur_pa = first_rec["projectile_a"]
        cur_tz = first_rec["target_z"]
        cur_bin = int(math.floor((first_rec["collision_energy_MeV_per_u"] - 0.5) / 1.0))
        start_cursor = cursor

        while cursor < len(all_events):
            rec = all_events[cursor][0]
            pz = rec["projectile_z"]
            pa = rec["projectile_a"]
            tz = rec["target_z"]
            b = int(math.floor((rec["collision_energy_MeV_per_u"] - 0.5) / 1.0))
            if pz != cur_pz or pa != cur_pa or tz != cur_tz or b != cur_bin:
                break

            packed_interactions.append(cinel02._pack_fixed(rec))
            prods = all_events[cursor][1]
            for prod in prods:
                packed_products.append(cinel02._pack_product(prod))

            cursor += 1

        cell_count = cursor - start_cursor
        e_low = 0.5 + cur_bin * 1.0
        e_up = e_low + 1.0
        cells.append({
            "projectile_z": cur_pz,
            "projectile_a": cur_pa,
            "target_element_z": cur_tz,
            "energy_bin": cur_bin,
            "interaction_offset": start_cursor,
            "interaction_count": cell_count,
            "energy_lower_MeV_per_u": e_low,
            "energy_upper_MeV_per_u": e_up
        })

    pkg.cells = cells
    pkg.interactions = packed_interactions
    pkg.products = packed_products

    # Build energy nodes and offsets
    print("Building global energy nodes and binary indices...")
    cursor = 0
    while cursor < len(all_events):
        first = all_events[cursor][0]
        node_pz = first["projectile_z"]
        node_pa = first["projectile_a"]
        node_tz = first["target_z"]
        node_e = first["collision_energy_MeV_per_u"]
        node_start = cursor

        while cursor < len(all_events):
            cur = all_events[cursor][0]
            if cur["projectile_z"] != node_pz or cur["projectile_a"] != node_pa or \
               cur["target_z"] != node_tz or cur["collision_energy_MeV_per_u"] != node_e:
                break
            cursor += 1

        pkg.energy_nodes.append((node_pz, node_pa, node_tz, node_e))
        pkg.event_offsets.append(node_start)

    pkg.event_offsets.append(len(all_events))
    pkg.event_indices = list(range(len(all_events)))

    print(f"Writing binary package to {bin_path}...")
    pkg.write_binary(bin_path)
    bin_sha256 = sha256_file(bin_path)

    # Compile exact coverage enumeration
    coverage_by_projectile = defaultdict(lambda: {"targets": set(), "energies": set(), "events": 0})
    for n in pkg.energy_nodes:
        k = (n[0], n[1])
        coverage_by_projectile[k]["targets"].add(n[2])
        coverage_by_projectile[k]["energies"].add(round(n[3], 1))
        coverage_by_projectile[k]["events"] += 1

    coverage_summary = []
    for (pz, pa), cov in sorted(coverage_by_projectile.items()):
        coverage_summary.append({
            "projectile_z": pz,
            "projectile_a": pa,
            "unique_targets_covered": len(cov["targets"]),
            "targets": sorted(list(cov["targets"])),
            "energy_nodes_count": len(cov["energies"]),
            "total_events": cov["events"]
        })

    meta = {
        "schema_version": 1,
        "format": "CINPKG04",
        "data_filename": "cinel03_secondary_targets.bin",
        "data_sha256": bin_sha256,
        "file_size_bytes": bin_path.stat().st_size,
        "campaign_uuid": CAMPAIGN_UUID,
        "total_interactions": len(all_events),
        "total_products": len(packed_products),
        "total_energy_nodes": len(pkg.energy_nodes),
        "coverage_enumeration": coverage_summary,
        "policy": {
            "be6_status": "EXCLUDED (TopasCompatKill)",
            "aliasing_status": "FORBIDDEN (No O/H alias, strict target element Z lookup)"
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(dict(rejection_audit), indent=2), encoding="utf-8")
    print(f"Successfully compiled {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Metadata saved to {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="Secondary elemental campaigns generator and compiler")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compile", action="store_true")
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
        compile_secondary_package()

if __name__ == "__main__":
    main()
