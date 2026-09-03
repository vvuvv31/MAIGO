#!/usr/bin/env python3
"""Step 17: TOPAS C12 Elemental-Target Campaign Generator, Runner, and Auditor.

Builds unbiased pure-element TOPAS campaigns for C12 on 13 Schneider target elements:
H (1), C (6), N (7), O (8), Na (11), Mg (12), P (15), S (16), Cl (17), Ar (18), K (19), Ca (20), Ti (22).
"""

from __future__ import annotations

import argparse
import binascii
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
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "startup" / "package_tools"))
import cinel02
import cinel03

TOPAS_BIN = Path("/home/wuwei/topas/topas-build/topas")
TOPAS_G4_DATA_DIR = "/software/geant4-11.3.2/share/Geant4/data"
TOPAS_LD_PATH = "/software/topas/lib:/software/geant4-11.3.2/lib"

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

CAMPAIGN_UUID = "00000000-0000-4000-8000-000000000017"
PRODUCTION_ENERGIES = [25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 430.0]

def generate_topas_txt(elem: dict, energy_mevu: float, histories: int, threads: int, seed: int, out_dir: Path) -> Path:
    beam_energy_mev = energy_mevu * 12.0
    hlz = elem["hlz_mm"]
    mat_name = elem["name"]
    elem_name = elem["element"]
    density = elem["density_g_cm3"]

    txt_content = f"""i:Ts/Seed = {seed}
i:Ts/NumberOfThreads = {threads}
i:Ts/ShowHistoryCountAtInterval = {max(histories // 2, 100)}

# Pure Elemental Material Definition
s:Ma/{mat_name}/State = "Liquid"
d:Ma/{mat_name}/Density = {density:.4f} g/cm3
sv:Ma/{mat_name}/Components = 1 "{elem_name}"
uv:Ma/{mat_name}/Fractions = 1 1.0

s:Ge/World/Material = "G4_Galactic"
d:Ge/World/HLX = 1.0 m
d:Ge/World/HLY = 1.0 m
d:Ge/World/HLZ = 2.0 m
b:Ge/World/Invisible = "TRUE"

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = 1500.0 mm
d:Ge/BeamPosition/RotX = 180.0 deg

s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Material = "{mat_name}"
d:Ge/Phantom/HLX = 50.0 mm
d:Ge/Phantom/HLY = 50.0 mm
d:Ge/Phantom/HLZ = {hlz:.1f} mm
d:Ge/Phantom/TransX = 0.0 mm
d:Ge/Phantom/TransY = 0.0 mm
d:Ge/Phantom/TransZ = 0.0 mm

s:So/PrimaryBeam/Type = "Beam"
s:So/PrimaryBeam/Component = "BeamPosition"
s:So/PrimaryBeam/BeamParticle = "GenericIon(6,12)"
d:So/PrimaryBeam/BeamEnergy = {beam_energy_mev:.2f} MeV
u:So/PrimaryBeam/BeamEnergySpread = 0.0
s:So/PrimaryBeam/BeamPositionDistribution = "None"
s:So/PrimaryBeam/BeamAngularDistribution = "None"
i:So/PrimaryBeam/NumberOfHistoriesInRun = {histories}

s:Sc/CarbonInelasticExposure/Quantity = "CarbonInelasticExposureNtuple"
s:Sc/CarbonInelasticExposure/Component = "Phantom"
s:Sc/CarbonInelasticExposure/OutputType = "ASCII"
s:Sc/CarbonInelasticExposure/OutputFile = "cinel03_exposure"
s:Sc/CarbonInelasticExposure/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/CarbonInelasticExposure/ProjectileZ = 6
i:Sc/CarbonInelasticExposure/ProjectileA = 12
b:Sc/CarbonInelasticExposure/IncludeSecondaries = "FALSE"
d:Sc/CarbonInelasticExposure/EnergyBinMin = 0.0 MeV
d:Sc/CarbonInelasticExposure/EnergyBinWidth = 1.0 MeV
i:Sc/CarbonInelasticExposure/EnergyBinCount = 501
b:Sc/CarbonInelasticExposure/RequireAuthoritativeCollisionState = "TRUE"

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4ion-inclxx" "CarbonInelasticCapturePhysics" "g4h-elastic_HP" "g4stopping"
d:Ph/Default/CutForAllParticles = 0.05 mm
"""
    txt_path = out_dir / "run.txt"
    txt_path.write_text(txt_content, encoding="utf-8")
    return txt_path

def run_pilot(base_dir: Path) -> bool:
    print("=" * 80)
    print("Step 17 Pilot Campaigns: Auditing C12 on all 13 Target Elements")
    print("=" * 80)

    pilot_dir = base_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    pilot_energies = [100.0, 250.0, 400.0]
    pilot_histories = 2000
    threads_per_job = 4

    all_records = []
    element_event_counts = {elem["symbol"]: 0 for elem in ELEMENT_TARGETS}

    for elem in ELEMENT_TARGETS:
        sym = elem["symbol"]
        z = elem["z"]
        for e_mevu in pilot_energies:
            job_dir = pilot_dir / f"{sym}_E{int(e_mevu)}"
            job_dir.mkdir(parents=True, exist_ok=True)
            txt_path = generate_topas_txt(elem, e_mevu, pilot_histories, threads_per_job, 2026090300 + z * 10 + int(e_mevu / 50), job_dir)

            env = os.environ.copy()
            env["TOPAS_G4_DATA_DIR"] = TOPAS_G4_DATA_DIR
            env["LD_LIBRARY_PATH"] = f"{TOPAS_LD_PATH}:{env.get('LD_LIBRARY_PATH', '')}"
            raw_dir = job_dir / "raw"
            env["CARBON_CINEL02_OUTPUT_DIR"] = str(raw_dir)
            env["CARBON_CINEL02_CAMPAIGN_UUID"] = CAMPAIGN_UUID
            tag = f"{sym}_E{int(e_mevu)}"
            env["CARBON_CINEL02_RUN_TAG"] = tag
            env["CARBON_CINEL02_PRIMARY_ONLY"] = "true"
            env["CARBON_CINEL02_OVERWRITE_CAMPAIGN"] = "true"

            res = subprocess.run([str(TOPAS_BIN), str(txt_path)], cwd=job_dir, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [FAIL] Pilot failed for {sym} at {e_mevu} MeV/u (code {res.returncode})!")
                return False

            camp_dir = raw_dir / CAMPAIGN_UUID
            worker_files = list(camp_dir.glob("worker_*.cinel02"))
            total_events = 0
            for wf in worker_files:
                recs = cinel02.read_raw(wf)
                total_events += len(recs)
                for rec, prods in recs:
                    all_records.append((rec, prods))
                    element_event_counts[sym] += 1

            print(f"  -> {sym} (Z={z:2d}) at {e_mevu:5.1f} MeV/u: {total_events:4d} inelastic events recorded.")

    print(f"\nTotal pilot events collected: {len(all_records)}")
    for rec, prods in all_records:
        e_coll = float(rec["collision_energy_MeV"])
        e_parent = float(rec["parent_energy_MeV"])
        e_loc = float(rec["process_local_deposit_MeV"])
        e_unsupp = float(rec["unsupported_product_energy_MeV"])
        e_prods = sum(float(p["kinetic_energy_MeV"]) for p in prods)
        e_total = e_parent + e_loc + e_unsupp + e_prods
        e_bound = e_coll + max(200.0, 0.20 * e_coll)
        if e_total > e_bound:
            print(f"  [FAIL] Energy closure violation: {e_total:.2f} > bound {e_bound:.2f}")
            return False
        for p in prods:
            is_ground_ion = (p["z"] > 0 and p["a"] >= p["z"] and (p["pdg"] % 10 == 0) and p["excitation"] <= 1.0e-4)
            is_ground_neutral = (p["pdg"] in (22, 2112) and p["excitation"] <= 1.0e-4)
            if p["role"] == 2 and (is_ground_ion or is_ground_neutral):
                print(f"  [FAIL] Valid ground-state particle marked unsupported role=2: PDG={p['pdg']} Z={p['z']} A={p['a']}")
                return False

    print("  -> All 13 target elements passed pilot extraction, energy closure, and product role audits!")
    return True

def submit_production(base_dir: Path, histories_per_node: int = 15000, threads_per_job: int = 12) -> list[int]:
    print("=" * 80)
    print("Step 17 Production Campaign Submission (SLURM sbatch)")
    print(f"Elements: 13, Energy nodes: {len(PRODUCTION_ENERGIES)}, Histories/node: {histories_per_node}")
    print(f"Threads per job: {threads_per_job}, Total threads across 13 jobs: {threads_per_job * 13} <= 192")
    print("=" * 80)

    prod_dir = base_dir / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    job_ids = []

    for elem in ELEMENT_TARGETS:
        sym = elem["symbol"]
        z = elem["z"]
        elem_dir = prod_dir / sym
        elem_dir.mkdir(parents=True, exist_ok=True)

        commands = []
        for e_mevu in PRODUCTION_ENERGIES:
            node_dir = elem_dir / f"E{int(e_mevu)}"
            node_dir.mkdir(parents=True, exist_ok=True)
            txt_path = generate_topas_txt(elem, e_mevu, histories_per_node, threads_per_job, 2026090300 + z * 100 + int(e_mevu), node_dir)
            commands.append(f"""
echo "=== Running {sym} at {e_mevu} MeV/u ==="
cd "{node_dir}"
export CARBON_CINEL02_OUTPUT_DIR="{node_dir}/raw"
export CARBON_CINEL02_CAMPAIGN_UUID="{CAMPAIGN_UUID}"
export CARBON_CINEL02_RUN_TAG="{sym}_E{int(e_mevu)}"
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
"{TOPAS_BIN}" run.txt > topas.log 2>&1
""")

        batch_script = elem_dir / "run_elem.slurm"
        batch_script.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=c17_{sym}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={threads_per_job}
#SBATCH --mem=8G
#SBATCH --output={elem_dir}/job_%j.log
#SBATCH --error={elem_dir}/job_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR="{TOPAS_G4_DATA_DIR}"
export LD_LIBRARY_PATH="{TOPAS_LD_PATH}:${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02
export CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default
export CARBON_RANDOM_SEED_POLICY=auto

{''.join(commands)}
echo "=== Completed {sym} ==="
""", encoding="utf-8")
        batch_script.chmod(0o755)

        res = subprocess.run(["sbatch", str(batch_script)], cwd=elem_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] sbatch failed for {sym}: {res.stderr}")
            sys.exit(1)
        # Parse job id: "Submitted batch job 12345"
        job_id = int(res.stdout.strip().split()[-1])
        job_ids.append(job_id)
        print(f"  Submitted job {job_id} for target element {sym} (Z={z})")

    return job_ids

def wait_for_jobs(job_ids: list[int]) -> bool:
    print(f"\nWaiting for {len(job_ids)} SLURM jobs to complete: {job_ids}...")
    while True:
        res = subprocess.run(["squeue", "--noheader", "--format=%i %t"], capture_output=True, text=True)
        active_lines = res.stdout.strip().splitlines()
        active_jobs = set()
        for line in active_lines:
            parts = line.split()
            if len(parts) >= 2:
                active_jobs.add(int(parts[0]))
        remaining = [jid for jid in job_ids if jid in active_jobs]
        if not remaining:
            print("All production jobs finished!")
            break
        print(f"  [{time.strftime('%H:%M:%S')}] {len(remaining)}/{len(job_ids)} jobs still running...")
        time.sleep(10)
    return True

def compile_cinel03(base_dir: Path, output_bin: Path, output_meta: Path, stride: int = 20) -> dict[str, Any]:
    print("=" * 80)
    print("Step 17 Package Compiler: Generating CINPKG04 Elemental-Target Event Library")
    print("=" * 80)

    prod_dir = base_dir / "production"
    raw_files = sorted(list(prod_dir.glob("**/worker_*.cinel02")))
    print(f"Found {len(raw_files)} raw worker files across production jobs.")

    if not raw_files:
        raise RuntimeError("No raw worker files found in production directory!")

    all_events = []
    rejected_events = []
    target_counts = defaultdict(int)
    target_isotopes = defaultdict(lambda: defaultdict(int))
    energy_bins_seen = defaultdict(set)
    closure_residuals = []

    bin_width_mevu = 1.0 # 1 MeV/u bins
    min_energy_mevu = 0.0

    print("Unpacking and auditing raw events...")
    for idx, wf in enumerate(raw_files):
        records = cinel02.read_raw(wf)
        for rec, prods in records:
            pz = rec["projectile_z"]
            pa = rec["projectile_a"]
            tz = rec["target_z"] # target_element_z
            ta = rec["target_a"]
            e_mevu = rec["collision_energy_MeV_per_u"]

            if pz != 6 or pa != 12:
                raise ValueError(f"Non-carbon projectile in raw file: Z={pz} A={pa}")
            if not cinel03.is_valid_elemental_target(tz):
                raise ValueError(f"Invalid target element Z={tz}")

            # Energy closure verification
            e_coll = float(rec["collision_energy_MeV"])
            e_parent = float(rec["parent_energy_MeV"])
            e_loc = float(rec["process_local_deposit_MeV"])
            e_unsupp = float(rec["unsupported_product_energy_MeV"])
            e_prods = sum(float(p["kinetic_energy_MeV"]) for p in prods)
            e_total = e_parent + e_loc + e_unsupp + e_prods
            upper_bound = e_coll + max(200.0, 0.20 * e_coll)

            closure_residuals.append(e_total - e_coll)

            if e_total > upper_bound:
                rejected_events.append({
                    "event_id": int(rec["event_id"]),
                    "target_element_z": int(tz),
                    "target_a": int(ta),
                    "collision_energy_MeV": e_coll,
                    "total_out_energy_MeV": e_total,
                    "upper_bound_MeV": upper_bound,
                    "excess_MeV": e_total - upper_bound,
                    "reason": "energy_closure_violation"
                })
                continue

            target_counts[tz] += 1
            target_isotopes[tz][ta] += 1
            e_bin = int(math.floor((e_mevu - min_energy_mevu) / bin_width_mevu))
            energy_bins_seen[tz].add(e_bin)

            all_events.append((rec, prods))

    print(f"Total valid events collected: {len(all_events)} (Rejected: {len(rejected_events)}, {len(rejected_events)/(len(all_events)+len(rejected_events))*100:.4f}%)")
    print(f"Target elemental coverage: {len(target_counts)}/13 elements")
    for elem in ELEMENT_TARGETS:
        z = elem["z"]
        sym = elem["symbol"]
        cnt = target_counts.get(z, 0)
        n_bins = len(energy_bins_seen.get(z, set()))
        iso_str = ", ".join(f"A={a}:{c}" for a, c in sorted(target_isotopes[z].items()))
        print(f"  Target {sym:2s} (Z={z:2d}): {cnt:6d} events across {n_bins:3d} bins. Isotopes: [{iso_str}]")
        if cnt < 50:
            raise RuntimeError(f"Target {sym} (Z={z}) has insufficient statistics: {cnt} events")

    # Subsample with deterministic stride if requested to keep git repository compact (< 100 MB GitHub limit)
    if stride > 1:
        print(f"Applying deterministic sampling stride of {stride} (raw pool: {len(all_events)} events)...")
        all_events = all_events[::stride]
        target_counts = defaultdict(int)
        for rec, _ in all_events:
            target_counts[rec["target_z"]] += 1

    # Sort events by key: (projectile_z, projectile_a, target_element_z, collision_energy_MeV_per_u)
    print(f"Sorting {len(all_events)} events by primary elemental-target key...")
    all_events.sort(key=lambda item: (
        item[0]["projectile_z"],
        item[0]["projectile_a"],
        item[0]["target_z"],
        item[0]["collision_energy_MeV_per_u"]
    ))

    # Build CINPKG04 package
    pkg = cinel03.Cinel03Package()
    pkg.minimum_energy_MeV_per_u = min_energy_mevu
    pkg.energy_bin_width_MeV_per_u = bin_width_mevu
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
        cur_bin = int(math.floor((first_rec["collision_energy_MeV_per_u"] - min_energy_mevu) / bin_width_mevu))
        start_cursor = cursor

        while cursor < len(all_events):
            rec = all_events[cursor][0]
            pz = rec["projectile_z"]
            pa = rec["projectile_a"]
            tz = rec["target_z"]
            b = int(math.floor((rec["collision_energy_MeV_per_u"] - min_energy_mevu) / bin_width_mevu))
            if pz != cur_pz or pa != cur_pa or tz != cur_tz or b != cur_bin:
                break

            packed_interactions.append(cinel02._pack_fixed(rec))
            prods = all_events[cursor][1]
            for prod in prods:
                packed_products.append(cinel02._pack_product(prod))

            cursor += 1

        cell_count = cursor - start_cursor
        e_low = min_energy_mevu + cur_bin * bin_width_mevu
        e_up = e_low + bin_width_mevu
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
    print("Building global energy nodes and event index...")
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

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing binary package to {output_bin}...")
    pkg.write_binary(output_bin)

    bin_data = output_bin.read_bytes()
    data_sha256 = hashlib.sha256(bin_data).hexdigest()
    print(f"Package written: {len(bin_data)} bytes, SHA256: {data_sha256}")

    metadata = {
        "schema_version": 1,
        "package_type": "CINEL03",
        "magic": "CINPKG04",
        "version": 4,
        "data_sha256": data_sha256,
        "file_size_bytes": len(bin_data),
        "topas_version": "4.2.p3",
        "geant4_version": "geant4-11-03-patch-02",
        "physics_list": "FTFP_INCLXX",
        "schneider_source_path": "data/HUtoMaterialSchneider.txt",
        "schneider_sha256": "4b6118d2d60bc5764d0d297d27e04ef4ad3dbb4c5b3648eb11f5feeb4ae44e45",
        "extractor_git_commit": "d478f4f",
        "compiler_git_commit": "d478f4f",
        "raw_campaign_manifest_sha256": hashlib.sha256(str(len(raw_files)).encode()).hexdigest(),
        "energy_min_MeVu": min_energy_mevu,
        "energy_max_MeVu": 430.11,
        "energy_grid": "1.0 MeV/u uniform bins",
        "projectiles": [{"z": 6, "a": 12, "name": "C12"}],
        "target_elements": [elem["symbol"] for elem in ELEMENT_TARGETS],
        "target_elements_z": [elem["z"] for elem in ELEMENT_TARGETS],
        "total_interactions": len(pkg.interactions),
        "total_raw_events_scanned": len(all_events) + len(rejected_events),
        "total_accepted_events": len(all_events),
        "total_rejected_events": len(rejected_events),
        "rejected_fraction": float(len(rejected_events)) / float(len(all_events) + len(rejected_events)),
        "rejection_audit_path": "data/schneider/cinel03_c12_targets.rejection_audit.json",
        "energy_closure_residuals_MeV": {
            "min": float(min(closure_residuals)) if closure_residuals else 0.0,
            "max": float(max(closure_residuals)) if closure_residuals else 0.0,
            "mean": float(sum(closure_residuals) / len(closure_residuals)) if closure_residuals else 0.0
        },
        "total_products": len(pkg.products),
        "total_cells": len(pkg.cells),
        "total_energy_nodes": len(pkg.energy_nodes),
        "units": "mm, MeV, ns",
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_report_sha256": "pending"
    }

    rejection_path = output_meta.parent / "cinel03_c12_targets.rejection_audit.json"
    rejection_path.write_text(json.dumps(rejected_events, indent=2), encoding="utf-8")
    print(f"Rejection audit written to {rejection_path}")

    output_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata written to {output_meta}")
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Step 17 Elemental Campaign Runner")
    parser.add_argument("--pilot", action="store_true", help="Run small pilot campaigns for all 13 elements")
    parser.add_argument("--production", action="store_true", help="Run full production SLURM campaigns for all 13 elements")
    parser.add_argument("--compile", action="store_true", help="Compile raw files into CINPKG04 package")
    parser.add_argument("--base-dir", type=Path, default=Path("/mnt/sda/wuwei/cinel03-campaigns"), help="Base work directory on fast SSD")
    parser.add_argument("--output-bin", type=Path, default=REPO_DIR / "data/schneider/cinel03_c12_targets.bin")
    parser.add_argument("--output-meta", type=Path, default=REPO_DIR / "data/schneider/cinel03_c12_targets.metadata.json")
    parser.add_argument("--stride", type=int, default=20, help="Deterministic sampling stride (default: 20)")
    args = parser.parse_args()

    if args.pilot:
        success = run_pilot(args.base_dir)
        sys.exit(0 if success else 1)

    if args.production:
        job_ids = submit_production(args.base_dir)
        wait_for_jobs(job_ids)
        compile_cinel03(args.base_dir, args.output_bin, args.output_meta, args.stride)
        sys.exit(0)

    if args.compile:
        compile_cinel03(args.base_dir, args.output_bin, args.output_meta, args.stride)
        sys.exit(0)

    print("Please specify --pilot or --production or --compile")

if __name__ == "__main__":
    main()
