#!/usr/bin/env python3
"""
tools/compile_schneider_stopping.py

Step 14 Compiler:
Compiles raw TOPAS/Geant4 C12 stopping-power extraction into official repository data products:
  1. data/schneider/c12_schneider_stopping_power.csv + .metadata.json
  2. data/schneider/schneider_stopping_v1.bin + .metadata.json

Enforces:
  - Byte-for-byte determinism
  - Strict canonical ordering: [section, energy]
  - Exact 25 sections, 4302 uniform energy nodes (0.01 to 430.11 MeV/u, step 0.1)
  - Full domain coverage up to 430.11 MeV/u (supporting 430.0 MeV/u source domain plus guard node)
  - Unit consistency check: linear_stopping == mass_stopping * density within 1e-4 relative
  - Monotonicity of energy and CSDA range
  - Strict Section Provenance & Composition Hash binding to data/HUtoMaterialSchneider.txt
"""

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/sdb/wuwei/MAIGO")
SCHNEIDER_TXT = REPO_ROOT / "data/HUtoMaterialSchneider.txt"

EXPECTED_SECTIONS = 25
EXPECTED_ENERGIES = 4302
ENERGY_MIN = 0.01
ENERGY_MAX = 430.11
ENERGY_STEP = 0.1

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def parse_schneider_compositions():
    schneider_txt = SCHNEIDER_TXT.read_text()
    weights = []
    for line in schneider_txt.splitlines():
        if "SchneiderMaterialsWeight" in line:
            parts = line.split("=")[1].strip().split()
            n = int(parts[0])
            w = [float(x) for x in parts[1:1+n]]
            weights.append(w)
    if len(weights) != EXPECTED_SECTIONS:
        raise ValueError(f"Expected {EXPECTED_SECTIONS} material weights in Schneider file, found {len(weights)}")
    
    comp_hashes = []
    for i, w in enumerate(weights):
        h = hashlib.sha256(",".join(f"{x:.6f}" for x in w).encode('utf-8')).hexdigest()
        comp_hashes.append(h)
    return weights, comp_hashes

def main():
    parser = argparse.ArgumentParser(description="Compile Schneider stopping-power tables.")
    parser.add_argument("--raw-csv", default="/mnt/sda/wuwei/step14_schneider_stopping/raw/topas_c12_schneider_stopping.csv")
    parser.add_argument("--raw-json", default="/mnt/sda/wuwei/step14_schneider_stopping/raw/topas_c12_schneider_stopping.json")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/schneider"))
    args = parser.parse_args()

    raw_csv_path = Path(args.raw_csv)
    raw_json_path = Path(args.raw_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv_path.is_file():
        print(f"Error: Raw CSV not found: {raw_csv_path}")
        sys.exit(1)
    if not raw_json_path.is_file():
        print(f"Error: Raw JSON not found: {raw_json_path}")
        sys.exit(1)

    weights, comp_hashes = parse_schneider_compositions()

    with open(raw_json_path) as f:
        raw_meta = json.load(f)

    # 1. Parse CSV data into [section][energy_idx]
    # Header: energy_mevu,section_id,material_name,density_g_cm3,mass_stopping_power_mev_mm_per_g_cm3,linear_stopping_power_mev_per_mm,csda_range_mm
    mass_sp = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    linear_sp = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    csda_range = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    densities = [0.0] * EXPECTED_SECTIONS
    material_names = [""] * EXPECTED_SECTIONS

    with open(raw_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == "energy_mevu":
                continue
            e = float(row[0])
            s = int(row[1])
            mat = row[2]
            rho = float(row[3])
            m_sp = float(row[4])
            l_sp = float(row[5])
            r_mm = float(row[6])

            e_idx = int(round((e - ENERGY_MIN) / ENERGY_STEP))
            if not (0 <= e_idx < EXPECTED_ENERGIES):
                print(f"Error: energy out of range: {e} (e_idx={e_idx})")
                sys.exit(1)
            if not (0 <= s < EXPECTED_SECTIONS):
                print(f"Error: section out of range: {s}")
                sys.exit(1)

            mass_sp[s][e_idx] = m_sp
            linear_sp[s][e_idx] = l_sp
            csda_range[s][e_idx] = r_mm
            densities[s] = rho
            material_names[s] = mat

    # 2. Strict Validations
    print(f"Validating data integrity for {EXPECTED_SECTIONS} sections across {EXPECTED_ENERGIES} energies...")
    for s in range(EXPECTED_SECTIONS):
        rho = densities[s]
        assert rho > 0.0, f"Section {s} has non-positive density"
        for i in range(EXPECTED_ENERGIES):
            m = mass_sp[s][i]
            l = linear_sp[s][i]
            r = csda_range[s][i]
            assert m > 0.0, f"Non-positive mass stopping power at s={s}, i={i}"
            assert l > 0.0, f"Non-positive linear stopping power at s={s}, i={i}"
            assert r > 0.0, f"Non-positive CSDA range at s={s}, i={i}"
            # Linear == mass * density within 1e-4 relative
            rel_err = abs(l - m * rho) / l
            assert rel_err < 1e-4, f"Mass/linear mismatch at s={s}, i={i}: l={l}, m*rho={m*rho}, rel_err={rel_err}"
            # Monotonicity of range
            if i > 0:
                assert r > csda_range[s][i-1], f"Non-monotonic CSDA range at s={s}, i={i}"

    print(f"Data validation PASSED for {EXPECTED_SECTIONS} sections across {EXPECTED_ENERGIES} energies (0.01 to {ENERGY_MAX} MeV/u).")

    # 3. Canonical CSV Output
    canon_csv_path = out_dir / "c12_schneider_stopping_power.csv"
    with open(canon_csv_path, 'w', encoding='utf-8') as f:
        f.write("# Canonical Geant4/TOPAS C12 Stopping Power Table for 25 Schneider Sections\n")
        f.write("energy_mevu,section_id,material_name,density_g_cm3,mass_stopping_power_mev_mm_per_g_cm3,linear_stopping_power_mev_per_mm,csda_range_mm\n")
        for i in range(EXPECTED_ENERGIES):
            e = ENERGY_MIN + i * ENERGY_STEP
            for s in range(EXPECTED_SECTIONS):
                f.write(f"{e:.2f},{s},{material_names[s]},{densities[s]:.7f},{mass_sp[s][i]:.10f},{linear_sp[s][i]:.10f},{csda_range[s][i]:.10f}\n")
    print(f"Wrote canonical CSV: {canon_csv_path}")

    # 4. Binary Output: schneider_stopping_v1.bin
    # Magic: 8 bytes 'SCHNSTOP'
    # Header:
    #   uint32 version(1)
    #   uint32 num_sections(25)
    #   uint32 num_energies(4302)
    #   double energy_min(0.01)
    #   double energy_max(430.11)
    #   double energy_step(0.1)
    #   double section_densities[25]
    # Payload 1:
    #   double mass_stopping_power[25][4302]
    # Payload 2:
    #   double csda_range_mm[25][4302]
    bin_path = out_dir / "schneider_stopping_v1.bin"
    with open(bin_path, 'wb') as f:
        f.write(b"SCHNSTOP")
        header = struct.pack("<IIIddd", 1, EXPECTED_SECTIONS, EXPECTED_ENERGIES, ENERGY_MIN, ENERGY_MAX, ENERGY_STEP)
        f.write(header)
        for s in range(EXPECTED_SECTIONS):
            f.write(struct.pack("<d", densities[s]))
        # Payload 1: mass stopping power [25][4302]
        for s in range(EXPECTED_SECTIONS):
            for i in range(EXPECTED_ENERGIES):
                f.write(struct.pack("<d", mass_sp[s][i]))
        # Payload 2: CSDA range [25][4302]
        for s in range(EXPECTED_SECTIONS):
            for i in range(EXPECTED_ENERGIES):
                f.write(struct.pack("<d", csda_range[s][i]))
    print(f"Wrote canonical binary: {bin_path} ({bin_path.stat().st_size} bytes)")

    # 5. Metadata JSONs with full Section Identity & Composition Provenance
    bin_sha = sha256_file(bin_path)
    csv_sha = sha256_file(canon_csv_path)
    schneider_sha = sha256_file(SCHNEIDER_TXT)

    section_manifest = []
    for s in range(EXPECTED_SECTIONS):
        section_manifest.append({
            "section_id": s,
            "material_name": material_names[s],
            "nominal_density_g_cm3": densities[s],
            "composition_sha256": comp_hashes[s],
            "element_weights": weights[s]
        })

    meta_dict = {
        "schema_version": 2,
        "data_filename": "schneider_stopping_v1.bin",
        "data_sha256": bin_sha,
        "canonical_csv_filename": "c12_schneider_stopping_power.csv",
        "canonical_csv_sha256": csv_sha,
        "binary_magic": "SCHNSTOP",
        "binary_version": 1,
        "topas_version": "4.2.p3",
        "geant4_version": "geant4-11-03-patch-02 [MT]",
        "em_physics_module": "g4em-standard_opt4",
        "projectile": {"name": "GenericIon(6,12)", "z": 6, "a": 12},
        "schneider_source_path": "data/HUtoMaterialSchneider.txt",
        "schneider_source_sha256": schneider_sha,
        "sections_count": EXPECTED_SECTIONS,
        "energies_count": EXPECTED_ENERGIES,
        "energy_grid": {
            "type": "uniform",
            "min_MeVu": ENERGY_MIN,
            "max_MeVu": ENERGY_MAX,
            "step_MeVu": ENERGY_STEP,
            "nodes": EXPECTED_ENERGIES,
            "covers_430mevu_domain": True,
            "has_guard_node": True
        },
        "section_manifest": section_manifest,
        "dimension_order": {
            "payload_1": ["section_id (0..24)", "energy_node (0..4301)"],
            "payload_2": ["section_id (0..24)", "energy_node (0..4301)"]
        },
        "units": {
            "mass_stopping_power": "(MeV/mm) / (g/cm3)",
            "csda_range": "mm",
            "density": "g/cm3",
            "energy": "MeV/u"
        }
    }

    bin_meta_path = out_dir / "schneider_stopping_v1.metadata.json"
    csv_meta_path = out_dir / "c12_schneider_stopping_power.metadata.json"
    with open(bin_meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)
    with open(csv_meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)
    print(f"Wrote metadata: {bin_meta_path} and {csv_meta_path}")

if __name__ == "__main__":
    main()
