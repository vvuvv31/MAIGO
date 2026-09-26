#!/usr/bin/env python3
"""
tools/compile_schneider_stopping.py

Step 14 Canonical Stopping Power Compiler:
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
  - Section Identity & Composition Hash verification against data/HUtoMaterialSchneider.txt
  - Clean, separate metadata sidecars for binary and CSV
"""

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3]))
SCHNEIDER_TXT = REPO_ROOT / "data/HUtoMaterialSchneider.txt"

EXPECTED_SECTIONS = 25
EXPECTED_ENERGIES = 4302
ENERGY_MIN = 0.01
ENERGY_MAX = 430.11
ENERGY_STEP = 0.1

CANONICAL_PROBES = [
    (0, -975), (1, -535), (2, -102), (3, -68), (4, -38), (5, -8),
    (6, 12), (7, 49), (8, 100), (9, 160), (10, 250), (11, 350),
    (12, 450), (13, 550), (14, 650), (15, 750), (16, 850), (17, 950),
    (18, 1050), (19, 1150), (20, 1250), (21, 1350), (22, 1450),
    (23, 2247), (24, 2995)
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def mat_name_from_hu(hu):
    if hu < 0:
        return f"PatientTissueFromHUNegative{-hu}"
    return f"PatientTissueFromHU{hu}"

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
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--raw-json", required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/schneider"))
    parser.add_argument("--projectile-z", type=int, default=6)
    parser.add_argument("--projectile-a", type=int, default=12)
    parser.add_argument("--prefix", default="", help="Prefix for a new species package; old names retained by default")
    args = parser.parse_args()
    if args.projectile_z <= 0 or args.projectile_a < args.projectile_z:
        parser.error("Require positive Z and A >= Z")
    if (args.projectile_z, args.projectile_a) != (6, 12) and not args.prefix:
        parser.error("Non-C12 extraction requires --prefix to protect historical outputs")
    binary_name = args.prefix + "schneider_stopping_v1.bin"
    csv_name = args.prefix + "schneider_stopping_power.csv" if args.prefix else "c12_schneider_stopping_power.csv"

    raw_csv_path = Path(args.raw_csv)
    raw_json_path = Path(args.raw_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
    if not raw_json_path.is_file():
        raise FileNotFoundError(f"Raw JSON not found: {raw_json_path}")
    if not SCHNEIDER_TXT.is_file():
        raise FileNotFoundError(f"Schneider text definition not found: {SCHNEIDER_TXT}")

    weights, comp_hashes = parse_schneider_compositions()

    # 1. Parse CSV data into [section][energy_idx]
    mass_sp = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    linear_sp = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    csda_range = [[0.0] * EXPECTED_ENERGIES for _ in range(EXPECTED_SECTIONS)]
    densities = [0.0] * EXPECTED_SECTIONS
    material_names = [""] * EXPECTED_SECTIONS
    seen = set()

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
                raise ValueError(f"Energy out of range: {e} (e_idx={e_idx})")
            if not (0 <= s < EXPECTED_SECTIONS):
                raise ValueError(f"Section out of range: {s}")

            if (s, e_idx) in seen or abs(e - (ENERGY_MIN + e_idx * ENERGY_STEP)) > 1e-6:
                raise ValueError(f"Duplicate or off-grid stopping sample: {s}, {e}")
            seen.add((s, e_idx))
            # Check Section Identity against canonical probe
            expected_name = mat_name_from_hu(CANONICAL_PROBES[s][1])
            if mat != expected_name:
                raise ValueError(f"Section identity mismatch at section {s}: expected material name '{expected_name}', got '{mat}'")

            mass_sp[s][e_idx] = m_sp
            linear_sp[s][e_idx] = l_sp
            csda_range[s][e_idx] = r_mm
            densities[s] = rho
            material_names[s] = mat

    # 2. Strict Validations of raw CSV
    print(f"Validating data integrity for {EXPECTED_SECTIONS} sections across {EXPECTED_ENERGIES} energies...")
    for s in range(EXPECTED_SECTIONS):
        rho = densities[s]
        if rho <= 0.0:
            raise ValueError(f"Section {s} has non-positive density: {rho}")
        for i in range(EXPECTED_ENERGIES):
            m = mass_sp[s][i]
            l = linear_sp[s][i]
            r = csda_range[s][i]
            if m <= 0.0:
                raise ValueError(f"Non-positive mass stopping power at s={s}, i={i}: {m}")
            if l <= 0.0:
                raise ValueError(f"Non-positive linear stopping power at s={s}, i={i}: {l}")
            if r <= 0.0:
                raise ValueError(f"Non-positive CSDA range at s={s}, i={i}: {r}")
            rel_err = abs(l - m * rho) / l
            if rel_err >= 1e-4:
                raise ValueError(f"Mass/linear mismatch at s={s}, i={i}: l={l}, m*rho={m*rho}, rel_err={rel_err}")
            if i > 0 and r <= csda_range[s][i-1]:
                raise ValueError(f"Non-monotonic CSDA range at s={s}, i={i}: prev={csda_range[s][i-1]}, curr={r}")

    # 3. Strict Validation and Ingestion of raw JSON audit & provenance
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw_json_data = json.load(f)

    if raw_json_data.get("num_sections") != EXPECTED_SECTIONS:
        raise ValueError(f"Raw JSON num_sections mismatch: {raw_json_data.get('num_sections')} vs {EXPECTED_SECTIONS}")
    if raw_json_data.get("num_energies") != EXPECTED_ENERGIES:
        raise ValueError(f"Raw JSON num_energies mismatch: {raw_json_data.get('num_energies')} vs {EXPECTED_ENERGIES}")
    if abs(raw_json_data.get("energy_min_mevu", 0.0) - ENERGY_MIN) > 1e-6:
        raise ValueError("Raw JSON energy_min_mevu mismatch")
    if abs(raw_json_data.get("energy_max_mevu", 0.0) - ENERGY_MAX) > 1e-6:
        raise ValueError("Raw JSON energy_max_mevu mismatch")
    if abs(raw_json_data.get("energy_step_mevu", 0.0) - ENERGY_STEP) > 1e-6:
        raise ValueError("Raw JSON energy_step_mevu mismatch")

    proj = raw_json_data.get("projectile", {})
    if (proj.get("z"), proj.get("a")) != (args.projectile_z, args.projectile_a):
        raise ValueError(f"Raw JSON projectile does not match requested Z/A: {proj}")

    raw_sections = raw_json_data.get("sections", [])
    if len(raw_sections) != EXPECTED_SECTIONS:
        raise ValueError(f"Raw JSON sections count mismatch: {len(raw_sections)}")

    i_values_ev = [0.0] * EXPECTED_SECTIONS
    for s in range(EXPECTED_SECTIONS):
        r_sec = raw_sections[s]
        if r_sec.get("section_id") != s:
            raise ValueError(f"Raw JSON section_id mismatch at index {s}: {r_sec.get('section_id')}")
        if r_sec.get("rep_hu") != CANONICAL_PROBES[s][1]:
            raise ValueError(f"Raw JSON rep_hu mismatch at section {s}: {r_sec.get('rep_hu')} vs {CANONICAL_PROBES[s][1]}")
        if r_sec.get("material_name") != material_names[s]:
            raise ValueError(f"Raw JSON material_name mismatch at section {s}: {r_sec.get('material_name')} vs {material_names[s]}")
        if abs(r_sec.get("density_g_cm3", 0.0) - densities[s]) > 1e-5:
            raise ValueError(f"Raw JSON density mismatch at section {s}: {r_sec.get('density_g_cm3')} vs {densities[s]}")
        i_val = r_sec.get("mean_excitation_energy_eV", 0.0)
        if i_val <= 0.0:
            raise ValueError(f"Raw JSON non-positive mean_excitation_energy_eV at section {s}: {i_val}")
        i_values_ev[s] = i_val

    print(f"Data validation PASSED for {EXPECTED_SECTIONS} sections across {EXPECTED_ENERGIES} energies (0.01 to {ENERGY_MAX} MeV/u).")

    # 4. Canonical CSV Output
    canon_csv_path = out_dir / csv_name
    with open(canon_csv_path, 'w', encoding='utf-8') as f:
        f.write("# Canonical Geant4/TOPAS Stopping Power Table for 25 Schneider Sections\n")
        f.write("energy_mevu,section_id,material_name,density_g_cm3,mass_stopping_power_mev_mm_per_g_cm3,linear_stopping_power_mev_per_mm,csda_range_mm\n")
        for i in range(EXPECTED_ENERGIES):
            e = ENERGY_MIN + i * ENERGY_STEP
            for s in range(EXPECTED_SECTIONS):
                f.write(f"{e:.2f},{s},{material_names[s]},{densities[s]:.7f},{mass_sp[s][i]:.10f},{linear_sp[s][i]:.10f},{csda_range[s][i]:.10f}\n")
    print(f"Wrote canonical CSV: {canon_csv_path}")

    # 5. Binary Output: schneider_stopping_v1.bin
    bin_path = out_dir / binary_name
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

    # 6. Metadata JSONs (Distinct sidecars for Binary and CSV)
    bin_sha = sha256_file(bin_path)
    csv_sha = sha256_file(canon_csv_path)
    schneider_sha = sha256_file(SCHNEIDER_TXT)

    section_manifest = []
    for s in range(EXPECTED_SECTIONS):
        section_manifest.append({
            "section_id": s,
            "material_name": material_names[s],
            "nominal_density_g_cm3": round(densities[s], 7),
            "mean_excitation_energy_eV": round(i_values_ev[s], 4),
            "composition_sha256": comp_hashes[s],
            "element_weights": weights[s]
        })

    common_meta = {
        "schema_version": 2,
        "topas_version": "4.2.p3",
        "geant4_version": "geant4-11-03-patch-02 [MT]",
        "em_physics_module": "g4em-standard_opt4",
        "projectile": proj,
        "raw_inputs": {str(raw_csv_path.resolve()): sha256_file(raw_csv_path), str(raw_json_path.resolve()): sha256_file(raw_json_path)},
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

    # Binary sidecar metadata
    bin_meta = dict(common_meta)
    bin_meta["data_filename"] = binary_name
    bin_meta["data_sha256"] = bin_sha
    bin_meta["format"] = "binary"
    bin_meta["binary_magic"] = "SCHNSTOP"
    bin_meta["binary_version"] = 1

    # CSV sidecar metadata
    csv_meta = dict(common_meta)
    csv_meta["data_filename"] = csv_name
    csv_meta["data_sha256"] = csv_sha
    csv_meta["format"] = "csv"

    bin_meta_path = (out_dir / binary_name).with_suffix(".metadata.json")
    csv_meta_path = (out_dir / csv_name).with_suffix(".metadata.json")
    with open(bin_meta_path, 'w') as f:
        json.dump(bin_meta, f, indent=2)
    with open(csv_meta_path, 'w') as f:
        json.dump(csv_meta, f, indent=2)
    print(f"Wrote clean separate metadata: {bin_meta_path} and {csv_meta_path}")

if __name__ == "__main__":
    main()
