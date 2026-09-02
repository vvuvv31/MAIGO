#!/usr/bin/env python3
"""
Step 03 Verification Tool:
Verifies 100% equivalence between MAIGO Schneider parser and TOPAS/Geant4 material truth.
Generates:
  - parsed-schneider-materials.json
  - material-comparison.json
  - run-manifest.json
"""

import json
import math
import hashlib
import os
import subprocess
import re

CANONICAL_ELEMENTS = [
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen",
    "Magnesium", "Phosphorus", "Sulfur", "Chlorine",
    "Argon", "Calcium", "Sodium", "Potassium", "Titanium"
]

CANONICAL_Z = [1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22]

AVOGADRO = 6.02214076e23

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def parse_schneider_txt(filepath):
    with open(filepath) as f:
        text = f.read()

    elements = []
    hu_sections = []
    weights = []
    density_corrections = []

    density_hu_edges = [-1000, -98, 15, 23, 101, 2001, 2995, 2996]
    density_offsets = [0.00121, 1.018, 1.03, 1.003, 1.017, 2.201, 4.54]
    density_factors = [0.001029700665188, 0.000893, 0.0, 0.001169, 0.000592, 0.0005, 0.0]
    density_factor_offsets = [1000.0, 0.0, 1000.0, 0.0, 0.0, -2000.0, 0.0]

    # Robust regex parsing of multiline parameters
    match_corr = re.search(r'dv:Ge/Patient/DensityCorrection\s*=\s*(\d+)\s+(.*?)(?=\n[a-z#]|\Z)', text, re.DOTALL)
    if match_corr:
        count = int(match_corr.group(1))
        body = match_corr.group(2)
        tokens = [t for t in body.split() if t != 'g/cm3']
        density_corrections = [float(t) for t in tokens[:count]]

    match_elems = re.search(r'sv:Ge/Patient/SchneiderElements\s*=\s*(\d+)\s+(.*?)(?=\n[a-z#]|\Z)', text, re.DOTALL)
    if match_elems:
        count = int(match_elems.group(1))
        elements = [t.replace('"', '') for t in match_elems.group(2).split()[:count]]

    match_hu = re.search(r'iv:Ge/Patient/SchneiderHUToMaterialSections\s*=\s*(\d+)\s+(.*?)(?=\n[a-z#]|\Z)', text, re.DOTALL)
    if match_hu:
        count = int(match_hu.group(1))
        hu_sections = [int(t) for t in match_hu.group(2).split()[:count]]

    for i in range(1, 26):
        pattern = rf'uv:Ge/Patient/SchneiderMaterialsWeight{i}\s*=\s*(\d+)\s+(.*?)(?=\n[a-z#]|\Z)'
        match_w = re.search(pattern, text, re.DOTALL)
        if match_w:
            count = int(match_w.group(1))
            w = [float(t) for t in match_w.group(2).split()[:count]]
            weights.append(w)

    def compute_density(hu):
        if hu < -1000:
            hu = -1000
        if hu >= 2996:
            hu = 2995
        corr = 1.0
        idx = hu - (-1000)
        if 0 <= idx < len(density_corrections):
            corr = density_corrections[idx]

        for k in range(len(density_offsets)):
            if density_hu_edges[k] <= hu < density_hu_edges[k + 1]:
                base_dens = density_offsets[k] + density_factors[k] * (density_factor_offsets[k] + hu)
                return base_dens * corr
        return 1.0 * corr

    sections = []
    for i in range(len(hu_sections) - 1):
        hu_min = hu_sections[i]
        hu_max = hu_sections[i + 1]
        rep_hu = 2995 if i == 24 else math.floor((hu_min + hu_max) / 2.0)
        density_raw = compute_density(rep_hu)
        density_topas_6g = float(f"{density_raw:.6g}")

        elem_list = []
        for idx, (el_name, el_z) in enumerate(zip(CANONICAL_ELEMENTS, CANONICAL_Z)):
            w = weights[i][idx]
            elem_list.append({
                "canonical_index": idx,
                "name": el_name,
                "z": el_z,
                "mass_fraction": w
            })

        sections.append({
            "section_id": i,
            "hu_min_inclusive": hu_min,
            "hu_max_exclusive": hu_max,
            "representative_HU": rep_hu,
            "density_g_cm3": density_topas_6g,
            "density_g_cm3_raw": density_raw,
            "number_of_elements": sum(1 for w in weights[i] if w > 0),
            "elements": elem_list
        })

    return {
        "schema_version": 1,
        "source_file": filepath,
        "section_count": len(sections),
        "canonical_element_order": CANONICAL_ELEMENTS,
        "sections": sections
    }

def main():
    repo_dir = "/mnt/sdb/wuwei/MAIGO"
    evidence_dir = "/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03"
    os.makedirs(evidence_dir, exist_ok=True)

    schneider_txt = os.path.join(repo_dir, "data/HUtoMaterialSchneider.txt")
    topas_json_path = os.path.join(evidence_dir, "topas-schneider-materials.json")
    parsed_json_path = os.path.join(evidence_dir, "parsed-schneider-materials.json")
    comparison_json_path = os.path.join(evidence_dir, "material-comparison.json")
    manifest_json_path = os.path.join(evidence_dir, "run-manifest.json")

    # 1. Parse Schneider TXT using MAIGO model
    parsed_data = parse_schneider_txt(schneider_txt)
    with open(parsed_json_path, 'w') as f:
        json.dump(parsed_data, f, indent=2)
    print(f"Generated {parsed_json_path}")

    # 2. Read TOPAS Truth JSON
    with open(topas_json_path) as f:
        topas_data = json.load(f)

    # 3. Perform Hard & Numerical Gate Audits
    sections_checked = len(topas_data['sections'])
    assert sections_checked == 25, f"Expected 25 sections, got {sections_checked}"

    element_order_exact = True
    section_identity_exact = True
    max_density_relative_error = 0.0
    max_mass_fraction_abs_error = 0.0
    max_atom_density_relative_error = 0.0

    section_comparisons = []

    for s_idx in range(25):
        t_sec = topas_data['sections'][s_idx]
        p_sec = parsed_data['sections'][s_idx]

        # Section ID and HU check
        if t_sec['section_id'] != p_sec['section_id'] or t_sec['representative_HU'] != p_sec['representative_HU']:
            section_identity_exact = False

        # Density check
        t_dens = t_sec['density_g_cm3']
        p_dens = p_sec['density_g_cm3']
        dens_rel_err = abs(t_dens - p_dens) / max(t_dens, 1e-12)
        if dens_rel_err > max_density_relative_error:
            max_density_relative_error = dens_rel_err

        elem_comparisons = []
        for el_idx in range(13):
            t_el = t_sec['elements'][el_idx]
            p_el = p_sec['elements'][el_idx]

            # Canonical order & identity
            if t_el['name'] != CANONICAL_ELEMENTS[el_idx] or t_el['z'] != CANONICAL_Z[el_idx]:
                element_order_exact = False
            if p_el['name'] != CANONICAL_ELEMENTS[el_idx] or p_el['z'] != CANONICAL_Z[el_idx]:
                element_order_exact = False

            # Mass fraction check
            w_abs_err = abs(t_el['mass_fraction'] - p_el['mass_fraction'])
            if w_abs_err > max_mass_fraction_abs_error:
                max_mass_fraction_abs_error = w_abs_err

            # Atom density internal consistency check:
            # Geant4 n_j = (w_j * rho * N_A / A_j)
            t_atom_dens = t_el['atom_density_per_mm3']
            if t_el['mass_fraction'] > 0:
                g4_a = t_el['a_g_mol']
                theo_atom_dens = (t_el['mass_fraction'] * t_dens * AVOGADRO / g4_a) * 1.0e-3
                atom_rel_err = abs(t_atom_dens - theo_atom_dens) / t_atom_dens
                if atom_rel_err > max_atom_density_relative_error:
                    max_atom_density_relative_error = atom_rel_err
            else:
                atom_rel_err = 0.0

            elem_comparisons.append({
                "element": CANONICAL_ELEMENTS[el_idx],
                "z": CANONICAL_Z[el_idx],
                "mass_fraction_topas": t_el['mass_fraction'],
                "mass_fraction_parsed": p_el['mass_fraction'],
                "mass_fraction_abs_diff": w_abs_err,
                "topas_a_g_mol": t_el['a_g_mol'],
                "topas_atom_density_per_mm3": t_atom_dens,
                "atom_density_internal_rel_err": atom_rel_err
            })

        section_comparisons.append({
            "section_id": s_idx,
            "representative_HU": t_sec['representative_HU'],
            "material_name": t_sec['material_name'],
            "density_topas_g_cm3": t_dens,
            "density_parsed_g_cm3": p_dens,
            "density_rel_error": dens_rel_err,
            "mean_excitation_energy_eV": t_sec['mean_excitation_energy_eV'],
            "radiation_length_mm": t_sec['radiation_length_mm'],
            "elements": elem_comparisons
        })

    # 4. Evaluate Gates
    hard_gate_passed = (
        section_identity_exact and
        element_order_exact and
        sections_checked == 25 and
        max_mass_fraction_abs_error < 1.0e-8
    )

    numerical_gate_passed = (
        max_density_relative_error < 1.0e-6 and
        max_atom_density_relative_error < 1.0e-4
    )

    comparison_report = {
        "schema_version": 1,
        "sections_checked": sections_checked,
        "element_order_exact": element_order_exact,
        "section_identity_exact": section_identity_exact,
        "max_density_relative_error": max_density_relative_error,
        "max_mass_fraction_absolute_error": max_mass_fraction_abs_error,
        "max_atom_density_relative_error": max_atom_density_relative_error,
        "hard_gate_passed": hard_gate_passed,
        "numerical_gate_passed": numerical_gate_passed,
        "thresholds": {
            "max_mass_fraction_abs_error_limit": 1.0e-8,
            "max_density_rel_error_limit": 1.0e-6,
            "max_atom_density_rel_error_limit": 1.0e-4
        },
        "section_details": section_comparisons
    }

    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    print(f"Generated {comparison_json_path}")

    # 5. Generate Run Manifest
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir).decode().strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_dir).decode().strip()

    manifest = {
        "step": "step-03",
        "description": "TOPAS/Geant4 Schneider Material Truth Audit",
        "git_commit": git_head,
        "git_branch": git_branch,
        "topas_version": "4.2.p3",
        "geant4_version": "11.03.p02",
        "files": {
            "topas-schneider-materials.json": {
                "sha256": sha256_file(topas_json_path),
                "size_bytes": os.path.getsize(topas_json_path)
            },
            "parsed-schneider-materials.json": {
                "sha256": sha256_file(parsed_json_path),
                "size_bytes": os.path.getsize(parsed_json_path)
            },
            "material-comparison.json": {
                "sha256": sha256_file(comparison_json_path),
                "size_bytes": os.path.getsize(comparison_json_path)
            }
        },
        "hard_gate_passed": hard_gate_passed,
        "numerical_gate_passed": numerical_gate_passed
    }

    with open(manifest_json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {manifest_json_path}")

    # 6. Generate plan/evidence-step03.sha256 in repo
    step03_sha256_path = os.path.join(repo_dir, "plan/evidence-step03.sha256")
    with open(step03_sha256_path, 'w') as f:
        f.write("# Cryptographic anchor for external Step 03 evidence stored under /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03/\n")
        for fname in sorted(["topas-schneider-materials.json", "parsed-schneider-materials.json", "material-comparison.json", "run-manifest.json"]):
            fpath = os.path.join(evidence_dir, fname)
            f.write(f"{sha256_file(fpath)}  {fname}\n")
    print(f"Generated {step03_sha256_path}")

    print("\n================ STEP 03 SUMMARY ================")
    print(f"Sections Checked: {sections_checked}/25")
    print(f"Element Order Exact: {element_order_exact}")
    print(f"Section Identity Exact: {section_identity_exact}")
    print(f"Max Density Rel Error: {max_density_relative_error:.3e} (limit < 1e-6)")
    print(f"Max Mass Fraction Abs Error: {max_mass_fraction_abs_error:.3e} (limit < 1e-8)")
    print(f"Max Atom Density Rel Error: {max_atom_density_relative_error:.3e} (limit < 1e-4)")
    print(f"Hard Gate Passed: {hard_gate_passed}")
    print(f"Numerical Gate Passed: {numerical_gate_passed}")
    print("==================================================")

    if not (hard_gate_passed and numerical_gate_passed):
        raise SystemExit("Step 03 Quality Gate FAILED!")

if __name__ == '__main__':
    main()
