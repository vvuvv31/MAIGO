#!/usr/bin/env python3
"""
Step 07 Verification & Audit Tool:
Audits compiled Schneider C12 rate products:
  1. data/schneider/c12_schneider_inelastic_mass_xs.csv + .metadata.json
  2. data/schneider/schneider_inelastic_rates_v1.bin + .metadata.json

Verifies:
  - 100% Byte-for-byte determinism
  - Strict compliance with authoritative README metadata schema (including energy_grid and target_elements)
  - Complete 25 sections x 13 targets x 860 energy nodes grid
  - Identity between compiled CSV total, compiled binary total, and raw Step 06 truth
  - Strict partial sum closure: sum_targets compiled_partial == compiled_total
  - Comprehensive real negative tests:
      * Gate 1: shifted energy grid
      * Gate 2: missing energy node
      * Gate 3: shuffled/out-of-order input
      * Gate 4: wrong section count
      * Gate 5: missing/mixed provenance (physics_list, topas_ver, schneider_sha, timestamp, git_commit)
      * Gate 6: wrong/duplicate target
      * Gate 7: negative rate value
      * Gate 8: real tampered metadata data_sha256 detection
      * Gate 9: real unit mismatch detection
"""

import copy
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

from compile_schneider_c12_rates import (
    CANONICAL_TARGET_NAMES,
    CANONICAL_Z_ORDER,
    EXPECTED_SECTIONS,
    EXPECTED_TARGETS,
    EXPECTED_ENERGIES,
    ENERGY_MIN,
    ENERGY_MAX,
    ENERGY_STEP,
    compile_rates,
    sha256_file
)

README_REQUIRED_FIELDS = [
    "schema_version",
    "data_sha256",
    "topas_version",
    "geant4_version",
    "physics_list",
    "schneider_source_path",
    "schneider_sha256",
    "extractor_git_commit",
    "compiler_git_commit",
    "raw_campaign_manifest_sha256",
    "energy_min_MeVu",
    "energy_max_MeVu",
    "energy_grid",
    "projectiles",
    "target_elements",
    "units",
    "generation_timestamp_utc",
    "validation_report_sha256"
]

def verify_metadata_schema(meta_dict, filename):
    for f in README_REQUIRED_FIELDS:
        if f not in meta_dict:
            raise AssertionError(f"Metadata {filename} missing authoritative README required field: {f}")
        val = meta_dict[f]
        if val is None or val == "" or val == "TODO" or val == "placeholder":
            raise AssertionError(f"Metadata {filename} field {f} contains placeholder/empty value: {val}")

    # Check energy_grid object schema
    eg = meta_dict["energy_grid"]
    assert isinstance(eg, dict), f"energy_grid must be an object in {filename}"
    assert eg.get("type") == "uniform"
    assert abs(eg.get("min_MeVu") - ENERGY_MIN) < 1e-6
    assert abs(eg.get("max_MeVu") - ENERGY_MAX) < 1e-6
    assert abs(eg.get("step_MeVu") - ENERGY_STEP) < 1e-6
    assert eg.get("nodes") == EXPECTED_ENERGIES

    # Check target_elements
    te = meta_dict["target_elements"]
    assert isinstance(te, list), f"target_elements must be a list in {filename}"
    assert len(te) == EXPECTED_TARGETS
    assert te == CANONICAL_TARGET_NAMES

    # Check units object
    units = meta_dict["units"]
    assert isinstance(units, dict)
    assert units.get("energy") == "MeV/u", f"Units mismatch in {filename}: energy != MeV/u"
    if "mass_cross_section" in units:
        assert units.get("mass_cross_section") == "mm^-1 / (g/cm^3)", f"Units mismatch in {filename}"
    if "mass_partial_rate" in units:
        assert units.get("mass_partial_rate") == "mm^-1 / (g/cm^3)", f"Units mismatch in {filename}"
    if "mass_total_rate" in units:
        assert units.get("mass_total_rate") == "mm^-1 / (g/cm^3)", f"Units mismatch in {filename}"

def verify_data_sha(meta_dict, data_filepath):
    actual_sha = sha256_file(data_filepath)
    expected_sha = meta_dict.get("data_sha256")
    if actual_sha != expected_sha:
        raise AssertionError(f"Data SHA256 mismatch for {data_filepath}: actual {actual_sha} != expected {expected_sha}")

def run_negative_compiler_tests(raw_manifest_path, raw_json_path, repo_dir):
    print("\n--- Running Step 07 Compiler Negative & Robustness Gates ---")
    with open(raw_manifest_path) as f:
        base_manifest = json.load(f)
    with open(raw_json_path) as f:
        base_json = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_root:
        # Helper runner
        def try_compile(manifest_obj, json_obj, test_name):
            t_dir = os.path.join(tmp_root, test_name)
            os.makedirs(t_dir, exist_ok=True)
            m_path = os.path.join(t_dir, "manifest.json")
            j_path = os.path.join(t_dir, os.path.basename(raw_json_path))
            with open(j_path, 'w') as f:
                json.dump(json_obj, f)
            manifest_obj["files"][os.path.basename(raw_json_path)]["sha256"] = sha256_file(j_path)
            with open(m_path, 'w') as f:
                json.dump(manifest_obj, f)

            compile_rates(m_path, j_path, os.path.join(t_dir, "out"), os.path.join(t_dir, "ev"))

        # Gate 1: Shifted / Malformed Energy Grid
        print("Testing Gate 1: Shifted / Malformed Energy Grid rejection...")
        bad_json = copy.deepcopy(base_json)
        bad_json["sections"][0]["grid"][5]["energy_mevu"] = 12.3456
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_shifted_grid")
        except ValueError as e:
            threw = True
            assert "Energy grid deviation" in str(e)
        assert threw, "Compiler accepted shifted energy grid!"
        print("  -> Passed: Shifted energy grid rejected.")

        # Gate 2: Missing Energy Node
        print("Testing Gate 2: Missing Energy Node rejection...")
        bad_json = copy.deepcopy(base_json)
        bad_json["sections"][0]["grid"].pop()
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_missing_node")
        except ValueError as e:
            threw = True
            assert "energy nodes" in str(e) or "bounds mismatch" in str(e)
        assert threw, "Compiler accepted truncated energy nodes!"
        print("  -> Passed: Missing energy node rejected.")

        # Gate 3: Shuffled / Out-of-Order Input
        print("Testing Gate 3: Shuffled / Out-of-Order Input rejection...")
        bad_json = copy.deepcopy(base_json)
        # Swap sections 0 and 1
        bad_json["sections"][0], bad_json["sections"][1] = bad_json["sections"][1], bad_json["sections"][0]
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_shuffled_sections")
        except ValueError as e:
            threw = True
            assert "ordering mismatch" in str(e)
        assert threw, "Compiler accepted shuffled section order!"
        print("  -> Passed: Shuffled input rejected.")

        # Gate 4: Wrong Section Count
        print("Testing Gate 4: Wrong Section Count rejection...")
        bad_json = copy.deepcopy(base_json)
        bad_json["sections"].pop()  # 24 sections
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_wrong_sec_count")
        except ValueError as e:
            threw = True
            assert "Expected 25 sections" in str(e)
        assert threw, "Compiler accepted wrong section count!"
        print("  -> Passed: Wrong section count rejected.")

        # Gate 5: Missing / Mixed Provenance
        print("Testing Gate 5: Missing / Mixed Provenance rejection...")
        # 5A: Missing physics_list
        bad_manifest = copy.deepcopy(base_manifest)
        bad_manifest.pop("physics_list", None)
        bad_manifest.pop("physics_modules", None)
        threw = False
        try:
            try_compile(bad_manifest, base_json, "test_missing_physics")
        except ValueError as e:
            threw = True
            assert "physics_list" in str(e)
        assert threw, "Compiler accepted missing physics_list!"

        # 5B: Wrong topas_version
        bad_manifest = copy.deepcopy(base_manifest)
        bad_manifest["topas_version"] = ""
        threw = False
        try:
            try_compile(bad_manifest, base_json, "test_bad_topas_ver")
        except ValueError as e:
            threw = True
            assert "topas_version" in str(e)
        assert threw, "Compiler accepted empty topas_version!"

        # 5C: Schneider SHA256 mismatch
        bad_manifest = copy.deepcopy(base_manifest)
        bad_manifest["schneider_sha256"] = "0" * 64
        threw = False
        try:
            try_compile(bad_manifest, base_json, "test_bad_schneider_sha")
        except ValueError as e:
            threw = True
            assert "Schneider provenance mismatch" in str(e)
        assert threw, "Compiler accepted Schneider SHA mismatch!"

        # 5D: Missing generation_timestamp_utc
        bad_manifest = copy.deepcopy(base_manifest)
        bad_manifest.pop("generation_timestamp_utc", None)
        threw = False
        try:
            try_compile(bad_manifest, base_json, "test_missing_ts")
        except ValueError as e:
            threw = True
            assert "generation_timestamp_utc" in str(e)
        assert threw, "Compiler accepted missing generation_timestamp_utc!"

        # 5E: Missing git_commit
        bad_manifest = copy.deepcopy(base_manifest)
        bad_manifest.pop("git_commit", None)
        threw = False
        try:
            try_compile(bad_manifest, base_json, "test_missing_commit")
        except ValueError as e:
            threw = True
            assert "git_commit" in str(e)
        assert threw, "Compiler accepted missing git_commit!"
        print("  -> Passed: Missing/mixed provenance strictly rejected.")

        # Gate 6: Wrong / Duplicate Target
        print("Testing Gate 6: Wrong / Duplicate Target rejection...")
        bad_json = copy.deepcopy(base_json)
        bad_json["sections"][0]["grid"][0]["elements"][1] = copy.deepcopy(
            bad_json["sections"][0]["grid"][0]["elements"][0]
        )
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_dup_target")
        except ValueError as e:
            threw = True
            assert "Target Z order mismatch" in str(e) or "Duplicate key" in str(e)
        assert threw, "Compiler accepted duplicate target!"
        print("  -> Passed: Duplicate/wrong target rejected.")

        # Gate 7: Negative Rate Value
        print("Testing Gate 7: Negative Rate Value rejection...")
        bad_json = copy.deepcopy(base_json)
        bad_json["sections"][0]["grid"][0]["elements"][0]["mass_partial_per_mm_at_1g_cm3"] = -0.05
        bad_manifest = copy.deepcopy(base_manifest)
        threw = False
        try:
            try_compile(bad_manifest, bad_json, "test_neg_rate")
        except ValueError as e:
            threw = True
            assert "Invalid mass partial" in str(e)
        assert threw, "Compiler accepted negative rate!"
        print("  -> Passed: Negative rate rejected.")

        # Gate 8: Real Tampered Metadata SHA rejection
        print("Testing Gate 8: Real Tampered Metadata SHA rejection...")
        csv_path = os.path.join(repo_dir, "data/schneider/c12_schneider_inelastic_mass_xs.csv")
        csv_meta_path = os.path.join(repo_dir, "data/schneider/c12_schneider_inelastic_mass_xs.metadata.json")
        with open(csv_meta_path) as f:
            meta_copy = json.load(f)
        meta_copy["data_sha256"] = "0" * 64
        threw = False
        try:
            verify_data_sha(meta_copy, csv_path)
        except AssertionError as e:
            threw = True
            assert "Data SHA256 mismatch" in str(e)
        assert threw, "verify_data_sha accepted tampered SHA!"
        print("  -> Passed: Real tampered metadata hash rejected.")

        # Gate 9: Real Unit Mismatch rejection
        print("Testing Gate 9: Real Unit Mismatch rejection...")
        with open(csv_meta_path) as f:
            meta_copy = json.load(f)
        meta_copy["units"]["energy"] = "keV"
        threw = False
        try:
            verify_metadata_schema(meta_copy, "test_unit_mismatch.json")
        except AssertionError as e:
            threw = True
            assert "Units mismatch" in str(e)
        assert threw, "verify_metadata_schema accepted wrong units!"
        print("  -> Passed: Real unit mismatch rejected.")

    print("All Step 07 Compiler Negative & Robustness Gates PASSED successfully.\n")

def main():
    repo_dir = "/mnt/sdb/wuwei/MAIGO"
    output_dir = os.path.join(repo_dir, "data/schneider")
    evidence_dir = "/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-07"
    raw_manifest_path = "/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-04/run-manifest.json"
    raw_json_path = "/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-04/topas-c12-schneider-inelastic-xs.json"

    os.makedirs(evidence_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "c12_schneider_inelastic_mass_xs.csv")
    csv_meta_path = os.path.join(output_dir, "c12_schneider_inelastic_mass_xs.metadata.json")
    bin_path = os.path.join(output_dir, "schneider_inelastic_rates_v1.bin")
    bin_meta_path = os.path.join(output_dir, "schneider_inelastic_rates_v1.metadata.json")

    # 1. Real Negative Compiler Tests (Gates 1 through 9)
    run_negative_compiler_tests(raw_manifest_path, raw_json_path, repo_dir)

    # 2. Audit Metadata Files against README schema
    print("Auditing generated metadata schemas against README requirements...")
    with open(csv_meta_path) as f:
        csv_meta = json.load(f)
    verify_metadata_schema(csv_meta, "c12_schneider_inelastic_mass_xs.metadata.json")
    verify_data_sha(csv_meta, csv_path)

    with open(bin_meta_path) as f:
        bin_meta = json.load(f)
    verify_metadata_schema(bin_meta, "schneider_inelastic_rates_v1.metadata.json")
    verify_data_sha(bin_meta, bin_path)
    assert bin_meta.get("target_z_order") == CANONICAL_Z_ORDER, "Binary metadata target Z order mismatch!"
    assert "dimension_order" in bin_meta, "Binary metadata missing dimension_order!"
    print("Metadata audits passed (all 18 README provenance fields verified, no placeholders).")

    # 3. Audit CSV Data
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        csv_rows = list(reader)

    assert header[0] == "energy_MeV_per_u"
    assert len(header) == 26
    assert len(csv_rows) == EXPECTED_ENERGIES

    # 4. Audit Binary Data
    with open(bin_path, 'rb') as f:
        magic = f.read(8)
        assert magic == b"SCHNRATE", f"Invalid magic: {magic}"
        header_bytes = f.read(struct.calcsize("<IIIIddd13i"))
        version, num_sec, num_tar, num_e, min_e, max_e, step_e, *tar_z = struct.unpack("<IIIIddd13i", header_bytes)
        assert version == 1
        assert num_sec == EXPECTED_SECTIONS
        assert num_tar == EXPECTED_TARGETS
        assert num_e == EXPECTED_ENERGIES
        assert abs(min_e - ENERGY_MIN) < 1e-6
        assert abs(max_e - ENERGY_MAX) < 1e-6
        assert abs(step_e - ENERGY_STEP) < 1e-6
        assert list(tar_z) == CANONICAL_Z_ORDER

        partial_floats = struct.unpack(f"<{num_sec * num_tar * num_e}d", f.read(num_sec * num_tar * num_e * 8))
        total_floats = struct.unpack(f"<{num_sec * num_e}d", f.read(num_sec * num_e * 8))

    print("Binary header & dimensions verified.")

    # 5. Cross-validate Raw Truth vs CSV vs Binary
    with open(raw_json_path) as f:
        raw_data = json.load(f)

    max_diff_raw_vs_csv = 0.0
    max_diff_raw_vs_bin = 0.0
    max_diff_bin_partials_sum = 0.0

    for s in range(EXPECTED_SECTIONS):
        raw_sec = raw_data["sections"][s]
        for e in range(EXPECTED_ENERGIES):
            raw_pt = raw_sec["grid"][e]
            raw_tot = raw_pt["mass_total_per_mm_at_1g_cm3"]

            csv_tot = float(csv_rows[e][s + 1])
            diff_csv = abs(raw_tot - csv_tot)
            if diff_csv > max_diff_raw_vs_csv:
                max_diff_raw_vs_csv = diff_csv

            bin_tot = total_floats[s * EXPECTED_ENERGIES + e]
            diff_bin = abs(raw_tot - bin_tot)
            if diff_bin > max_diff_raw_vs_bin:
                max_diff_raw_vs_bin = diff_bin

            # Partial sum closure in binary
            partial_sum = sum(partial_floats[s * (EXPECTED_TARGETS * EXPECTED_ENERGIES) + t * EXPECTED_ENERGIES + e]
                              for t in range(EXPECTED_TARGETS))
            diff_part = abs(partial_sum - bin_tot)
            if diff_part > max_diff_bin_partials_sum:
                max_diff_bin_partials_sum = diff_part

    assert max_diff_raw_vs_csv < 1.0e-9, f"CSV vs Raw diff: {max_diff_raw_vs_csv}"
    assert max_diff_raw_vs_bin < 1.0e-14, f"BIN vs Raw diff: {max_diff_raw_vs_bin}"
    assert max_diff_bin_partials_sum < 1.0e-12, f"Partial sum diff: {max_diff_bin_partials_sum}"

    print(f"Numerical invariants verified across 21,500 section-energy nodes:")
    print(f"  Max diff (CSV vs Raw Truth): {max_diff_raw_vs_csv:.3e}")
    print(f"  Max diff (Binary Total vs Raw Truth): {max_diff_raw_vs_bin:.3e}")
    print(f"  Max diff (Binary Partial Sum vs Total): {max_diff_bin_partials_sum:.3e}")

    # 6. Verify Byte-for-Byte Determinism
    with tempfile.TemporaryDirectory() as tmp_dir:
        res1 = compile_rates(raw_manifest_path, raw_json_path, os.path.join(tmp_dir, "out1"), os.path.join(tmp_dir, "ev1"))
        res2 = compile_rates(raw_manifest_path, raw_json_path, os.path.join(tmp_dir, "out2"), os.path.join(tmp_dir, "ev2"))
        assert res1["csv_sha256"] == res2["csv_sha256"]
        assert res1["bin_sha256"] == res2["bin_sha256"]
        assert res1["csv_meta_sha256"] == res2["csv_meta_sha256"]
        assert res1["bin_meta_sha256"] == res2["bin_meta_sha256"]
        print("Byte-for-byte compile determinism verified across independent passes.")

    # 7. Generate Step 07 Evidence & Manifest
    audit_report = {
        "schema_version": 1,
        "step": "step-07",
        "description": "Schneider C12 Inelastic Rate Product Compilation Audit",
        "quality_gate_passed": True,
        "sections_checked": EXPECTED_SECTIONS,
        "targets_checked": EXPECTED_TARGETS,
        "energy_nodes_checked": EXPECTED_ENERGIES,
        "max_diff_raw_vs_csv": max_diff_raw_vs_csv,
        "max_diff_raw_vs_bin": max_diff_raw_vs_bin,
        "max_diff_bin_partials_sum": max_diff_bin_partials_sum,
        "determinism_verified": True,
        "negative_tests_passed": True,
        "artifacts": {
            "c12_schneider_inelastic_mass_xs.csv": {
                "sha256": sha256_file(csv_path),
                "size_bytes": os.path.getsize(csv_path)
            },
            "c12_schneider_inelastic_mass_xs.metadata.json": {
                "sha256": sha256_file(csv_meta_path),
                "size_bytes": os.path.getsize(csv_meta_path)
            },
            "schneider_inelastic_rates_v1.bin": {
                "sha256": sha256_file(bin_path),
                "size_bytes": os.path.getsize(bin_path)
            },
            "schneider_inelastic_rates_v1.metadata.json": {
                "sha256": sha256_file(bin_meta_path),
                "size_bytes": os.path.getsize(bin_meta_path)
            }
        }
    }

    audit_path = os.path.join(evidence_dir, "compiler-audit.json")
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)

    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir).decode().strip()
    manifest = {
        "step": "step-07",
        "description": "Schneider C12 Inelastic Rate Compiler Delivery",
        "git_commit": git_head,
        "quality_gate_passed": True,
        "negative_tests_passed": True,
        "files": {
            "compiler-audit.json": {
                "sha256": sha256_file(audit_path),
                "size_bytes": os.path.getsize(audit_path)
            }
        }
    }
    manifest_path = os.path.join(evidence_dir, "run-manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    step07_sha256_path = os.path.join(repo_dir, "plan/evidence-step07.sha256")
    with open(step07_sha256_path, 'w') as f:
        f.write("# Cryptographic anchor for external Step 07 evidence stored under /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-07/\n")
        for fname in sorted(["compiler-audit.json", "run-manifest.json"]):
            fpath = os.path.join(evidence_dir, fname)
            f.write(f"{sha256_file(fpath)}  {fname}\n")
    print(f"Generated {step07_sha256_path}")

    print("\n================ STEP 07 VERIFICATION SUMMARY ================")
    print(f"Quality Gate Passed: True")
    print(f"Deterministic Compilation: True")
    print(f"Negative Gates Passed: True (all 9 malformed cases rejected)")
    print(f"Partial Sum Closure: True (diff < 1e-12)")
    print(f"Raw vs Compiled Total Match: True (diff < 1e-14)")
    print("==============================================================")

if __name__ == '__main__':
    main()
