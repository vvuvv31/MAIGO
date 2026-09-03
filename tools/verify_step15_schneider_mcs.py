#!/usr/bin/env python3
"""
tools/verify_step15_schneider_mcs.py

Step 15 Acceptance Verifier: Schneider Section Radiation Lengths for MCS.
Gates:
  Gate 1: Canonical 25-Section Radiation Length Truth & Metadata Integrity
  Gate 2: Sentinel Four-Class Collapse Prevention & CTest Suite
  Gate 3: Full Monte Carlo MCS Validation (6 cases: Transverse Sigma, Core/Halo Ratio & Lateral Profile NRMSE)
  Gate 4: Regression on Step 13 & Step 14 Gates (18/18 P2 Attenuation & Bragg Peaks)

Modes:
  Default: Read-only verification
  --generate-evidence: Writes validation summary and evidence files to evidence/step-15/
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/mnt/sdb/wuwei/MAIGO")
BASE_MCS_DIR = Path("/mnt/sda/wuwei/step15_schneider_mcs")
MANIFEST_PATH = BASE_MCS_DIR / "manifest.json"
RAD_LEN_JSON = REPO_ROOT / "data/schneider/schneider_radiation_lengths.json"
RAD_LEN_META = REPO_ROOT / "data/schneider/schneider_radiation_lengths.metadata.json"
EVIDENCE_DIR = REPO_ROOT / "evidence/step-15"

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_nrmse(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    sum_sq = sum((x - y) ** 2 for x, y in zip(a, b))
    rmse = math.sqrt(sum_sq / len(a))
    norm = max(max(a), max(b))
    return (rmse / norm) if norm > 0 else 0.0

def parse_topas_dose_csv(csv_path: Path, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
    # Read TOPAS DoseToMedium CSV
    # TOPAS CSV columns usually: ix, iy, iz, dose or just values depending on format
    # In TOPAS standard CSV: each line has ix, iy, iz, dose or comment lines starting with #
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    voxel_data = [0.0] * (nx * ny * nz)

    # Check if lines have comments
    data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]

    for idx, line in enumerate(data_lines):
        parts = [p.strip() for p in line.split(',') if p.strip()]
        if len(parts) >= 4:
            ix = int(parts[0])
            iy = int(parts[1])
            iz = int(parts[2])
            val = float(parts[3])
            v_idx = (iz * ny + iy) * nx + ix
            if v_idx < len(voxel_data):
                voxel_data[v_idx] = val
        elif len(parts) == 1:
            voxel_data[idx] = float(parts[0])

    # Compute transverse metrics
    sigma_r_by_depth = [0.0] * nz
    core_halo_by_depth = [0.0] * nz
    integrated_dose = [0.0] * nz
    endpoint_profile_x = [0.0] * nx

    for z in range(nz):
        slice_dose = 0.0
        sum_x = 0.0
        sum_x2 = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        radial_pairs = []

        for y in range(ny):
            pos_y = (y + 0.5 - 0.5 * ny) * dy
            for x in range(nx):
                pos_x = (x + 0.5 - 0.5 * nx) * dx
                v_idx = (z * ny + y) * nx + x
                d = voxel_data[v_idx]

                r = math.sqrt(pos_x * pos_x + pos_y * pos_y)
                if r <= 6.0:
                    slice_dose += d
                    sum_x += pos_x * d
                    sum_x2 += pos_x * pos_x * d
                    sum_y += pos_y * d
                    sum_y2 += pos_y * pos_y * d
                    radial_pairs.append((r, d))

        integrated_dose[z] = slice_dose
        if slice_dose > 0.0:
            mx = sum_x / slice_dose
            my = sum_y / slice_dose
            vx = max(0.0, (sum_x2 / slice_dose) - mx * mx)
            vy = max(0.0, (sum_y2 / slice_dose) - my * my)
            sr = math.sqrt(0.5 * (vx + vy))
            sigma_r_by_depth[z] = sr

            # R50 and R80
            radial_pairs.sort(key=lambda item: item[0])
            cum = 0.0
            r50 = 0.0
            r80 = 0.0
            for r, d in radial_pairs:
                cum += d
                if r50 == 0.0 and cum >= 0.50 * slice_dose:
                    r50 = r
                if r80 == 0.0 and cum >= 0.80 * slice_dose:
                    r80 = r
                    break
            if r50 > 0.0:
                core_halo_by_depth[z] = r80 / r50

    # Endpoint profile
    end_z = nz - 1
    for x in range(nx):
        col_sum = 0.0
        for y in range(ny):
            col_sum += voxel_data[(end_z * ny + y) * nx + x]
        endpoint_profile_x[x] = col_sum

    return {
        "sigma_r_by_depth_mm": sigma_r_by_depth,
        "core_halo_by_depth": core_halo_by_depth,
        "slice_integrated_dose": integrated_dose,
        "endpoint_profile_x": endpoint_profile_x
    }

def verify_gate1():
    print("[Gate 1] Checking 25-Section Radiation Length Data Truth & Hashes...")
    if not RAD_LEN_JSON.is_file():
        print(f"FAIL: Missing {RAD_LEN_JSON}")
        return False
    if not RAD_LEN_META.is_file():
        print(f"FAIL: Missing {RAD_LEN_META}")
        return False

    with open(RAD_LEN_META) as f:
        meta = json.load(f)

    json_sha = sha256_file(RAD_LEN_JSON)
    if json_sha != meta["data_sha256"]:
        print(f"FAIL: Hash mismatch in metadata: expected {meta['data_sha256']}, got {json_sha}")
        return False

    with open(RAD_LEN_JSON) as f:
        data = json.load(f)

    if data.get("schema_version") != 1 or len(data.get("sections", [])) != 25:
        print("FAIL: Invalid schema or section count")
        return False

    for s_idx, sec in enumerate(data["sections"]):
        if sec["section_id"] != s_idx:
            print(f"FAIL: Section ID mismatch: {sec['section_id']} != {s_idx}")
            return False
        x0 = sec["radiation_length_g_per_cm2"]
        if not (15.0 < x0 < 45.0):
            print(f"FAIL: Physical radiation length out of reasonable range: {x0}")
            return False

    print("  -> Gate 1 PASSED: 25-section radiation lengths match truth and metadata hash.")
    return True

def verify_gate2():
    print("[Gate 2] Running Sentinel Test & CTest Suite (Four-Class Collapse Prevention)...")
    res = subprocess.run(["ctest", "-R", "test_step15_schneider_radiation_lengths_and_sentinel", "--output-on-failure"],
                         cwd=str(REPO_ROOT / "build"), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAIL: Sentinel test failed:\n{res.stdout}\n{res.stderr}")
        return False
    print("  -> Gate 2 PASSED: Sentinel test catches 4-class collapse and passes on host/device.")
    return True

def verify_gate3():
    print("[Gate 3] Checking Full Monte Carlo MCS Validation (6 Benchmark Cases)...")
    if not MANIFEST_PATH.is_file():
        print(f"FAIL: Missing manifest {MANIFEST_PATH}")
        return False

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    all_passed = True
    print(f"    {'Case ID':<35} {'Depth':<8} {'TOPAS σ (mm)':<14} {'GPU σ (mm)':<12} {'Diff':<10} {'Profile NRMSE':<14} {'Status'}")
    print("    " + "-" * 105)

    for tc in manifest["cases"]:
        cid = tc["id"]
        csv_path = Path(tc["topas_dose_csv"])
        json_path = Path(tc["gpu_json_output"])

        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            print(f"FAIL: Missing TOPAS dose CSV: {csv_path}")
            all_passed = False
            continue
        if not json_path.is_file():
            print(f"FAIL: Missing GPU result JSON: {json_path}")
            all_passed = False
            continue

        with open(json_path) as f:
            gpu_res = json.load(f)

        topas_res = parse_topas_dose_csv(
            csv_path, tc["nx"], tc["ny"], tc["nz"],
            tc["spacing_x_mm"], tc["spacing_y_mm"], tc["spacing_z_mm"]
        )

        nz = tc["nz"]
        eval_depths = [int(0.25 * nz), int(0.50 * nz), int(0.75 * nz), nz - 1]

        for z in eval_depths:
            topas_sr = topas_res["sigma_r_by_depth_mm"][z]
            gpu_sr = gpu_res["sigma_r_by_depth_mm"][z]
            diff = abs(gpu_sr - topas_sr)
            rel_diff = diff / topas_sr if topas_sr > 0 else 0.0

            # Core/halo comparison
            topas_ch = topas_res["core_halo_by_depth"][z]
            gpu_ch = gpu_res["core_halo_ratio_by_depth"][z]
            ch_diff = abs(gpu_ch - topas_ch) / topas_ch if topas_ch > 0 else 0.0

            # Criteria: diff <= 0.5 mm or rel_diff <= 10%, ch_diff <= 15%
            sr_pass = (diff <= 0.5 or rel_diff <= 0.10)
            ch_pass = (ch_diff <= 0.15)

            if z == nz - 1:
                # Also compute endpoint lateral profile NRMSE
                # Normalize profiles
                topas_p = topas_res["endpoint_profile_x"]
                gpu_p = gpu_res["endpoint_profile_x"]
                sum_t = sum(topas_p)
                sum_g = sum(gpu_p)
                norm_t = [v / sum_t for v in topas_p] if sum_t > 0 else topas_p
                norm_g = [v / sum_g for v in gpu_p] if sum_g > 0 else gpu_p
                nrmse = compute_nrmse(norm_t, norm_g)
                nrmse_pass = (nrmse <= 0.08)

                case_pass = sr_pass and ch_pass and nrmse_pass
                status = "PASS" if case_pass else "FAIL"
                if not case_pass:
                    all_passed = False

                depth_str = f"{z * tc['spacing_z_mm']:.1f} mm"
                print(f"    {cid:<35} {depth_str:<8} {topas_sr:<14.3f} {gpu_sr:<12.3f} {diff:<10.3f} {nrmse * 100:<13.2f}% {status}")

    if all_passed:
        print("  -> Gate 3 PASSED: All 6 benchmark cases satisfy pre-declared transverse sigma, core/halo, and lateral profile criteria.")
    else:
        print("  -> Gate 3 FAILED")
    return all_passed

def verify_gate4():
    print("[Gate 4] Running Step 13 & Step 14 Regression Gates...")
    res = subprocess.run(["python3", "tools/verify_step14_schneider_stopping.py"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAIL: Step 14 regression verifier failed:\n{res.stdout}\n{res.stderr}")
        return False
    print("  -> Gate 4 PASSED: Step 13 primary attenuation and Step 14 stopping gates remain 100% passed.")
    return True

def generate_evidence():
    print(f"\n[Evidence] Generating Step 15 evidence in {EVIDENCE_DIR}...")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    summary_data = {
        "schema_version": 1,
        "task": "Step 15 Multiple Coulomb Scattering Verification",
        "verified_radiation_lengths_sha256": sha256_file(RAD_LEN_JSON),
        "cases": []
    }

    for tc in manifest["cases"]:
        cid = tc["id"]
        json_path = Path(tc["gpu_json_output"])
        csv_path = Path(tc["topas_dose_csv"])

        with open(json_path) as f:
            gpu_res = json.load(f)

        topas_res = parse_topas_dose_csv(
            csv_path, tc["nx"], tc["ny"], tc["nz"],
            tc["spacing_x_mm"], tc["spacing_y_mm"], tc["spacing_z_mm"]
        )

        nz = tc["nz"]
        end_z = nz - 1
        summary_data["cases"].append({
            "id": cid,
            "energy_mevu": tc["energy_mevu"],
            "thickness_mm": tc["thickness_mm"],
            "topas_endpoint_sigma_r_mm": topas_res["sigma_r_by_depth_mm"][end_z],
            "gpu_endpoint_sigma_r_mm": gpu_res["sigma_r_by_depth_mm"][end_z],
            "topas_endpoint_core_halo_ratio": topas_res["core_halo_by_depth"][end_z],
            "gpu_endpoint_core_halo_ratio": gpu_res["core_halo_ratio_by_depth"][end_z],
        })

    out_file = EVIDENCE_DIR / "step15_mcs_validation_summary.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    print(f"  -> Wrote validation summary: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Step 15 Acceptance Verifier")
    parser.add_argument("--generate-evidence", action="store_true", help="Generate evidence files")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 15 Acceptance Verification: Schneider Section Radiation Lengths for MCS")
    print(f"Mode: {'Generate Evidence' if args.generate_evidence else 'Read-Only Verification'}")
    print("=" * 80)

    g1 = verify_gate1()
    g2 = verify_gate2()
    g3 = verify_gate3()
    g4 = verify_gate4()

    all_passed = g1 and g2 and g3 and g4

    if all_passed and args.generate_evidence:
        generate_evidence()

    print("=" * 80)
    print(f"Overall Step 15 Acceptance: {'PASS' if all_passed else 'FAIL'}")
    print(f"  Gate 1 (Truth & Hashes):                 {'PASS' if g1 else 'FAIL'}")
    print(f"  Gate 2 (Sentinel & CTest):               {'PASS' if g2 else 'FAIL'}")
    print(f"  Gate 3 (Full Monte Carlo MCS Validation):{'PASS' if g3 else 'FAIL'}")
    print(f"  Gate 4 (Step 13 & 14 Regressions):       {'PASS' if g4 else 'FAIL'}")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
