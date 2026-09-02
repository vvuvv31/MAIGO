#!/usr/bin/env python3
"""
tools/verify_step13_p2_gates.py

Automated Verification Tool for Step 13 (Validate the C12 primary CT milestone):
Evaluates all 7 fixed P2 acceptance gates with a STRICT FAIL-CLOSED contract:

1. Primary survival integral relative difference < 1%
2. First-interaction-depth NRMSE < 2%
3. Range / Bragg position difference <= max(0.5 mm, one voxel size)
4. Incident = survived + inelastic + other-terminal categories exactly
5. Section mapping mismatch = 0
6. Secondary / replay count = 0 (observed real diagnostics)
7. Overflow count = 0 (observed real diagnostics)

FAIL-CLOSED RULES:
- Every required gate starts as False.
- Missing data, missing files, empty checkpoints, or inadequate sample size causes explicit FAIL.
- Only non-required gates are marked 'N/A' and exempted from case pass criteria.
- Exits 0 only if ALL required gates in ALL cases pass.

Outputs:
- /mnt/sda/wuwei/step13_primary_ct/evidence/step13-validation-report.json
- /mnt/sda/wuwei/step13_primary_ct/evidence/step13-validation-summary.md
- /mnt/sdb/wuwei/MAIGO/evidence/step13/evidence_manifest.json
- /mnt/sdb/wuwei/MAIGO/evidence/step13/step13-validation-report.json
- /mnt/sdb/wuwei/MAIGO/evidence/step13/step13-validation-summary.md
"""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/mnt/sda/wuwei/step13_primary_ct")
MANIFEST_PATH = BASE_DIR / "manifest.json"
SDA_EVIDENCE_DIR = BASE_DIR / "evidence"
REPO_EVIDENCE_DIR = Path("/mnt/sdb/wuwei/MAIGO/evidence/step13")

def sha256_file(filepath: Path) -> str:
    if not filepath.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def parse_topas_dose3d_csv(csv_path: Path, nx: int, ny: int, nz: int):
    """Parse TOPAS 3D dose CSV and compute lateral sum 1D depth-dose IDD."""
    if not csv_path.is_file():
        return None
    idd = [0.0] * nz
    parsed_any = False
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    iz = int(parts[2])
                    val = float(parts[3])
                    if 0 <= iz < nz:
                        idd[iz] += val
                        parsed_any = True
                except ValueError:
                    pass
    return idd if parsed_any else None

def find_bragg_metrics(idd: list, spacing_z: float, origin_z: float):
    if not idd or max(idd) <= 0.0:
        return None
    peak_val = max(idd)
    peak_idx = idd.index(peak_val)
    peak_depth = origin_z + (peak_idx + 0.5) * spacing_z
    
    r80_depth = None
    r50_depth = None
    for k in range(peak_idx, len(idd) - 1):
        v1, v2 = idd[k], idd[k + 1]
        z1 = origin_z + (k + 0.5) * spacing_z
        z2 = origin_z + (k + 1.5) * spacing_z
        if v1 >= 0.8 * peak_val and v2 <= 0.8 * peak_val:
            frac = (v1 - 0.8 * peak_val) / (v1 - v2) if v1 != v2 else 0.0
            r80_depth = z1 + frac * (z2 - z1)
        if v1 >= 0.5 * peak_val and v2 <= 0.5 * peak_val:
            frac = (v1 - 0.5 * peak_val) / (v1 - v2) if v1 != v2 else 0.0
            r50_depth = z1 + frac * (z2 - z1)
            break
    return {
        "peak_depth_mm": peak_depth,
        "peak_val": peak_val,
        "r80_distal_mm": r80_depth,
        "r50_distal_mm": r50_depth
    }

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    results = []
    all_cases_pass = True

    for case in manifest["cases"]:
        cid = case["id"]
        category = case["category"]
        req_gates = case.get("required_gates", [1, 2, 4, 5, 6, 7])
        gpu_json_path = Path(case["gpu_json_output"])
        topas_json_path = Path(case.get("topas_json_output", ""))
        topas_dose_path = Path(case.get("topas_dose_csv", ""))

        case_errors = []

        if not gpu_json_path.is_file():
            case_errors.append(f"GPU result JSON missing: {gpu_json_path}")
            results.append({
                "id": cid,
                "category": category,
                "passed_all_gates": False,
                "errors": case_errors
            })
            all_cases_pass = False
            continue

        with open(gpu_json_path) as f:
            gpu_data = json.load(f)

        topas_data = None
        if topas_json_path.is_file():
            with open(topas_json_path) as f:
                topas_data = json.load(f)

        # Gate 1: Primary survival integral relative difference < 1%
        gate1_required = 1 in req_gates
        gate1_pass = False
        gate1_rel_diff = None
        gate1_reason = ""
        topas_surv_int = None
        gpu_surv_int = None

        if not gate1_required:
            gate1_pass = True
            gate1_reason = "N/A (Not required for this case)"
        else:
            if topas_data is None:
                gate1_reason = f"TOPAS JSON missing: {topas_json_path}"
            elif "depth_checkpoints" not in topas_data:
                gate1_reason = "TOPAS depth_checkpoints key missing"
            elif "depth_checkpoints" not in gpu_data:
                gate1_reason = "GPU depth_checkpoints key missing"
            else:
                topas_cps = topas_data["depth_checkpoints"]
                gpu_cps = gpu_data["depth_checkpoints"]
                if len(topas_cps) == 0 or len(gpu_cps) == 0:
                    gate1_reason = "Empty depth checkpoints"
                elif len(topas_cps) != len(gpu_cps):
                    gate1_reason = f"Checkpoint count mismatch (TOPAS={len(topas_cps)}, GPU={len(gpu_cps)})"
                else:
                    topas_s_sum = sum(cp["survival_fraction"] for cp in topas_cps)
                    gpu_s_sum = sum(cp["survival_fraction"] for cp in gpu_cps)
                    topas_surv_int = topas_s_sum
                    gpu_surv_int = gpu_s_sum
                    if topas_s_sum <= 0:
                        gate1_reason = "TOPAS survival sum <= 0"
                    else:
                        gate1_rel_diff = abs(gpu_s_sum - topas_s_sum) / topas_s_sum
                        if gate1_rel_diff < 0.01:
                            gate1_pass = True
                            gate1_reason = f"PASS (rel_diff={gate1_rel_diff*100:.3f}% < 1.0%)"
                        else:
                            gate1_reason = f"FAIL (rel_diff={gate1_rel_diff*100:.3f}% >= 1.0%)"

        # Gate 2: First-interaction-depth NRMSE < 2%
        gate2_required = 2 in req_gates
        gate2_pass = False
        gate2_nrmse = None
        gate2_reason = ""

        if not gate2_required:
            gate2_pass = True
            gate2_reason = "N/A (Not required for this case)"
        else:
            if topas_data is None:
                gate2_reason = f"TOPAS JSON missing: {topas_json_path}"
            elif "first_interactions_sample" not in topas_data:
                gate2_reason = "TOPAS first_interactions_sample key missing"
            elif "first_interactions_sample" not in gpu_data:
                gate2_reason = "GPU first_interactions_sample key missing"
            else:
                t_samples = topas_data["first_interactions_sample"]
                g_samples = gpu_data["first_interactions_sample"]
                if len(t_samples) < 50:
                    gate2_reason = f"Inadequate TOPAS sample size: {len(t_samples)} < 50"
                elif len(g_samples) < 50:
                    gate2_reason = f"Inadequate GPU sample size: {len(g_samples)} < 50"
                else:
                    t_depths = sorted([r["depth_mm"] for r in t_samples])
                    g_depths = sorted([r["depth_mm"] for r in g_samples])
                    N_q = 100
                    t_q = [t_depths[int(i * (len(t_depths) - 1) / (N_q - 1))] for i in range(N_q)]
                    g_q = [g_depths[int(i * (len(g_depths) - 1) / (N_q - 1))] for i in range(N_q)]
                    mse = sum((g - t) ** 2 for g, t in zip(g_q, t_q)) / N_q
                    rmse = math.sqrt(mse)
                    z_range = max(max(t_depths), max(g_depths)) - min(min(t_depths), min(g_depths))
                    if z_range <= 0.0:
                        gate2_reason = "Depth range <= 0"
                    else:
                        gate2_nrmse = rmse / z_range
                        if gate2_nrmse < 0.02:
                            gate2_pass = True
                            gate2_reason = f"PASS (NRMSE={gate2_nrmse*100:.3f}% < 2.0%)"
                        else:
                            gate2_reason = f"FAIL (NRMSE={gate2_nrmse*100:.3f}% >= 2.0%)"

        # Gate 3: Range / Bragg position difference <= max(0.5 mm, one voxel size)
        gate3_required = 3 in req_gates
        gate3_pass = False
        gate3_diff_mm = None
        gate3_reason = ""

        if not gate3_required:
            gate3_pass = True
            gate3_reason = "N/A (Not required for this case)"
        else:
            spacing_z = case["spacing_z_mm"]
            voxel_gate = max(0.5, spacing_z)
            gpu_bp = gpu_data.get("bragg_peak_metrics", {})
            gpu_peak_z = gpu_bp.get("peak_depth_mm")

            if not topas_dose_path.is_file():
                gate3_reason = f"TOPAS dose CSV missing: {topas_dose_path}"
            elif gpu_peak_z is None:
                gate3_reason = "GPU bragg_peak_metrics missing peak_depth_mm"
            else:
                topas_idd = parse_topas_dose3d_csv(topas_dose_path, case["nx"], case["ny"], case["nz"])
                if topas_idd is None:
                    gate3_reason = "Failed to parse TOPAS dose3d CSV"
                else:
                    topas_bp = find_bragg_metrics(topas_idd, spacing_z, case["origin_z_mm"])
                    if topas_bp is None or topas_bp.get("peak_depth_mm") is None:
                        gate3_reason = "TOPAS Bragg peak could not be determined"
                    else:
                        topas_peak_z = topas_bp["peak_depth_mm"]
                        gate3_diff_mm = abs(gpu_peak_z - topas_peak_z)
                        if gate3_diff_mm <= voxel_gate + 1e-6:
                            gate3_pass = True
                            gate3_reason = f"PASS (|diff|={gate3_diff_mm:.2f}mm <= {voxel_gate:.2f}mm)"
                        else:
                            gate3_reason = f"FAIL (|diff|={gate3_diff_mm:.2f}mm > {voxel_gate:.2f}mm)"

        # Gate 4: Incident = survived + inelastic + other-terminal exactly
        gate4_required = 4 in req_gates
        gate4_pass = False
        gate4_reason = ""
        term_counts = gpu_data.get("terminal_counts", {})
        exact_cons = gpu_data.get("exact_terminal_conservation", False)
        histories = gpu_data.get("histories", 0)
        sum_term = sum(term_counts.values()) if term_counts else 0

        if not gate4_required:
            gate4_pass = True
            gate4_reason = "N/A"
        else:
            if not exact_cons or sum_term != histories or histories == 0:
                gate4_reason = f"FAIL (histories={histories}, sum={sum_term}, exact={exact_cons})"
            else:
                gate4_pass = True
                gate4_reason = f"PASS (exact sum={sum_term} == {histories})"

        # Gate 5: Section mapping mismatch = 0
        gate5_required = 5 in req_gates
        gate5_pass = False
        gate5_reason = ""
        if "section_mapping_mismatch" not in gpu_data:
            gate5_reason = "section_mapping_mismatch key missing in GPU JSON"
        else:
            mismatches = gpu_data["section_mapping_mismatch"]
            if mismatches == 0:
                gate5_pass = True
                gate5_reason = "PASS (0 mismatches)"
            else:
                gate5_reason = f"FAIL ({mismatches} mismatches observed)"

        # Gate 6: Secondary / replay count = 0 (observed real diagnostics)
        gate6_required = 6 in req_gates
        gate6_pass = False
        gate6_reason = ""
        diag = gpu_data.get("diagnostics")
        if not gate6_required:
            gate6_pass = True
            gate6_reason = "N/A"
        elif diag is None:
            gate6_reason = "diagnostics object missing in GPU JSON"
        else:
            gen_sec = diag.get("generated_direct_secondaries", -1)
            q_sec = diag.get("queued_secondaries", -1)
            t_sec = diag.get("transported_secondaries", -1)
            rep_sum = diag.get("cinel02_replay_valid_sum", -1)
            if gen_sec == 0 and q_sec == 0 and t_sec == 0 and rep_sum == 0:
                gate6_pass = True
                gate6_reason = "PASS (all secondary/replay counts == 0)"
            else:
                gate6_reason = f"FAIL (gen={gen_sec}, queued={q_sec}, trans={t_sec}, replays={rep_sum})"

        # Gate 7: Overflow count = 0 (observed real diagnostics)
        gate7_required = 7 in req_gates
        gate7_pass = False
        gate7_reason = ""
        if not gate7_required:
            gate7_pass = True
            gate7_reason = "N/A"
        elif diag is None:
            gate7_reason = "diagnostics object missing in GPU JSON"
        else:
            sec_ovf = diag.get("secondary_queue_overflow", -1)
            el_ovf = diag.get("elastic_queue_overflow", -1)
            neut_ovf = diag.get("neutral_queue_overflow", -1)
            elec_ovf = diag.get("electron_queue_overflow", -1)
            resamp = diag.get("fred_resample_failed_events", -1)
            cap_ovf = diag.get("fred_product_capacity_overflow_events", -1)
            other_term = diag.get("primary_other_terminal_count", -1)
            if (sec_ovf == 0 and el_ovf == 0 and neut_ovf == 0 and elec_ovf == 0 and
                resamp == 0 and cap_ovf == 0 and other_term == 0):
                gate7_pass = True
                gate7_reason = "PASS (all overflow/retry diagnostics == 0)"
            else:
                gate7_reason = f"FAIL (sec_ovf={sec_ovf}, el_ovf={el_ovf}, neut_ovf={neut_ovf}, elec_ovf={elec_ovf}, resamp={resamp}, cap_ovf={cap_ovf}, other_term={other_term})"

        case_all_pass = (gate1_pass and gate2_pass and gate3_pass and gate4_pass and
                          gate5_pass and gate6_pass and gate7_pass)
        if not case_all_pass:
            all_cases_pass = False

        record = {
            "id": cid,
            "category": category,
            "required_gates": req_gates,
            "passed_all_gates": case_all_pass,
            "gate1_survival_integral": {
                "required": gate1_required,
                "topas_integral": topas_surv_int,
                "gpu_integral": gpu_surv_int,
                "rel_diff": gate1_rel_diff,
                "gate_passed": gate1_pass,
                "reason": gate1_reason
            },
            "gate2_first_interaction_nrmse": {
                "required": gate2_required,
                "nrmse": gate2_nrmse,
                "gate_passed": gate2_pass,
                "reason": gate2_reason
            },
            "gate3_bragg_range": {
                "required": gate3_required,
                "peak_diff_mm": gate3_diff_mm,
                "gate_passed": gate3_pass,
                "reason": gate3_reason
            },
            "gate4_terminal_conservation": {
                "required": gate4_required,
                "exact_conservation": exact_cons,
                "histories": histories,
                "sum_terminal_counts": sum_term,
                "counts": term_counts,
                "gate_passed": gate4_pass,
                "reason": gate4_reason
            },
            "gate5_section_mapping": {
                "required": gate5_required,
                "mismatches": gpu_data.get("section_mapping_mismatch"),
                "first_interactions_by_expected_section": gpu_data.get("first_interactions_by_expected_section"),
                "first_interactions_by_recorded_section": gpu_data.get("first_interactions_by_recorded_section"),
                "gate_passed": gate5_pass,
                "reason": gate5_reason
            },
            "gate6_secondary_count": {
                "required": gate6_required,
                "gate_passed": gate6_pass,
                "reason": gate6_reason,
                "diagnostics": {
                    "generated_direct_secondaries": diag.get("generated_direct_secondaries") if diag else None,
                    "queued_secondaries": diag.get("queued_secondaries") if diag else None,
                    "transported_secondaries": diag.get("transported_secondaries") if diag else None,
                    "cinel02_replay_valid_sum": diag.get("cinel02_replay_valid_sum") if diag else None
                }
            },
            "gate7_overflow_count": {
                "required": gate7_required,
                "gate_passed": gate7_pass,
                "reason": gate7_reason,
                "diagnostics": {
                    "secondary_queue_overflow": diag.get("secondary_queue_overflow") if diag else None,
                    "elastic_queue_overflow": diag.get("elastic_queue_overflow") if diag else None,
                    "neutral_queue_overflow": diag.get("neutral_queue_overflow") if diag else None,
                    "electron_queue_overflow": diag.get("electron_queue_overflow") if diag else None,
                    "fred_resample_failed_events": diag.get("fred_resample_failed_events") if diag else None,
                    "fred_product_capacity_overflow_events": diag.get("fred_product_capacity_overflow_events") if diag else None,
                    "primary_other_terminal_count": diag.get("primary_other_terminal_count") if diag else None
                }
            },
            "energy_balance_rel_error": gpu_data.get("relative_energy_balance_error", 0.0)
        }
        results.append(record)

    # Output JSON and Markdown
    report_dict = {
        "schema_version": 2,
        "task": "Step 13 C12 Primary CT Milestone Gate Verification (Fail-Closed)",
        "overall_pass": all_cases_pass,
        "total_cases": len(results),
        "passed_cases": sum(1 for r in results if r["passed_all_gates"]),
        "cases": results
    }

    SDA_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    REPO_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    sda_report_json = SDA_EVIDENCE_DIR / "step13-validation-report.json"
    repo_report_json = REPO_EVIDENCE_DIR / "step13-validation-report.json"
    with open(sda_report_json, "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(repo_report_json, "w") as f:
        json.dump(report_dict, f, indent=2)

    def write_summary_md(path: Path):
        with open(path, "w") as f:
            f.write("# Step 13 — C12 Primary CT Milestone Validation Report\n\n")
            f.write(f"**Overall Status**: {'PASS' if all_cases_pass else 'FAIL'}\n")
            f.write(f"**Passed Cases**: {sum(1 for r in results if r['passed_all_gates'])} / {len(results)}\n\n")
            f.write("## Acceptance Gates Summary Table\n\n")
            f.write("| Case ID | Category | Survival Int Diff (<1%) | First Int NRMSE (<2%) | Bragg Diff (mm) | Term Conservation | Section Match | Overflow | Status |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for r in results:
                s_diff = f"{r['gate1_survival_integral']['rel_diff']*100:.3f}%" if r['gate1_survival_integral']['rel_diff'] is not None else ("N/A" if not r['gate1_survival_integral']['required'] else "FAIL (missing)")
                f_nrmse = f"{r['gate2_first_interaction_nrmse']['nrmse']*100:.3f}%" if r['gate2_first_interaction_nrmse']['nrmse'] is not None else ("N/A" if not r['gate2_first_interaction_nrmse']['required'] else "FAIL (missing)")
                b_diff = f"{r['gate3_bragg_range']['peak_diff_mm']:.2f}" if r['gate3_bragg_range']['peak_diff_mm'] is not None else ("N/A" if not r['gate3_bragg_range']['required'] else "FAIL (missing)")
                cons = "EXACT" if r['gate4_terminal_conservation']['gate_passed'] else "FAIL"
                sec = "MATCH (0)" if r['gate5_section_mapping']['gate_passed'] else f"FAIL ({r['gate5_section_mapping']['mismatches']})"
                ovf = "0" if r['gate7_overflow_count']['gate_passed'] else "FAIL"
                st = "✅ PASS" if r['passed_all_gates'] else "❌ FAIL"
                f.write(f"| {r['id']} | {r['category']} | {s_diff} | {f_nrmse} | {b_diff} | {cons} | {sec} | {ovf} | {st} |\n")

    write_summary_md(SDA_EVIDENCE_DIR / "step13-validation-summary.md")
    write_summary_md(REPO_EVIDENCE_DIR / "step13-validation-summary.md")

    # Cryptographic evidence manifest
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(Path("/mnt/sdb/wuwei/MAIGO"))).stdout.strip()
    evidence_manifest = {
        "schema_version": 1,
        "git_commit_sha": git_sha,
        "overall_validation_pass": all_cases_pass,
        "cases": []
    }
    for c in manifest["cases"]:
        cid = c["id"]
        evidence_manifest["cases"].append({
            "id": cid,
            "category": c["category"],
            "required_gates": c.get("required_gates", []),
            "cctg_sha256": sha256_file(Path(c["cctg_file"])),
            "param_file_sha256": sha256_file(Path(c["param_file"])),
            "topas_json_sha256": sha256_file(Path(c["topas_json_output"])),
            "topas_dose_csv_sha256": sha256_file(Path(c["topas_dose_csv"])),
            "gpu_json_sha256": sha256_file(Path(c["gpu_json_output"])),
        })

    with open(REPO_EVIDENCE_DIR / "evidence_manifest.json", "w") as f:
        json.dump(evidence_manifest, f, indent=2)

    print(f"Validation finished. Overall Pass: {all_cases_pass}")
    print(f"Passed: {sum(1 for r in results if r['passed_all_gates'])} / {len(results)}")
    print(f"Saved reports to {sda_report_json} and {repo_report_json}")
    print(f"Saved evidence manifest to {REPO_EVIDENCE_DIR / 'evidence_manifest.json'}")

    if not all_cases_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
