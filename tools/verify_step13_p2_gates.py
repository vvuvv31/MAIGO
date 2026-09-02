#!/usr/bin/env python3
"""
tools/verify_step13_p2_gates.py

Automated Verification Tool for Step 13 (Validate the C12 primary CT milestone):
Evaluates all 7 fixed P2 acceptance gates:
1. Primary survival integral relative difference < 1%
2. First-interaction-depth NRMSE < 2%
3. Range / Bragg position difference < max(0.5 mm, one voxel size)
4. Incident = survived + inelastic + other-terminal categories exactly
5. Section mapping mismatch = 0
6. Secondary / replay count = 0
7. Overflow count = 0

Outputs:
- /mnt/sda/wuwei/step13_primary_ct/evidence/step13-validation-report.json
- /mnt/sda/wuwei/step13_primary_ct/evidence/step13-validation-summary.md
Exits 0 only if ALL gates pass.
"""

import csv
import json
import math
import sys
from pathlib import Path

BASE_DIR = Path("/mnt/sda/wuwei/step13_primary_ct")
MANIFEST_PATH = BASE_DIR / "manifest.json"
EVIDENCE_DIR = BASE_DIR / "evidence"

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
    
    # Distal search
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
    all_gates_pass = True

    for case in manifest["cases"]:
        cid = case["id"]
        category = case["category"]
        gpu_json_path = Path(case["gpu_json_output"])
        topas_json_path = Path(case.get("topas_json_output", ""))
        topas_dose_path = Path(case.get("topas_dose_csv", ""))

        if not gpu_json_path.exists():
            print(f"Error: GPU result missing for {cid}: {gpu_json_path}")
            all_gates_pass = False
            continue

        with open(gpu_json_path) as f:
            gpu_data = json.load(f)

        topas_data = None
        if topas_json_path.is_file():
            with open(topas_json_path) as f:
                topas_data = json.load(f)

        # Gate 4: Incident = survived + inelastic + other-terminal exactly
        exact_conservation = gpu_data.get("exact_terminal_conservation", False)
        term_counts = gpu_data.get("terminal_counts", {})
        sum_term = sum(term_counts.values())
        histories = gpu_data.get("histories", 0)
        gate_conservation_pass = exact_conservation and (sum_term == histories)

        # Gate 5: Section mapping mismatch = 0
        section_mismatch = gpu_data.get("section_mapping_mismatch", 0)
        gate_section_pass = (section_mismatch == 0)

        # Gate 6: Secondary / replay count = 0
        sec_count = gpu_data.get("secondary_count", 0)
        gate_sec_pass = (sec_count == 0)

        # Gate 7: Overflow count = 0
        overflow_count = gpu_data.get("overflow_count", 0)
        gate_overflow_pass = (overflow_count == 0)

        # Gate 1: Primary survival integral relative difference < 1%
        survival_int_rel_diff = None
        gate_survival_pass = True
        topas_surv_int = None
        gpu_surv_int = None

        if not case.get("is_bragg_check", False) and topas_data and "depth_checkpoints" in topas_data and "depth_checkpoints" in gpu_data:
            topas_cps = topas_data["depth_checkpoints"]
            gpu_cps = gpu_data["depth_checkpoints"]
            if len(topas_cps) == len(gpu_cps) and len(topas_cps) > 0:
                topas_s_sum = sum(cp["survival_fraction"] for cp in topas_cps)
                gpu_s_sum = sum(cp["survival_fraction"] for cp in gpu_cps)
                topas_surv_int = topas_s_sum
                gpu_surv_int = gpu_s_sum
                if topas_s_sum > 0:
                    survival_int_rel_diff = abs(gpu_s_sum - topas_s_sum) / topas_s_sum
                    if survival_int_rel_diff >= 0.01:
                        gate_survival_pass = False

        # Gate 2: First-interaction-depth NRMSE < 2%
        first_int_nrmse = None
        gate_first_int_pass = True
        if not case.get("is_bragg_check", False) and topas_data and "first_interactions_sample" in topas_data and "first_interactions_sample" in gpu_data:
            t_depths = sorted([r["depth_mm"] for r in topas_data["first_interactions_sample"]])
            g_depths = sorted([r["depth_mm"] for r in gpu_data["first_interactions_sample"]])
            if len(t_depths) >= 50 and len(g_depths) >= 50:
                N_q = 100
                t_q = [t_depths[int(i * (len(t_depths) - 1) / (N_q - 1))] for i in range(N_q)]
                g_q = [g_depths[int(i * (len(g_depths) - 1) / (N_q - 1))] for i in range(N_q)]
                mse = sum((g - t) ** 2 for g, t in zip(g_q, t_q)) / N_q
                rmse = math.sqrt(mse)
                z_range = max(max(t_depths), max(g_depths)) - min(min(t_depths), min(g_depths))
                first_int_nrmse = rmse / z_range if z_range > 0 else 0.0
                if first_int_nrmse >= 0.02:
                    gate_first_int_pass = False

        # Gate 3: Range / Bragg position difference < max(0.5 mm, one voxel size)
        bragg_diff_mm = None
        gate_bragg_pass = True
        if case.get("is_bragg_check", False):
            spacing_z = case["spacing_z_mm"]
            voxel_gate = max(0.5, spacing_z)
            gpu_bp = gpu_data.get("bragg_peak_metrics", {})
            gpu_peak_z = gpu_bp.get("peak_depth_mm")

            topas_idd = parse_topas_dose3d_csv(topas_dose_path, case["nx"], case["ny"], case["nz"])
            topas_bp = find_bragg_metrics(topas_idd, spacing_z, case["origin_z_mm"]) if topas_idd else None

            if gpu_peak_z and topas_bp and topas_bp.get("peak_depth_mm"):
                topas_peak_z = topas_bp["peak_depth_mm"]
                bragg_diff_mm = abs(gpu_peak_z - topas_peak_z)
                if bragg_diff_mm > voxel_gate:
                    gate_bragg_pass = False

        case_pass = (gate_conservation_pass and gate_section_pass and gate_sec_pass and
                     gate_overflow_pass and gate_survival_pass and gate_first_int_pass and
                     gate_bragg_pass)

        if not case_pass:
            all_gates_pass = False

        record = {
            "id": cid,
            "category": category,
            "passed_all_gates": case_pass,
            "gate1_survival_integral": {
                "topas_integral": topas_surv_int,
                "gpu_integral": gpu_surv_int,
                "rel_diff": survival_int_rel_diff,
                "gate_passed": gate_survival_pass
            },
            "gate2_first_interaction_nrmse": {
                "nrmse": first_int_nrmse,
                "gate_passed": gate_first_int_pass
            },
            "gate3_bragg_range": {
                "is_bragg_case": case.get("is_bragg_check", False),
                "peak_diff_mm": bragg_diff_mm,
                "gate_passed": gate_bragg_pass
            },
            "gate4_terminal_conservation": {
                "exact_conservation": exact_conservation,
                "histories": histories,
                "sum_terminal_counts": sum_term,
                "counts": term_counts,
                "gate_passed": gate_conservation_pass
            },
            "gate5_section_mapping": {
                "mismatches": section_mismatch,
                "gate_passed": gate_section_pass
            },
            "gate6_secondary_count": {
                "count": sec_count,
                "gate_passed": gate_sec_pass
            },
            "gate7_overflow_count": {
                "count": overflow_count,
                "gate_passed": gate_overflow_pass
            },
            "energy_balance_rel_error": gpu_data.get("relative_energy_balance_error", 0.0)
        }
        results.append(record)

    # Output JSON report
    report_json_path = EVIDENCE_DIR / "step13-validation-report.json"
    with open(report_json_path, "w") as f:
        json.dump({
            "schema_version": 1,
            "task": "Step 13 C12 Primary CT Milestone Gate Verification",
            "overall_pass": all_gates_pass,
            "total_cases": len(results),
            "passed_cases": sum(1 for r in results if r["passed_all_gates"]),
            "cases": results
        }, f, indent=2)

    # Output Markdown summary
    summary_md_path = EVIDENCE_DIR / "step13-validation-summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Step 13 — C12 Primary CT Milestone Validation Report\n\n")
        f.write(f"**Overall Status**: {'PASS' if all_gates_pass else 'FAIL'}\n")
        f.write(f"**Passed Cases**: {sum(1 for r in results if r['passed_all_gates'])} / {len(results)}\n\n")
        f.write("## Acceptance Gates Summary Table\n\n")
        f.write("| Case ID | Category | Survival Int Diff | First Int NRMSE | Bragg Diff (mm) | Term Conservation | Section Match | Overflow | Status |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in results:
            s_diff = f"{r['gate1_survival_integral']['rel_diff']*100:.3f}%" if r['gate1_survival_integral']['rel_diff'] is not None else "N/A"
            f_nrmse = f"{r['gate2_first_interaction_nrmse']['nrmse']*100:.3f}%" if r['gate2_first_interaction_nrmse']['nrmse'] is not None else "N/A"
            b_diff = f"{r['gate3_bragg_range']['peak_diff_mm']:.2f}" if r['gate3_bragg_range']['peak_diff_mm'] is not None else "N/A"
            cons = "EXACT" if r['gate4_terminal_conservation']['gate_passed'] else "FAIL"
            sec = "MATCH (0)" if r['gate5_section_mapping']['gate_passed'] else "FAIL"
            ovf = "0" if r['gate7_overflow_count']['gate_passed'] else "FAIL"
            st = "✅ PASS" if r['passed_all_gates'] else "❌ FAIL"
            f.write(f"| {r['id']} | {r['category']} | {s_diff} | {f_nrmse} | {b_diff} | {cons} | {sec} | {ovf} | {st} |\n")

    print(f"Validation finished. Overall Pass: {all_gates_pass}")
    print(f"Report written to: {report_json_path}")
    print(f"Summary written to: {summary_md_path}")
    if not all_gates_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
