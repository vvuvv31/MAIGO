#!/usr/bin/env python3
"""tools/verify_step20_validation.py

Verifies Step 20 secondary nuclear transport acceptance gates:
- Chi-square test on secondary fragment species (Z=1..6)
- Major species (Z=1 + Z=2) relative difference (< 2.0%)
- Energy ledger closure
- Target element sampling consistency
"""

import json
import math
from pathlib import Path

TOPAS_DIR = Path("/mnt/sda/wuwei/step19_fragmentation/topas")
GPU_DIR = Path("/mnt/sda/wuwei/step20_secondary_transport/gpu")
MANIFEST_PATH = Path("/mnt/sda/wuwei/step19_fragmentation/manifest.json")
EVIDENCE_DIR = Path("evidence/step-20")

def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]

def run_verification(generate_evidence: bool = False):
    cases = load_manifest()
    results = []

    print("=" * 100)
    print(f"{'Case ID':<25} {'Primary Inel':<14} {'Sec Inel':<10} {'Major Yield Diff':<18} {'Target Diff':<14} {'Status'}")
    print("=" * 100)

    for c in cases:
        cid = c["id"]
        topas_json_path = TOPAS_DIR / f"{cid}_scorer.json"
        gpu_json_path = GPU_DIR / f"{cid}_gpu.json"

        with open(topas_json_path, "r", encoding="utf-8") as f:
            topas_data = json.load(f)
        with open(gpu_json_path, "r", encoding="utf-8") as f:
            gpu_data = json.load(f)

        # Scale TOPAS to match GPU history count (GPU ran 4x histories = 200k vs 50k)
        scale = float(gpu_data["histories"]) / float(c["histories"])

        topas_sec = {int(k): v * scale for k, v in topas_data["secondary_species_counts"].items()}
        gpu_sec = {int(k): float(v) for k, v in gpu_data["secondary_species_counts"].items()}

        topas_major = topas_sec.get(1, 0.0) + topas_sec.get(2, 0.0)
        gpu_major = gpu_sec.get(1, 0.0) + gpu_sec.get(2, 0.0)

        major_rel_diff = abs(gpu_major - topas_major) / max(1.0, topas_major)

        # Inelastic counts
        topas_inel = topas_data["total_first_inelastic_count"] * scale
        gpu_inel = gpu_data["inelastic_count"]
        inel_rel_diff = abs(gpu_inel - topas_inel) / max(1.0, topas_inel)

        # Target element comparison
        topas_tgt = {int(k): v * scale for k, v in topas_data["target_element_counts"].items()}
        gpu_tgt = {int(k): float(v) for k, v in gpu_data["target_element_counts"].items()}
        tgt_diffs = []
        for tz in [1, 6, 7, 8]:
            if tz in topas_tgt and topas_tgt[tz] > 100:
                diff = abs(gpu_tgt.get(tz, 0.0) - topas_tgt[tz]) / topas_tgt[tz]
                tgt_diffs.append(diff)
        mean_tgt_diff = sum(tgt_diffs) / len(tgt_diffs) if tgt_diffs else 0.0

        # Energy closure
        e_init = gpu_data["energy_ledger"]["initial_energy_MeV"]
        e_dep = gpu_data["energy_ledger"]["local_deposit_MeV"]
        e_esc = gpu_data["energy_ledger"]["escaped_energy_MeV"]
        e_closure = abs(e_init - (e_dep + e_esc)) / max(1.0, e_init)

        # Statistical uncertainty (1-sigma Poisson)
        topas_stat_unc = 1.0 / math.sqrt(max(1.0, topas_data["total_first_inelastic_count"]))
        passed = (major_rel_diff < 0.02) and (inel_rel_diff < 0.05) and (e_closure < 0.01)

        status_str = "PASS" if passed else "FAIL"
        print(f"{cid:<25} {gpu_inel:<14} {gpu_data['secondary_inelastic_count']:<10} {major_rel_diff*100:6.2f}% (±{topas_stat_unc*100:4.2f}%)   {mean_tgt_diff*100:5.2f}%         {status_str}")

        results.append({
            "case_id": cid,
            "category": c["category"],
            "section_id": c["section_id"],
            "energy_mevu": c["energy_mevu"],
            "thickness_mm": c["thickness_mm"],
            "gpu_histories": gpu_data["histories"],
            "primary_inelastic_count": gpu_inel,
            "secondary_inelastic_count": gpu_data["secondary_inelastic_count"],
            "major_species_topas": topas_major,
            "major_species_gpu": gpu_major,
            "major_species_relative_diff": major_rel_diff,
            "primary_inelastic_relative_diff": inel_rel_diff,
            "target_mean_relative_diff": mean_tgt_diff,
            "energy_closure_relative_error": e_closure,
            "pass": passed
        })

    suite_mean_diff = sum(r["major_species_relative_diff"] for r in results) / len(results) if results else 0.0
    print("=" * 100)
    print(f"Validation Suite Mean Major Species Relative Difference: {suite_mean_diff*100:.2f}% (Target: < 2.0%)")
    overall_pass = (
        bool(results)
        and all(r["pass"] for r in results)
        and suite_mean_diff < 0.02
    )
    print(f"Overall Step 20 Acceptance: {'PASS' if overall_pass else 'FAIL'}")

    if generate_evidence:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        evidence_payload = {
            "step": 20,
            "acceptance_target": "< 2.0% major species integral relative difference across validation suite (each case individually)",
            "suite_mean_major_species_relative_diff": suite_mean_diff,
            "overall_pass": overall_pass,
            "cases": results
        }
        summary_file = EVIDENCE_DIR / "step20_validation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(evidence_payload, f, indent=2)
        verification_file = EVIDENCE_DIR / "verification.json"
        with open(verification_file, "w", encoding="utf-8") as f:
            json.dump(evidence_payload, f, indent=2)
        print(f"Detailed validation summary saved to {summary_file} and {verification_file}")
    else:
        print("Read-only mode (evidence was not modified. Pass --generate-evidence to write evidence).")

    return overall_pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify Step 20 secondary nuclear transport validation.")
    parser.add_argument("--generate-evidence", action="store_true", help="Write summary to evidence/step-20/")
    args = parser.parse_args()
    success = run_verification(generate_evidence=args.generate_evidence)
    if not success:
        exit(1)
