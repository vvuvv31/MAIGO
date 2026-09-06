"""Read-only density reference audit; never promotes the inactive candidate."""
import argparse
import json
from pathlib import Path
from analyze_longitudinal_holdout import analyze, sha

def inactive_scope(base_sha, candidate_sha, ledger):
    return (base_sha == candidate_sha and
            ledger["E_schneider_primary_delta_longitudinal_moved_MeV"] == 0 and
            ledger["E_schneider_primary_delta_longitudinal_escaped_scorer_MeV"] == 0 and
            ledger["longitudinal_candidate"]["invalid_marches"] == 0)

def audit(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for relative, expected in manifest["prepared_files"].items():
        if sha(root/relative) != expected:
            raise ValueError("Prepared input changed: " + relative)
    results = {}
    all_ok = True
    for label in ("hu975","hu951"):
        case = root/label
        report = analyze(case)
        ledger = json.loads((case/"gpu_long/out/run/energy_ledger.json").read_text())
        base_sha, candidate_sha = sha(case/"gpu_base/dose.raw"), sha(case/"gpu_long/dose.raw")
        scope_ok = inactive_scope(base_sha,candidate_sha,ledger)
        report["holdout_gate_not_applicable"] = report.pop("status")
        report["status"] = "REFERENCE_ONLY_UNSUPPORTED_DENSITY"
        report["scope_inactivity_verified"] = scope_ok
        report["density_g_cm3"] = json.loads((case/"campaign.json").read_text())["density_g_cm3"]
        # This is a data/numerical gate only, not a physical-kernel acceptance gate.
        report["reference_audit_ok"] = bool(report["quality_gate"] and report["noise_pilot_gate"] and scope_ok)
        all_ok &= report["reference_audit_ok"]
        results[label] = report
    return dict(status="REFERENCE_COLLECTED" if all_ok else "REFERENCE_AUDIT_FAILED",
                physical_density_model_validated=False, cases=results,
                manifest_sha256=sha(root/"manifest.json"),
                limitations=["No fixed-birth-family experiment or joint-response validation.",
                             "No source-density extrapolation or interface model enabled.",
                             "Current candidate is deliberately inactive at both densities."])

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.root),indent=2))
