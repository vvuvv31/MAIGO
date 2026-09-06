"""Frozen-ROI reference and candidate acceptance. No fit, scale or bin removal."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_air_tissue_interface_pilot import analyze,gpu
from analyze_longitudinal_holdout import sha

def compare(roots,candidate="gpu_mass_candidate"):
    reports=[analyze(root) for root in roots]
    if len(reports)!=2:raise ValueError("Two buffer thicknesses required")
    reference_ok=all(r["status"]=="BASELINE_INTERFACE_DIAGNOSIS_ONLY" for r in reports)
    out={}
    for name in reports[0]["cases"]:
        c0,c1=[r["cases"][name] for r in reports]
        buffer_delta=[abs(a["topas_Gy_per_primary"]/b["topas_Gy_per_primary"]-1)
                      for a,b in zip(c0["windows"],c1["windows"])]
        reference_ok &= max(buffer_delta)<=.01
        variants={}
        for root,r in zip(roots,reports):
            case=r["cases"][name];manifest=json.loads((root/"manifest.json").read_text())
            spec=manifest["cases"][name];nz=2*sum(s["length_mm"] for s in spec["segments"])
            z=(np.arange(nz)+.5)*.5-spec["interface_mm"]
            truth=np.array(case["interface_profiles"]["topas_Gy_per_primary"])
            roi=(z>=-5)&(z<5)
            for label in ("gpu","gpu_candidate",candidate):
                folder=root/name/label
                if not (folder/"dose.mhd").exists():continue
                actual=gpu(folder/"dose.mhd",nz)/manifest["histories"]
                errors=[]
                for w in case["windows"]:
                    lo,hi=w["relative_depth_mm"]
                    errors.append(float(actual[(z>=lo)&(z<hi)].sum()/w["topas_Gy_per_primary"]-1))
                l=json.loads((folder/"out/run/energy_ledger.json").read_text())
                q=json.loads((folder/"out/run/quality_report.json").read_text())
                joint=bool(l.get("electron_joint_response",{}).get("queries",0))
                expected=[] if label=="gpu" else (["unvalidated_electron_joint_response"] if joint else ["unvalidated_longitudinal_candidate"])
                numerical_ok=[f["code"] for f in q["failures"]]==expected
                kernel=l.get("electron_joint_response" if joint else "longitudinal_candidate",{})
                numerical_ok &= kernel.get("domain_misses",0)==0
                numerical_ok &= kernel.get("invalid_marches",0)==0 and kernel.get("out_of_domain_queries",0)==0
                entry=dict(window_relative_errors=errors,
                    bin_relative_errors=(actual[roi]/truth-1).tolist(),
                    max_abs_bin_error=float(np.max(np.abs(actual[roi]/truth-1))),
                    numerical_ok=bool(numerical_ok),quality_accepted=q["accepted"],kernel=kernel,
                    artifacts={str(p):sha(p) for p in [folder/"dose.raw",folder/"run.yaml",folder/"out/run/quality_report.json",folder/"out/run/energy_ledger.json"]})
                entry["interface_error_gate"]=bool(numerical_ok and max(abs(x) for x in errors)<=.01 and entry["max_abs_bin_error"]<=.02)
                variants[str(root)+"/"+label]=entry
        out[name]=dict(buffer_window_relative_differences=buffer_delta,variants=variants)
    candidates=[v for c in out.values() for k,v in c["variants"].items() if k.endswith("/"+candidate)]
    complete=len(candidates)==2*len(roots)
    passed=reference_ok and complete and all(c["interface_error_gate"] for c in candidates)
    return dict(status="PASS_NARROW_INTERFACE_DIAGNOSTIC" if passed else "FAILED_OR_INCOMPLETE",
        reference_gate=bool(reference_ok),candidate_complete=complete,
        thresholds=dict(replica_and_buffer=.01,window_error=.01,individual_half_mm_bin_error=.02),
        scope="175 MeV/u, EM-only HU -1000/100 axis-aligned interfaces; no patient/general-CT validation",
        cases=out,reference_reports=reports,patient_gamma_improvement_proven=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("roots",nargs=2,type=Path)
    p.add_argument("--candidate",default="gpu_mass_candidate")
    p.add_argument("--output",type=Path)
    a=p.parse_args();r=compare(a.roots,a.candidate)
    text=json.dumps(r,indent=2)+"\n"
    if a.output:
        with a.output.open("x") as f:f.write(text)
    else:print(text)
    raise SystemExit(0 if r["status"]=="PASS_NARROW_INTERFACE_DIAGNOSTIC" else 1)
