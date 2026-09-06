"""Read-only domain-log reconciliation and energy/depth census."""
import argparse,json,math
from collections import Counter
from pathlib import Path
from analyze_longitudinal_holdout import sha

def summarize(ledger):
    log=ledger["longitudinal_domain_log"]
    diag=ledger["longitudinal_candidate"]
    records=log["records"]
    if log["truncated"] or len(records)!=diag["out_of_domain_queries"]:
        raise ValueError("Incomplete domain log")
    keys=[(r["history"],r["step"]) for r in records]
    if len(set(keys))!=len(keys):
        raise ValueError("Duplicate query")
    fields=["initial_energy_MeVu","query_energy_MeVu","step_start_z_mm","step_length_mm","retained_MeV"]
    if any(not math.isfinite(r[f]) for r in records for f in fields):
        raise ValueError("Nonfinite query")
    if any(r["retained_MeV"]<0 or r["step_length_mm"]<0 for r in records):
        raise ValueError("Negative energy/step")
    total=sum(r["retained_MeV"] for r in records)
    residual=total-diag["out_of_domain_local_energy_MeV"]
    # Existing counter truncates each positive deposit to integer microMeV.
    if not -1e-6 <= residual <= len(records)*1e-6+1e-6:
        raise ValueError("Record energy does not reconcile to fixed-point counter")
    count=Counter("below" if r["query_energy_MeVu"]<150 else
                  "above" if r["query_energy_MeVu"]>225 else "unexpected_in_domain"
                  for r in records)
    if count["unexpected_in_domain"]:
        raise ValueError("Domain log contains covered energy")
    return dict(record_count=len(records),unique_histories=len(set(r["history"] for r in records)),
                classification=dict(count),energy_counter_residual_MeV=residual,
                ranges={f:([min(r[f] for r in records),max(r[f] for r in records)] if records else None) for f in fields},
                queries_by_10mm_bin=dict(sorted(Counter(int(r["step_start_z_mm"]//10)*10 for r in records).items())),
                retained_MeV=total,
                all_initially_in_domain=all(150<=r["initial_energy_MeVu"]<=225 for r in records))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--before",type=Path,required=True)
    a=p.parse_args()
    path=a.root/"out/run/energy_ledger.json"
    result=summarize(json.loads(path.read_text()))
    result["raw_unchanged"]=sha(a.root/"dose.raw")==sha(a.before/"dose.raw")
    if not result["raw_unchanged"]:
        raise ValueError("Diagnostic changed 3-D raw dose")
    result["artifacts"]={str(p):sha(p) for p in [path,a.root/"run.yaml",a.root/"dose.raw",a.before/"dose.raw"]}
    print(json.dumps(result,indent=2))
