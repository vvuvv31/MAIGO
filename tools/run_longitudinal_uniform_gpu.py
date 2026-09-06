"""Explicit opt-in local CUDA three-density experiment (no scheduler/GPU remote).
All outputs go to a fresh directory. Candidate runs must remain unaccepted.
"""
import argparse,json,os,struct,subprocess
from pathlib import Path
from analyze_longitudinal_holdout import gpu_curve,sha
from analyze_longitudinal_mass_thickness import metrics
from compile_schneider_delta_longitudinal import lateral_integral
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--density",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(["python3",str(repo/"tools/verify_schneider_v2_1_data.py")],check=True)
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=a.binary.resolve()
    env=dict(os.environ,ONEAPI_DEVICE_SELECTOR="cuda:*")
    env["LD_LIBRARY_PATH"]="/home/wuwei/sycl_workspace/llvm/build/install/lib:"+env.get("LD_LIBRARY_PATH","")
    results={}
    def run(case,text):
        case.mkdir(parents=True)
        (case/"run.yaml").write_text(text)
        r=subprocess.run([str(binary),"--config",str(case/"run.yaml"),"--device","cuda"],cwd=case,env=env,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (case/"run.log").write_text(r.stdout)
        return r
    for hu,old in [("hu1000",a.reference.resolve()),("hu975",a.density.resolve()/"hu975"),("hu951",a.density.resolve()/"hu951")]:
        truth=(lateral_integral(old/"topas_s1/dose.csv")+lateral_integral(old/"topas_s2/dose.csv"))/40000
        results[hu]={}
        for enabled in (False,True):
            label="on" if enabled else "off"
            case=root/hu/label
            text=(old/"gpu_long/run.yaml").read_text().replace(str(old/"gpu_long"),str(case))
            text+="\nct_longitudinal_homogeneous_density_diagnostic: "+str(enabled).lower()+"\n"
            r=run(case,text)
            q=json.loads((case/"out/run/quality_report.json").read_text())
            ledger=json.loads((case/"out/run/energy_ledger.json").read_text())
            failures=[f["code"] for f in q["failures"]]
            complete=(case/"dose.raw").exists()
            ok=r.returncode!=0 and failures==["unvalidated_longitudinal_candidate"] and complete
            if not enabled:
                ok &= sha(case/"dose.raw")==sha(old/"gpu_long/dose.raw")
            if enabled:
                ok &= ledger["E_schneider_primary_delta_longitudinal_moved_MeV"]>0
                ok &= ledger["longitudinal_candidate"]["invalid_marches"]==0
                ok &= ledger["longitudinal_candidate"]["out_of_domain_queries"]==0
            curve=gpu_curve(case)/20000
            m=metrics(curve,truth)
            if enabled:
                ok &= m["max_window_error"]<=.01 and m["max_rebin_error"]<=.02
            results[hu][label]=dict(numerical_and_pilot_gate=bool(ok),accepted=q["accepted"],
                failures=failures,raw_sha256=sha(case/"dose.raw"),metrics=m,
                moved_MeV=ledger["E_schneider_primary_delta_longitudinal_moved_MeV"],
                kernel=ledger["longitudinal_candidate"],config_sha256=sha(case/"run.yaml"))
            print(hu,label,"gate",ok,"max window",m["max_window_error"],flush=True)
    # Real launch rejection, not only a helper-unit test.
    template=(root/"hu975/on/run.yaml").read_text()
    gridpath=next(x.split(":",1)[1].strip() for x in template.splitlines() if x.startswith("ct_grid_file:"))
    original=Path(gridpath).read_bytes()
    for name in ("mixed_density","mixed_section"):
        content=bytearray(original)
        if name=="mixed_density":struct.pack_into("<f",content,48,0.06621015816926956)
        else:content[44+4*100*100*440]=1
        phantom=root/(name+".cctg");phantom.write_bytes(content)
        case=root/name
        text=template.replace(gridpath,str(phantom)).replace(str(root/"hu975/on"),str(case))
        r=run(case,text)
        results[name]=dict(rejected=r.returncode!=0 and "heterogeneous grid" in r.stdout,
                           grid_sha256=sha(phantom),config_sha256=sha(case/"run.yaml"))
    passed=all(v[label]["numerical_and_pilot_gate"] for k,v in results.items() if k.startswith("hu") for label in ("on","off"))
    passed &= all(results[k]["rejected"] for k in ("mixed_density","mixed_section"))
    report=dict(status="PASS_DIAGNOSTIC_ONLY" if passed else "FAILED",cases=results,
                binary_sha256=sha(binary),runner_sha256=sha(Path(__file__)),
                scope="Uniform 175 MeV/u section-0 only; no patient/interface authorization")
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if passed else 1

if __name__=="__main__":raise SystemExit(main())
