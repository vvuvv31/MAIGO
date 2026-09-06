"""Local-only paired interface diagnostic; never promotes a candidate to accepted."""
import argparse,json,os,subprocess
from pathlib import Path
from analyze_longitudinal_holdout import sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--candidate",action="store_true")
    p.add_argument("--interface-mass",action="store_true")
    p.add_argument("--joint-response",type=Path)
    p.add_argument("--label")
    p.add_argument("--step-mm",type=float)
    p.add_argument("--seed",type=int)
    a=p.parse_args();root=a.root.resolve();binary=a.binary.resolve()
    if a.joint_response:
        if a.interface_mass or a.candidate:p.error("joint response excludes old candidate modes")
        a.candidate=True
        a.joint_response=a.joint_response.resolve()
    if a.interface_mass and not a.candidate:p.error("--interface-mass requires --candidate")
    if a.label and ('/' in a.label or a.label in ('.','..')):p.error("label must be a directory name")
    if a.step_mm is not None and (not a.candidate or not 0<a.step_mm<=.5):p.error("step must be in (0,.5]")
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(["python3",str(repo/"tools/verify_schneider_v2_1_data.py")],check=True)
    env=dict(os.environ,ONEAPI_DEVICE_SELECTOR="cuda:*")
    env["LD_LIBRARY_PATH"]="/home/wuwei/sycl_workspace/llvm/build/install/lib:"+env.get("LD_LIBRARY_PATH","")
    report={}
    for name in json.loads((root/"manifest.json").read_text())["cases"]:
        base=root/name/"gpu"
        label="gpu_mass_candidate" if a.interface_mass else "gpu_candidate"
        if a.label:label=a.label
        dest=root/name/(label if a.candidate or a.label else "gpu")
        if a.candidate or a.label:
            dest.mkdir(exist_ok=False)
            text=(base/"run.yaml").read_text().replace(str(base),str(dest))
            if a.seed is not None:
                lines=text.splitlines()
                if sum(x.startswith('random_seed:') for x in lines)!=1:raise ValueError('Missing/duplicate seed')
                text='\n'.join(f'random_seed: {a.seed}' if x.startswith('random_seed:') else x for x in lines)+'\n'
            if a.joint_response:
                text+=f"\nct_electron_joint_response_diagnostic_file: {a.joint_response}\nct_electron_joint_response_sha256: {sha(a.joint_response)}\nct_electron_joint_response_metadata_sha256: {sha(a.joint_response.with_suffix('.metadata.json'))}\n"
            elif a.candidate:text+=f"\nct_schneider_delta_longitudinal_file: {repo}/data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv\n"
            if a.interface_mass:text+="ct_longitudinal_interface_mass_diagnostic: true\n"
            if a.step_mm is not None:text=text.replace("maximum_step_mm: 0.5",f"maximum_step_mm: {a.step_mm}")
            (dest/"run.yaml").write_text(text)
        if (dest/"dose.raw").exists():raise ValueError("Refusing to overwrite dose")
        with (dest/"run.log").open("x") as log:
            run=subprocess.run([str(binary),"--config",str(dest/"run.yaml"),"--device","cuda"],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT)
        q=json.loads((dest/"out/run/quality_report.json").read_text())
        failures=[v["code"] for v in q["failures"]]
        expected=["unvalidated_longitudinal_candidate"] if a.candidate else []
        if a.joint_response:expected=["unvalidated_electron_joint_response"]
        if failures!=expected or q["accepted"]==a.candidate or not (dest/"dose.raw").exists():
            raise ValueError(f"Unexpected quality: {name}: {q}")
        if (run.returncode==0)==a.candidate:raise ValueError("Unexpected return status")
        report[name]=dict(binary_sha256=sha(binary),config_sha256=sha(dest/"run.yaml"),
                          raw_sha256=sha(dest/"dose.raw"),quality=q)
        print(name,"completed; candidate remains unvalidated" if a.candidate else "baseline accepted",flush=True)
    with (root/(label+"_runs.json" if a.candidate or a.label else "gpu_baseline_runs.json")).open('x') as f:
        f.write(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
