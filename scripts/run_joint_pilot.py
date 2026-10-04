#!/usr/bin/env python3
"""Run the opt-in lung joint electron pilot, or its matched legacy control."""
import argparse, json, os, subprocess, time, hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=ROOT/'build/carbon_mc')
    p.add_argument('--package',type=Path)
    p.add_argument('--histories',type=int,default=1000000)
    p.add_argument('--seed',type=int,default=1002450001)
    p.add_argument('--batches',type=int,default=1)
    a=p.parse_args();assert a.histories>0 and a.batches>0
    d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    for name in ['data','inputs']:(d/name).symlink_to(ROOT/name,target_is_directory=True)
    c=yaml.safe_load((ROOT/'config/proton_ct_minibeam.yaml').read_text())
    c.update(number_of_histories=a.histories,random_seed=a.seed)
    if a.package:
        c.update(material_electron_joint_pilot_file=str(a.package.resolve(strict=True)),material_electron_joint_pilot_sha256=sha(a.package))
    (d/'case.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    command=[str(a.binary.resolve()),'--config','case.yaml']
    if a.batches>1:command+=['--histories-per-batch',str(a.histories),'--batch-count',str(a.batches),'--batch-output',str(d/'batches')]
    env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
    env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','')
    env['CARBON_RUNTIME_BREAKDOWN']='1'
    m=dict(command=command,binary_sha256=sha(a.binary),config_sha256=sha(d/'case.yaml'),histories_per_batch=a.histories,batches=a.batches,seed=a.seed)
    (d/'execution.json').write_text(json.dumps(m,indent=2))
    start=time.perf_counter()
    with (d/'gpu.log').open('x') as log:r=subprocess.run(command,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
    m.update(exit_code=r.returncode,wall_seconds=time.perf_counter()-start)
    (d/'execution.json').write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True)
    raise SystemExit(r.returncode)
if __name__=='__main__':main()
