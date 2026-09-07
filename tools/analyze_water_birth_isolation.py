"""Verify same-binary birth scorer noninvasiveness before interpreting products."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from analyze_water_helium_birth import dose_identity
from run_topas10x_gpu_benchmark import sha

def check(root,job,mode):
    state=subprocess.check_output(['sacct','-j',str(job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{job}|COMPLETED|0:0' not in state.splitlines():raise ValueError('Job not completed')
    m=json.loads((root/'manifest.json').read_text())
    if m['mode']!=mode:raise ValueError('Wrong control mode')
    for path,h in m['pins'].items():
        if sha(Path(path))!=h:raise ValueError('Changed input: '+path)
    return m,np.fromfile(root/'dose_topas.bin','<f8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--birth',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--birth-job',type=int,required=True);p.add_argument('--control-job',type=int,required=True)
    a=p.parse_args();bm,b=check(a.birth,a.birth_job,'birth');cm,c=check(a.control,a.control_job,'control')
    if bm['baseline']!=cm['baseline'] or bm['histories']!=cm['histories']:raise ValueError('Different controls')
    baseline=np.fromfile(bm['baseline'],'<f8')
    pairs={'control_vs_original_reference':dose_identity(baseline,c),'birth_vs_control':dose_identity(c,b)}
    passed=all(v['total_unchanged'] for v in pairs.values())
    files=[a.birth/'dose_topas.bin',a.control/'dose_topas.bin',Path(bm['baseline']),Path(__file__).resolve()]
    result=dict(status='PASS' if passed else 'FAIL',jobs=[a.birth_job,a.control_job],comparisons=pairs,
        pins={str(f):sha(f) for f in files})
    with (a.birth/'isolation.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
