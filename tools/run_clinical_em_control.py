"""Clinical-width CT EM-only control; local Slurm TOPAS and local GPU only."""
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
import yaml
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import sha,config_write
from evaluate_topas10x_gpu_gamma import pass_mask


def main():
    data=Path('/mnt/sda/wuwei');root=data/'ct_clinical_em_control_20260907'
    source=data/'ct_single_spot_residual_20260906/high'
    binary=Path('build/oneapi-nvidia-electron-bounds/carbon_mc').resolve()
    topas=Path('/home/wuwei/topas/topas-build/topas')
    if sha(binary)!='253e9ce8e3a5b15cafd111013438573d12cccf9d7b7d23b5167e47be71fca4fc':raise ValueError('Frozen GPU changed')
    # This entrypoint must be invoked only after resource review; refuse other
    # user jobs instead of overbooking unknown CPU/memory allocations.
    active=subprocess.check_output(['squeue','-u','wuwei','-h','-o','%i'],text=True).strip()
    if active:raise RuntimeError('Review existing Slurm allocation first: '+active)
    root.mkdir(exist_ok=False)
    text=(source/'topas.txt').read_text()
    for pattern,replacement in [
        (r'^i:Ts/NumberOfThreads\s*=.*$', 'i:Ts/NumberOfThreads = 48'),
        (r'^sv:Ph/Default/Modules\s*=.*$', 'sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"'),
        (r'^s:Sc/OSMK_Dtotal/OutputFile\s*=.*$',f's:Sc/OSMK_Dtotal/OutputFile = "{root}/dose_topas"')]:
        text,n=re.subn(pattern,replacement,text,flags=re.M)
        if n!=1:raise ValueError('Template replacement count')
    (root/'topas.txt').write_text(text)
    (root/'run.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=ct_clinical_em
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{topas} {root}/topas.txt
''')
    cfg=yaml.safe_load((source/'gpu_step025.yaml').read_text())
    cfg.update(enable_inelastic=False,enable_secondary_transport=False)
    config_write(root/'gpu.yaml',cfg)
    pins={str(p):sha(p) for p in (source/'topas.txt',source/'gpu_step025.yaml',root/'topas.txt',
        root/'run.slurm',root/'gpu.yaml',topas,binary,Path(cfg['tps_spots_file']),Path(cfg['tps_beam_model_file']),Path(cfg['ct_grid_file']))}
    job=subprocess.check_output(['sbatch','--parsable',str(root/'run.slurm')],text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unrecognized Slurm job id')
    with (root/'manifest.json').open('x') as f:json.dump(dict(job=job,pins=pins,histories=300000,cpus=48,memory_GiB=32),f,indent=2)
    print('TOPAS job',job,'submitted; starting LOCAL GPU',flush=True)
    run(root/'gpu.yaml',root/'gpu',binary,data/'schneider_electron_ct_runtime_r3_20260906/joint_response.csv')
    deadline=time.monotonic()+4*3600
    while True:
        status=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
        rows=[r.split('|') for r in status.splitlines() if r.startswith(job+'|')]
        if rows and rows[0][1]=='COMPLETED' and rows[0][2]=='0:0':break
        if rows and rows[0][1] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            raise RuntimeError('TOPAS failed: '+str(rows[0]))
        if time.monotonic()>deadline:raise TimeoutError('TOPAS observer deadline; inspect saved job')
        time.sleep(15)
    if 'Finalization:' not in (root/f'job_{job}.log').read_text():raise ValueError('Missing TOPAS finalization')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Input changed '+p)
    header=(root/'dose_topas.binheader').read_text()
    if [int(re.search(rf'# {a} in (\d+) bins',header)[1]) for a in 'XYZ']!=[960,607,42]:raise ValueError('Grid mismatch')
    ref=np.fromfile(root/'dose_topas.bin','<f8').reshape((42,607,960))
    g=np.fromfile(root/'gpu/dose.raw','<f4').reshape(ref.shape).astype(float)
    if not np.isfinite(ref).all() or ref.max()<=0 or not np.isfinite(g).all():raise ValueError('Invalid dose')
    pts=np.argwhere(ref>=.1*ref.max());ix=tuple(pts.T)
    metrics=dict(rms_error_pct_peak=float(np.sqrt(np.mean(((g[ix]-ref[ix])/ref.max())**2))*100),sum_ratio=float(g.sum()/ref.sum()))
    for dd,dta in ((3,3),(2,2),(1,1),(3,0)):
        for local in (False,True):
            metrics[f'{"local" if local else "global"}_{dd}{dta}']=100*float(pass_mask(g,ref,pts,np.array([2.,.5,.5]),dd,dta,local).mean())
    with (root/'comparison.json').open('x') as f:json.dump(dict(status='CLINICAL_EM_CONTROL_NOT_PRODUCTION',metrics=metrics,
        histories=300000,job=job,reference_sha256=sha(root/'dose_topas.bin'),gpu_sha256=sha(root/'gpu/dose.raw'),
        limitation='Nuclear-off mechanism control; includes electron response approximation. Not a production dose.'),f,indent=2)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
