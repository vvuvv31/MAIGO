"""Compare completed 3-D CT single-spot runs, with no fitted normalization."""
import json
import re
import argparse
import subprocess
from pathlib import Path
import numpy as np
from evaluate_topas10x_gpu_gamma import pass_mask
from run_topas10x_gpu_benchmark import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/mnt/sda/wuwei/ct_single_spot_residual_20260906'))
    root=parser.parse_args().root
    manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['inputs'].items():
        if sha(p)!=h:raise ValueError('Changed input '+p)
    result={}
    for case in manifest['cases']:
        p=Path(case['directory']);header=(p/'dose_topas.binheader').read_text()
        dims=[int(re.search(rf'# {a} in (\d+) bins',header)[1]) for a in 'XYZ']
        if dims!=[960,607,42]:raise ValueError('Unexpected TOPAS geometry')
        logs=list(p.glob('job_*.log'))
        finished=[]
        for log in logs:
            job=re.fullmatch(r'job_(\d+)\.log',log.name)
            if job:
                accounting=subprocess.check_output(['sacct','-j',job[1],'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
                if f'{job[1]}|COMPLETED|0:0' in accounting.splitlines() and 'Finalization:' in log.read_text():finished.append(log)
        if len(finished)!=1:
            raise ValueError('TOPAS not finished: '+str(p))
        ref=np.fromfile(p/'dose_topas.bin','<f8').reshape(tuple(dims[::-1]))
        if not np.isfinite(ref).all() or ref.max()<=0:raise ValueError('Invalid reference')
        pts=np.argwhere(ref>=.1*ref.max());out={}
        for tag in ('base','step025'):
            d=p/tag;r=json.loads((d/'run_report.json').read_text())
            if r['histories']!=case.get('histories',300000) or sha(d/'dose.raw')!=r['dose_sha256']:raise ValueError('GPU pin/count')
            g=np.fromfile(d/'dose.raw','<f4').reshape(ref.shape)
            err=(g-ref)/ref.max();m=ref>=.1*ref.max()
            profile_r=ref.sum(axis=(0,2));profile_g=g.sum(axis=(0,2))
            metrics=dict(rms_error_pct_peak=float(np.sqrt(np.mean(err[m]**2))*100),
                mean_error_pct_peak=float(err[m].mean()*100),sum_ratio=float(g.sum(dtype=float)/ref.sum()),
                idd_peak_delta_mm=float((np.argmax(profile_g)-np.argmax(profile_r))*.5))
            for dd,dta in ((3,0),(1,1)):
                metrics[f'global_{dd}{dta}']=100*float(pass_mask(g,ref,pts,np.array([2.,.5,.5]),dd,dta,False).mean())
            out[tag]=metrics
        result[case['label']]=out
        print(case['label'],json.dumps(out),flush=True)
    target=root/'comparison.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
