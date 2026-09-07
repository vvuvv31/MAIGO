"""Two-seed average and noise diagnostic; never fit reference or spatial dose."""
import json
from pathlib import Path
import numpy as np
from evaluate_topas10x_gpu_gamma import pass_mask
from run_topas10x_gpu_benchmark import sha


def main():
    data=Path('/mnt/sda/wuwei')
    roots=[data/'ct_secondary_exact_faces_20260907',data/'ct_secondary_exact_faces_seed9196301_20260907']
    out=roots[0]/'two_seed_analysis.json'
    if out.exists():raise FileExistsError(out)
    source=data/'ct_single_spot_residual_20260906/high/dose_topas.bin'
    ref=np.fromfile(source,'<f8').reshape((42,607,960))
    pts=np.argwhere(ref>=.1*ref.max());ix=tuple(pts.T);rr=ref[ix]
    results={}
    for tag in ('off','on'):
        arrays=[]
        for root in roots:
            p=root/tag;r=json.loads((p/'run_report.json').read_text())
            if r['histories']!=300000 or sha(p/'dose.raw')!=r['dose_sha256']:raise ValueError('Run pin/count')
            arrays.append(np.fromfile(p/'dose.raw','<f4').reshape(ref.shape).astype(float))
        avg=(arrays[0]+arrays[1])*.5
        err=(avg[ix]-rr)/rr
        noise=(arrays[0][ix]-arrays[1][ix])/(2*rr)
        result=dict(relative_error_rms_pct=float(np.sqrt(np.mean(err**2))*100),
            noise_of_mean_rms_pct=float(np.sqrt(np.mean(noise**2))*100))
        for dd,dta in ((1,1),(3,0)):
            for local in (False,True):
                result[f'{"local" if local else "global"}_{dd}{dta}']=100*float(pass_mask(avg,ref,pts,np.array([2.,.5,.5]),dd,dta,local).mean())
        results[tag]=result
    output=dict(status='TWO_SEED_DIAGNOSTIC_NOT_PRODUCTION',results=results,
        histories_per_side=600000,reference_histories=300000,scale=.5,
        reference_sha256=sha(source),limitation='Exact history normalization of two-seed sum; not a production Gamma or zero-noise extrapolation')
    with out.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
