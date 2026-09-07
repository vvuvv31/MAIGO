"""3D origin diagnostic at frozen clinical single spot; no transport changes."""
import json
import argparse
from pathlib import Path
import numpy as np
import yaml
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import sha,config_write
from evaluate_topas10x_gpu_gamma import pass_mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/mnt/sda/wuwei/ct_local_origin_20260907'))
    data=Path('/mnt/sda/wuwei');out=parser.parse_args().out.resolve()
    prior=data/'ct_secondary_source_score_20260907/off'
    refpath=data/'ct_single_spot_residual_20260906/high/dose_topas.bin'
    cfg=yaml.safe_load((prior/'run.yaml').read_text())
    cfg.update(enable_charged_origin_voxel_scoring=True,
        charged_origin_voxel_mhd_output_prefix=str(out/'origin'),
        charged_origin_voxel_output_file='',charged_origin_voxel_dose_Gy_output_file='')
    out.mkdir(exist_ok=False);config_write(out/'source.yaml',cfg)
    binary=Path('build/oneapi-nvidia-secondary-source-score/carbon_mc').resolve()
    joint=data/'schneider_electron_ct_runtime_r3_20260906/joint_response.csv'
    run(out/'source.yaml',out/'run',binary,joint)
    shape=(42,607,960)
    base=np.fromfile(prior/'dose.raw','<f4').reshape(shape)
    g=np.fromfile(out/'run/dose.raw','<f4').reshape(shape)
    delta=float(np.max(np.abs(g-base))/base.max())
    if delta>2e-7:raise ValueError('Origin changed total dose beyond float-storage tolerance')
    ref=np.fromfile(refpath,'<f8').reshape(shape);mask=ref>=.1*ref.max();pts=np.argwhere(mask);ix=tuple(pts.T)
    labels=('primary','secondary_carbon','secondary_boron','secondary_beryllium','secondary_lithium',
        'secondary_helium','secondary_proton','secondary_other_charged')
    origins={};summed=np.zeros(shape,float);pins={}
    for label in labels:
        p=out/f'origin_{label}.raw';a=np.fromfile(p,'<f4').reshape(shape)
        summed+=a;origins[label]=a[ix].astype(float);pins[str(p)]=sha(p)
    closure=float(np.max(np.abs(summed-g))/g.max())
    if closure>2e-6:raise ValueError('Origin categories do not close')
    with Path(cfg['ct_grid_file']).open('rb') as f:
        f.seek(44+g.size*4);section=np.fromfile(f,'u1',count=g.size).reshape(shape)[ix]
    rr=ref[ix];gg=g[ix].astype(float);err=gg-rr
    sec=sum(v for k,v in origins.items() if k!='primary')
    fail={f'local_{dd}{dta}':~pass_mask(g,ref,pts,np.array([2.,.5,.5]),dd,dta,True) for dd,dta in ((1,1),(3,0))}
    def stats(s):
        if not s.any():return {'voxels':0}
        return dict(voxels=int(s.sum()),mean_relative_error_pct=float(np.mean(err[s]/rr[s])*100),
            fractions={k:float(v[s].sum()/gg[s].sum()) for k,v in origins.items()},
            hot_excess_exceeds_all_gpu_secondary_dose=int((s & (err>sec)).sum()))
    result=dict(status='ORIGIN_DIAGNOSTIC_NOT_PRODUCTION',histories=300000,
        total_dose_max_delta_pct_peak=100*delta,origin_closure_max_pct_peak=100*closure,
        all=stats(np.ones(len(rr),bool)),failures={k:stats(v) for k,v in fail.items()},
        lung_failure_bands={f'{key}/{band}':stats(f & (section==1) & (rr>=lo*ref.max()) & (rr<hi*ref.max()))
            for key,f in fail.items() for band,lo,hi in [('10-20',.1,.2),('20-50',.2,.5),('>=50',.5,1.01)]},
        pins={**pins,str(refpath):sha(refpath),str(prior/'dose.raw'):sha(prior/'dose.raw')},
        limitation='GPU dose fractions are not error fractions; no TOPAS per-origin reference. Primary category includes its redistributed electron response.')
    with (out/'analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('total_dose_max_delta_pct_peak','origin_closure_max_pct_peak','failures','lung_failure_bands')},indent=2),flush=True)


if __name__=='__main__':main()
