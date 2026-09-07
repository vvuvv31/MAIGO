"""Paired 300k CT geometry repair probe; all physics data frozen."""
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
    parser.add_argument('--seed',type=int)
    parser.add_argument('--no-secondary-mcs',action='store_true')
    parser.add_argument('--binary',type=Path,default=Path('build/oneapi-nvidia-secondary-faces/carbon_mc'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--baseline-off',type=Path)
    parser.add_argument('--mode',choices=('faces','secondary-sp'),default='faces')
    args=parser.parse_args()
    if args.mode!='faces' and args.no_secondary_mcs:raise ValueError('Do not mix diagnostic mechanisms')
    data=Path('/mnt/sda/wuwei')
    source=data/'ct_single_spot_residual_20260906/high'
    root=data/'ct_secondary_exact_faces_20260907'
    if args.seed is not None:root=data/f'ct_secondary_exact_faces_seed{args.seed}_20260907'
    if args.no_secondary_mcs:root=root.with_name(root.name+'_no_secondary_mcs')
    if args.out:root=args.out.resolve()
    binary=args.binary.resolve()
    joint=data/'schneider_electron_ct_runtime_r3_20260906/joint_response.csv'
    cfg=yaml.safe_load((source/'gpu_step025.yaml').read_text())
    if args.seed is not None:cfg['random_seed']=args.seed
    if args.no_secondary_mcs:cfg['ct_secondary_mcs_off_diagnostic']=True
    root.mkdir(exist_ok=False)
    pins={str(p):sha(p) for p in (binary,source/'dose_topas.bin',source/'dose_topas.binheader',source/'gpu_step025.yaml')}
    with (root/'inputs.json').open('x') as f:json.dump(pins,f,indent=2)
    for tag,on in [('off',False),('on',True)]:
        c=dict(cfg,ct_secondary_exact_faces_diagnostic=on if args.mode=='faces' else False)
        if args.mode=='secondary-sp':c['ct_secondary_schneider_sp_diagnostic']=on
        config_write(root/(tag+'.yaml'),c)
        run(root/(tag+'.yaml'),root/tag,binary,joint)
        if not on and args.baseline_off and sha(root/tag/'dose.raw')!=sha(args.baseline_off):
            raise ValueError('OFF differs from pinned previous OFF dose')
        if not on and args.seed is None and not args.no_secondary_mcs and sha(root/tag/'dose.raw')!=sha(source/'step025/dose.raw'):
            raise ValueError('OFF differs from frozen bounds: isolate binary changes first')
    ref=np.fromfile(source/'dose_topas.bin','<f8').reshape((42,607,960))
    pts=np.argwhere(ref>=.1*ref.max());index=tuple(pts.T);results={}
    for tag in ('off','on'):
        p=root/tag;ledger=json.loads((p/'energy_ledger.json').read_text())
        flag='ct_secondary_exact_faces_diagnostic' if args.mode=='faces' else 'ct_secondary_schneider_sp_diagnostic'
        if ledger[flag]!=(tag=='on'):raise ValueError('Flag not recorded')
        if args.no_secondary_mcs and not ledger.get('ct_secondary_mcs_off_diagnostic',False):raise ValueError('MCS counterfactual not enabled')
        g=np.fromfile(p/'dose.raw','<f4').reshape(ref.shape).astype(float)
        result=dict(rms_error_pct_peak=float(np.sqrt(np.mean(((g[index]-ref[index])/ref.max())**2))*100),
            sum_ratio=float(g.sum()/ref.sum()))
        for dd,dta in ((1,1),(3,0)):
            for local in (False,True):
                result[f'{"local" if local else "global"}_{dd}{dta}']=100*float(pass_mask(g,ref,pts,np.array([2.,.5,.5]),dd,dta,local).mean())
        results[tag]=result
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Input changed')
    output=dict(status='NO_SECONDARY_MCS_COUNTERFACTUAL' if args.no_secondary_mcs else 'SMOKE_ONLY_NOT_PRODUCTION',results=results,pins=pins,
        histories=300000,seed=cfg['random_seed'],mode=args.mode,limitation='Single clinical spot; no patient-wide accuracy claim')
    with (root/'comparison.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
