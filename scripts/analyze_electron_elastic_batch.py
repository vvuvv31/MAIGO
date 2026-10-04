#!/usr/bin/env python3
"""Full charged/photon trajectory screen; keep dose-budget uncertainty explicit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp, fisher_exact

DT=np.dtype([('position','<f4',3),*[(x,'<u4') for x in ('status','steps','batched','elastic','photons','guard')]])
assert DT.itemsize==36


def slit_score(displacement, valley=False):
    # Uniform 0.5mm source slit, 3.6mm pitch; integrate source position
    # analytically for a 0.5mm peak/valley ROI, no extra sampling noise.
    phase=(displacement+1.8)%3.6-1.8
    centers=[-1.8,1.8] if valley else [-3.6,0,3.6]
    return sum(np.maximum(0,np.minimum(phase+.25,c+.25)-np.maximum(phase-.25,c-.25))/.5 for c in centers)


def bounded_mean(x, delta):
    # Empirical Bernstein radius for IID bounded [0,1] histories. Conservative
    # finite-sample bound; do not replace rare-tail uncertainty by a Gaussian Z.
    n=len(x);mean=float(np.mean(x));var=float(np.var(x,ddof=1));log=np.log(2/delta)
    radius=float(np.sqrt(2*var*log/n)+7*log/(3*(n-1)))
    return dict(mean=mean,radius=radius,nonzero=int(np.count_nonzero(x)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--samples',type=int,required=True);a=p.parse_args()
    cases=json.loads((a.root/'cases.json').read_text());n=a.samples;nc=len(cases)
    data={arm:np.memmap(a.root/f'packets.{arm}.bin',DT,mode='r').reshape(nc,n) for arm in ['reference','candidate']}
    rows=[];ks_fail=[];side_fail=[];budget_fail=[];budget_unresolved=[]
    # Family includes 2 arms * 2 ROIs * all cases. Domain includes escaped
    # packets with zero dose; no normalization to deposited histories.
    delta=.01/(4*nc)
    for i,case in enumerate(cases):
        row=dict(case=i,**case,arms={},ks={},dose={});coordinates={};scores={}
        for arm,d in data.items():
            r=d[i];assert np.all(r['status']<3) and not r['guard'].any() and np.isfinite(r['position']).all()
            deposited=r['status']<2;offset=r['position'].astype(float)-np.asarray(case['position']);coordinates[arm]=offset[deposited]
            scores[arm]={name:slit_score(offset[:,0],name=='valley')*deposited for name in ['peak','valley']}
            row['arms'][arm]=dict(counts=[int(np.count_nonzero(r['status']==k)) for k in range(3)],steps=int(r['steps'].sum()),batched=int(r['batched'].sum()),elastic=int(r['elastic'].sum()),photons=int(r['photons'].sum()),radial2_mean=float(np.mean(np.sum(offset[deposited,:2]**2,axis=1))),z_mean=float(np.mean(offset[deposited,2])))
        for axis in range(3):
            result=ks_2samp(coordinates['reference'][:,axis],coordinates['candidate'][:,axis])
            row['ks'][str(axis)]=dict(statistic=float(result.statistic),pvalue=float(result.pvalue))
            if result.pvalue<.01/(3*nc):ks_fail.append([i,axis])
        for status in range(3):
            x=row['arms']['reference']['counts'][status];y=row['arms']['candidate']['counts'][status]
            pv=float(fisher_exact([[x,n-x],[y,n-y]]).pvalue)
            if pv<.01/(3*nc):side_fail.append([i,status,pv])
        for name in ['peak','valley']:
            x=bounded_mean(scores['reference'][name],delta);y=bounded_mean(scores['candidate'][name],delta)
            diff=y['mean']-x['mean'];radius=x['radius']+y['radius']
            lower=max(0,x['mean']-x['radius']);upper=min(1,x['mean']+x['radius'])
            certified=bool(lower>0 and abs(diff)+radius<=.01*lower)
            rejected=bool(upper>0 and max(0,abs(diff)-radius)>.01*upper)
            if rejected:budget_fail.append([i,name])
            elif not certified:budget_unresolved.append([i,name])
            row['dose'][name]=dict(reference=x,candidate=y,change_percent=100*diff/x['mean'] if x['mean'] else None,
                absolute_difference=diff,difference_radius=radius,one_percent_budget_certified=certified,one_percent_budget_rejected=rejected)
        # Descriptive local lateral histogram (0.02mm). Small-count bins are
        # reported separately through CDFs, never declared 1% accurate.
        reference=coordinates['reference'][:,0];candidate=coordinates['candidate'][:,0]
        edge=max(.1,min(20,float(np.quantile(abs(reference),.999))))
        bins=np.arange(-np.ceil(edge/.02)*.02,np.ceil(edge/.02)*.02+.021,.02)
        x=np.histogram(reference,bins)[0];y=np.histogram(candidate,bins)[0];mask=x>=1000
        row['lateral_histogram']=dict(bin_width_mm=.02,reference_count_threshold=1000,qualified_bins=int(mask.sum()),max_local_change_percent=float(np.max(100*abs(y[mask]-x[mask])/x[mask])) if mask.any() else None)
        rows.append(row)
    totals={arm:{k:sum(r['arms'][arm][k] for r in rows) for k in ['steps','batched','elastic','photons']} for arm in data}
    report=dict(scope='Full electron/photon trajectories for the configured material/density cases, protected 0.005-loss elastic batching with explicit angles. No joint response or CT acceptance; no performance measurement.',density_override_cases=sum('density' in c for c in cases),cases=nc,samples_per_case_per_arm=n,total_per_arm=n*nc,family_alpha=.01,KS_failures=ks_fail,terminal_side_failures=side_fail,one_percent_budget_rejections=budget_fail,one_percent_budget_unresolved=budget_unresolved,work=totals,results=rows)
    (a.root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    # Distribution differences and budget uncertainty remain explicit; the
    # caller must review both instead of equating normal exit with acceptance.


if __name__=='__main__':main()
