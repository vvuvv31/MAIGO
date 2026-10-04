#!/usr/bin/env python3
"""Candidate weighted joint compression; preserve branch graphs and tail anchors.

This is not the rejected ordinary uniform reservoir: strata condition jointly
on deflection, energy loss, displacement length and terminal/continuing state.
Every occupied stratum retains its probability mass. Native child-producing
rows and the maximum-angle/maximum-displacement records remain exact.
"""
import argparse
from pathlib import Path
import numpy as np
from joint_family import load_model
from prototype import sha,save


def quantiles(x,w,bins):
    order=np.argsort(x,kind='stable');cdf=np.cumsum(w[order]);cdf/=cdf[-1]
    return x[order[np.minimum(np.searchsorted(cdf,np.linspace(0,1,bins+1)),len(x)-1)]]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=100308101)
    p.add_argument('--draws-per-cell',type=int,default=2);a=p.parse_args()
    assert a.draws_per_cell>0
    models,report=load_model(a.model);m=models['continuation'];rng=np.random.default_rng(a.seed)
    selected=[];weights=[];sizes=[];errors=[];protected_total=0
    for lo,hi in zip(m['offsets'][:-1],m['offsets'][1:]):
        ids=np.arange(lo,hi);mass=np.diff(np.r_[0.,m['cdf'][lo:hi]])
        if not len(ids):sizes.append(0);continue
        assert np.all(mass>0)
        cosine=m['postdir'][ids,2];loss=1-m['fraction'][ids,1]
        length=np.linalg.norm(m['delta'][ids],axis=1)
        lateral=np.linalg.norm(m['delta'][ids,:2],axis=1)
        terminal=m['fraction'][ids,0]>=1-1e-7
        protected=np.any(m['fraction'][ids,2:]>0,axis=1)
        protected[[np.argmin(cosine),np.argmax(length),np.argmax(lateral)]]=True
        keep=[ids[protected]];prob=[mass[protected]];protected_total+=int(protected.sum())
        for state in [False,True]:
            use=np.flatnonzero((~protected)&(terminal==state))
            if not len(use):continue
            cell=np.zeros(len(use),int)
            for feature,bins in [(cosine,8),(loss,4),(length,4)]:
                edges=quantiles(feature[use],mass[use],bins)
                key=np.clip(np.searchsorted(edges,feature[use],side='right')-1,0,bins-1)
                cell=cell*bins+key
            for c in np.unique(cell):
                members=use[cell==c];w=mass[members]
                if len(members)<=a.draws_per_cell:
                    keep.append(ids[members]);prob.append(w)
                else:
                    draw=rng.choice(members,size=a.draws_per_cell,replace=True,p=w/w.sum())
                    keep.append(ids[draw]);prob.append(np.full(len(draw),w.sum()/len(draw)))
        keep=np.concatenate(keep);prob=np.concatenate(prob)
        error=abs(prob.sum()-mass.sum());assert error<1e-12
        selected.append(keep);weights.append(prob);sizes.append(len(keep));errors.append(error)
    selected=np.concatenate(selected);weights=np.concatenate(weights)
    offsets=np.r_[0,np.cumsum(sizes)];cdf=np.empty(len(selected))
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        if hi>lo:cdf[lo:hi]=np.cumsum(weights[lo:hi])/weights[lo:hi].sum();cdf[hi-1]=1.
    old_count=len(m['cdf'])
    for key in list(m):
        if key not in ['edges','offsets','nearest']:m[key]=m[key][selected]
    m['offsets']=offsets;m['cdf']=cdf
    a.out.mkdir(parents=True,exist_ok=False);hashes={}
    for name,model in models.items():
        path=a.out/(name+'.npz');np.savez(path,**model);hashes[name]=sha(path)
    save(a.out/'report.json',dict(report,status='JOINT_STRATA_REDUCTION_NOT_CT_VALIDATED',hashes=hashes,
        reduction=dict(source_report_sha256=sha(a.model/'report.json'),original_rows=old_count,retained_rows=len(selected),
                       protected_rows=protected_total,draws_per_cell=a.draws_per_cell,seed=a.seed,maximum_probability_mass_error=max(errors)),
        payload_bytes=sum(v.nbytes for model in models.values() for v in model.values()),script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
