#!/usr/bin/env python3
"""Compare26 composition component candidates to explicitly selected83-node sources.

Finite-bin centroid approximation and continuous-density approximation are
reported separately. This is not a GPU gamma/throughput acceptance harness.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from build_electron_response26 import birth_nodes,collision_nodes,raw_rate


def weights(degree,x,domain):
    if not degree:return np.ones(1)
    x=float(np.clip(x,0,1))
    low,high=domain;rho=low+x*(high-low)
    if degree==1:return np.array([1-x,x])
    edges=np.exp(np.linspace(np.log(low),np.log(high),degree+1));edges[0]=low;edges[-1]=high
    i=min(int(np.searchsorted(edges,rho,side='right')-1),degree-1);i=max(0,i)
    f=(rho-edges[i])/(edges[i+1]-edges[i]);w=np.zeros(degree+1);w[i]=1-f;w[i+1]=f
    return w


def masses(cdf):
    return np.diff(np.column_stack([np.zeros(len(cdf),np.uint64),cdf]),axis=1).astype(float)/float(np.iinfo(np.uint64).max)


def block_fields(b,birth,energy_bins,angle_bins):
    if birth:
        w=np.diff(np.r_[0.,b['cdf']]);e=b['energy'];pdg=np.full(len(b),11)
    else:
        w=np.diff(np.r_[np.uint64(0),b['cdf']]).astype(float)/float(np.iinfo(np.uint64).max)
        e=b['fraction'];pdg=b['pdg']
    mu=b['cosine'];low,high=(1e-7,.8) if birth else (1e-12,1.)
    ei=np.clip(((np.log(e)-math.log(low))/math.log(high/low)*energy_bins).astype(int),0,energy_bins-1)
    ai=np.clip(((mu+1)*.5*angle_bins).astype(int),0,angle_bins-1)
    code=np.where(pdg==0,0,np.where(pdg==11,1,2))
    return w,e,mu,(code*energy_bins+ei)*angle_bins+ai


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('data/archive/electron_density_nodes_20261006'))
    p.add_argument('--components',nargs='+',choices=['proton','carbon','charged','photon'],default=['proton','carbon','charged','photon'])
    p.add_argument('--sections',type=int,nargs='+',default=list(range(26)))
    p.add_argument('--channel-stride',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.channel_stride<1:p.error('Invalid stride')
    manifest=json.loads((args.candidate/'manifest.json').read_text())
    if not manifest['density_representation'].startswith('continuous piecewise affine'):
        raise ValueError('Retired density basis; use its frozen validation result, not this checker')
    eb,ab=manifest.get('energy_bins',128),manifest.get('angle_bins',256)
    reports=[]
    for component in args.components:
        birth=component in ('proton','carbon')
        source=birth_nodes(args.source/component/'birth_bank.bin') if birth else collision_nodes(args.source/'shared'/component,component)
        for section in args.sections:
            nodes=sorted([n for n in source if n['section']==section],key=lambda n:n['density'])
            model=np.load(args.candidate/component/f'composition{section:02d}.npz',allow_pickle=False)
            degree=int(model['density_degree'][0]);rlo,rhi=model['density_domain'].astype(float)
            metrics=dict(yield_absolute=0.,yield_relative_active=0.,joint_cell_total_variation=0.,
                         mean_energy_relative=0.,mean_cosine_absolute=0.,rate_peak_normalized_error=0.,
                         rate_relative_active=0.,majorant_ratio=0.)
            checked=0;missing=0;zero_rate_probes=0;branches=0
            rate_data={}
            if component=='charged':
                for process in range(2):
                    edges=model[f'rate{process}_edges'].astype(float)
                    values=model[f'rate{process}_values'].astype(float)
                    pieces=np.concatenate([np.broadcast_to(edges[:-1],values.shape[:2])[...,None],
                                           np.broadcast_to(edges[1:],values.shape[:2])[...,None],values],axis=-1)
                    probe=edges[:-1,None]+np.diff(edges)[:,None]*np.array([.01,.125,.375,.5,.625,.875,.99])
                    control=np.array([np.maximum(0.,raw_rate(rows,probe)) for rows in pieces])
                    majorant=float(model[f'rate{process}_majorant'][0])
                    metrics['majorant_ratio']=max(metrics['majorant_ratio'],float(control.max()/majorant))
                    peak=max(max(float(np.max(np.maximum(0.,raw_rate(n['rates'][process],probe))/n['density'])) for n in nodes),1e-30)
                    for n in nodes:
                        x=0 if rlo==rhi else (n['density']-rlo)/(rhi-rlo);bw=weights(degree,x,(rlo,rhi))
                        actual=np.tensordot(bw,control,axes=(0,0))
                        expected=np.maximum(0.,raw_rate(n['rates'][process],probe))/n['density']
                        error=abs(actual-expected)
                        metrics['rate_peak_normalized_error']=max(metrics['rate_peak_normalized_error'],float(error.max()/peak))
                        mask=expected>1e-5*peak
                        if mask.any():metrics['rate_relative_active']=max(metrics['rate_relative_active'],float(np.max(error[mask]/expected[mask])))
                    rate_data[process]=pieces
            for channel in range(0,len(nodes[0]['channels']),args.channel_stride):
                prefix=f'ch{channel:04d}_'
                if prefix+'key' not in model:
                    if any(n['channels'][channel]['count'] for n in nodes):missing+=1
                    continue
                key=model[prefix+'key'];cdf=model[prefix+'cdf'];pmf=masses(cdf)
                ce=model[prefix+'energy'];cm=model[prefix+'cosine'];total=model[prefix+'total'].astype(float)
                if not np.isfinite(ce).all() or not np.isfinite(cm).all() or np.any(ce<=0) or np.any(abs(cm)>1):
                    raise ValueError('Invalid joint state')
                if np.any(cdf[:,1:]<cdf[:,:-1]) or np.any(total<0):raise ValueError('Invalid probability controls')
                if np.any((total>0)&(cdf[:,-1]!=np.iinfo(np.uint64).max)):raise ValueError('Incomplete CDF')
                for n in nodes:
                    ch=n['channels'][channel]
                    if not ch['count']:continue
                    block=n['states'][int(ch['offset']):int(ch['offset']+ch['count'])]
                    w,e,mu,code=block_fields(block,birth,eb,ab)
                    reference=np.bincount(np.searchsorted(key,code),weights=w,minlength=len(key))
                    x=0 if rlo==rhi else (n['density']-rlo)/(rhi-rlo);bw=weights(degree,x,(rlo,rhi))
                    if birth:
                        unnormalized=bw*total;fraction=float(unnormalized.sum())
                        expected=float(ch['fraction'])
                        metrics['yield_absolute']=max(metrics['yield_absolute'],abs(fraction-expected))
                        if expected>1e-5:metrics['yield_relative_active']=max(metrics['yield_relative_active'],abs(fraction/expected-1))
                    elif component=='charged':
                        bin=channel//2;energy=np.sqrt(n['edges'][bin]*n['edges'][bin+1])
                        control=np.array([max(0.,float(raw_rate(rows,energy))) for rows in rate_data[channel%2]])
                        if max(0.,float(raw_rate(n['rates'][channel%2],energy)))==0:
                            zero_rate_probes+=1;continue
                        unnormalized=bw*control
                    else:unnormalized=bw
                    if unnormalized.sum()<=0:missing+=1;continue
                    weighted=unnormalized[:,None]*pmf/unnormalized.sum()
                    probability=weighted.sum(axis=0)
                    metrics['joint_cell_total_variation']=max(metrics['joint_cell_total_variation'],float(abs(probability-reference).sum()/2))
                    expected_e=float(w@e);actual_e=float(np.sum(weighted*ce))
                    if expected_e>0:metrics['mean_energy_relative']=max(metrics['mean_energy_relative'],abs(actual_e/expected_e-1))
                    metrics['mean_cosine_absolute']=max(metrics['mean_cosine_absolute'],abs(float(np.sum(weighted*cm))-float(w@mu)))
                    # Count distinct branch support; joint positive-cell mass
                    # errors are reported rather than hidden by moments.
                    for pdg in [0,11,22]:
                        native_mass=float(w[np.array([0,11,22])[(code//(eb*ab)).astype(int)]==pdg].sum())
                        new_mass=float(probability[model[prefix+'pdg']==pdg].sum())
                        if native_mass>0 and new_mass<=0:branches+=1
                    checked+=1
            row=dict(component=component,section=section,checked=checked,missing_support=missing,
                     missing_branches=branches,zero_midpoint_rate_probes=zero_rate_probes,metrics=metrics)
            reports.append(row);print(json.dumps(row),flush=True);model.close()
    summary={}
    for component in args.components:
        group=[r for r in reports if r['component']==component]
        summary[component]={k:max(r['metrics'][k] for r in group) for k in group[0]['metrics']}
        summary[component].update(checked=sum(r['checked'] for r in group),missing_support=sum(r['missing_support'] for r in group),
                                  missing_branches=sum(r['missing_branches'] for r in group))
    result=dict(status='COMPONENT_COMPARISON_NOT_GPU_OR_DOSE_ACCEPTANCE',channel_stride=args.channel_stride,
                density_points='original native node densities; hard rates additionally use7 points per energy segment',
                summaries=summary,rows=reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
