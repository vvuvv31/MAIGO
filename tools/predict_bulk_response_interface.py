"""Offline 1D mass-coordinate projection of independently measured JOINT response.

Derived from 3D scorers, never a 1D reference scorer. This is NOT GPU validation:
curved paths, source-energy conditioning, photon transport remain approximations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_air_tissue_interface_pilot import analyze,gpu
from analyze_longitudinal_holdout import sha

def kernel(response,medium,seed=None):
    report=json.loads((response/'report.json').read_text())['cases']
    def matches(k):
        if seed is None:return True
        if seed in ('even','odd'):return int(k.split('/')[-1])%2==(seed=='odd')
        return k.endswith('/'+seed)
    chosen=[(k,v) for k,v in report.items() if k.startswith(medium+'/') and matches(k)]
    if not chosen:raise ValueError('No response samples in requested group')
    total=sum(v['loss_MeV'] for k,v in chosen)
    local=sum(v['local_MeV'] for k,v in chosen)/total
    escape=sum(v['escaped_MeV'] for k,v in chosen)/total
    offsets=[0.];weights=[local+escape]
    for k,v in chosen:
        path=response/(k.replace('/','_')+'.npz')
        if sha(path)!=v.get('response_sha256'):raise ValueError('Missing/mismatched response SHA')
        data=np.load(path)
        if (data['centroid_mass_g_cm2'].shape!=(len(data['energy_MeV']),2) or
            not np.isfinite(data['centroid_mass_g_cm2']).all() or
            not np.isfinite(data['energy_MeV']).all() or np.any(data['energy_MeV']<0)):
            raise ValueError('Invalid correlated response')
        offsets.extend(data['centroid_mass_g_cm2'][:,0]);weights.extend(data['energy_MeV']/total)
    if abs(sum(weights)-1)>1e-5:raise ValueError('Kernel normalization mismatch')
    return np.array(offsets),np.array(weights),escape

def predict(edges,energy,materials,kernels):
    result=np.zeros(len(energy));escaped=0.
    for i,edep in enumerate(energy):
        offset,weight,_=kernels[materials[i]]
        lo=edges[i]+offset;hi=edges[i+1]+offset
        overlap=np.maximum(0.,np.minimum(edges[1:,None],hi)-np.maximum(edges[:-1,None],lo))
        shares=overlap/(edges[i+1]-edges[i])
        fractions=shares@weight
        result+=edep*fractions;escaped+=edep*(1-fractions.sum())
    if abs((result.sum()+escaped)/energy.sum()-1)>1e-10:raise ValueError('Projection closure')
    return result,escaped

def analyze_projection(reference,response):
    m=json.loads((reference/'manifest.json').read_text());ref=analyze(reference)
    if ref['status']!='BASELINE_INTERFACE_DIAGNOSIS_ONLY':raise ValueError('Unstable reference')
    outputs={}
    for seed in [None,'odd','even']:
        kernels={k:kernel(response,k,seed) for k in ['air','tissue']}
        for name,c in m['cases'].items():
            rho=np.concatenate([np.full(s['length_mm']*2,m['materials'][str(s['hu'])]['rho']) for s in c['segments']])
            materials=np.concatenate([np.full(s['length_mm']*2,'air' if s['hu']==-1000 else 'tissue') for s in c['segments']])
            baseline=gpu(reference/name/'gpu/dose.mhd',len(rho))/m['histories']
            factor=rho*2e-6/1.602176634e-13
            edges=np.r_[0,np.cumsum(rho*.5/10)]
            energy,escaped=predict(edges,baseline*factor,materials,kernels)
            curve=energy/factor;z=(np.arange(len(rho))+.5)*.5-c['interface_mm']
            windows=[]
            for w in ref['cases'][name]['windows']:
                lo,hi=w['relative_depth_mm'];sel=(z>=lo)&(z<hi)
                windows.append(dict(relative_depth_mm=[lo,hi],error=float(curve[sel].sum()/w['topas_Gy_per_primary']-1)))
            outputs[name+'/'+str(seed)]=dict(windows=windows,
                max_abs_bin_error=float(np.max(abs(curve[(z>=-5)&(z<5)]/ref['cases'][name]['interface_profiles']['topas_Gy_per_primary']-1))),
                escape_MeV_per_primary=escaped,bulk_escape_retained_fraction={k:v[2] for k,v in kernels.items()})
    return dict(status='OFFLINE_HYPOTHESIS_ONLY',cases=outputs,
                limitations=['170-175 MeV/u pooled parent window, not exact per-step energy matching',
                'net mass displacement, not curved electron paths across changed composition',
                'bulk photon escape explicitly retained at source; not neutral transport',
                'no new GPU dose or patient gamma'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('response',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=analyze_projection(a.reference,a.response)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    for k,v in r['cases'].items():print(k,[round(100*w['error'],3) for w in v['windows']],flush=True)
