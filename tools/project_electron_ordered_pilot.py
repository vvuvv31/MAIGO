"""Paired OFFLINE path-vs-chord projection, never patient acceptance.

Both variants use the same sampled full-family deposits, energy and source.
Source is baseline 3-D lateral integral at a pencil axis, bin-uniform births;
mean primary energy from cumulative loss (energy spread/MCS not reproduced).
This approximation is exposed: compare mechanisms, not a full MC replacement.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_air_tissue_interface_pilot import analyze,gpu
from analyze_longitudinal_holdout import sha

def move(points,v,boundary,rhos):
    side=points[:,2]>=boundary;rho=np.where(side,rhos[1],rhos[0]);end=points+10*v/rho[:,None]
    cross=(end[:,2]>=boundary)!=side
    if np.any(cross):
        t=(boundary-points[cross,2])*rho[cross]/(10*v[cross,2])
        crossing=points[cross]+t[:,None]*10*v[cross]/rho[cross,None]
        other=np.where(side[cross],rhos[0],rhos[1])
        end[cross]=crossing+(1-t[:,None])*10*v[cross]/other[:,None]
    return end

def pack(response):
    meta=json.loads((response/'manifest.json').read_text());records={};pooled={}
    for name,info in meta['sources'].items():
        p=response/name
        if sha(p)!=info['payload_sha256']:raise ValueError('Path payload SHA mismatch')
        # NPZ __getitem__ rereads an array each time. Retaining slices of each
        # reread silently retains N full copies. Materialize each member ONCE.
        with np.load(p) as archive:a={key:archive[key] for key in archive.files}
        hu=info['hu'];records.setdefault(hu,[]).append(a)
    for hu,items in records.items():
        loss=sum(a['loss'] for a in items);escape=sum(a['escape'] for a in items)
        for b in range(37):
            if loss[b]<=0:continue
            paths=[];weights=[]
            for a in items:
                for j in np.flatnonzero(a['bins']==b):
                    paths.append(a['vectors'][a['offsets'][j]:a['offsets'][j+1]])
                    weights.append(a['weights'][j]/loss[b])
            maximum=max([len(p) for p in paths],default=0)
            padded=np.zeros((len(paths),maximum,3))
            for j,p in enumerate(paths):padded[j,:len(p)]=p
            weights=np.array(weights)
            if weights.sum()>1+1e-7:raise ValueError('Nonlocal fraction >1')
            pooled[hu,b]=(padded,weights,1-weights.sum())
    return pooled

def project(root,response,birth_points=4):
    reference=analyze(root);manifest=json.loads((root/'manifest.json').read_text());kernels=pack(response);out={}
    radial2=((np.arange(100)+.5)*2-100)**2
    for name,spec in manifest['cases'].items():
        rho=np.concatenate([np.full(s['length_mm']*2,manifest['materials'][str(s['hu'])]['rho']) for s in spec['segments']])
        hu=np.concatenate([np.full(s['length_mm']*2,s['hu']) for s in spec['segments']]);nz=len(rho);length=nz*.5
        factor=rho*2e-6/1.602176634e-13
        energy=gpu(root/name/'gpu/dose.mhd',nz)/manifest['histories']*factor
        query=175-(np.cumsum(energy)-energy/2)/12
        boundary=spec['interface_mm'];rhos=(rho[0],rho[-1]);variants={}
        for mode in ('collapsed','ordered'):
            deposited=np.zeros(nz);moment=np.zeros(nz);outer=np.zeros(nz);escaped=0.
            for i,loss in enumerate(energy):
                if loss<=0:continue
                key=(hu[i],int(query[i]/5))
                if key not in kernels:raise ValueError(f'No conditional response {key}')
                paths,weights,local=kernels[key]
                deposited[i]+=loss*local;moment[i]+=2*loss*local
                if not len(weights):continue
                for birth in range(birth_points):
                    position=np.zeros((len(weights),3));position[:,2]=(i+(birth+.5)/birth_points)*.5
                    active=np.ones(len(weights),dtype=bool)
                    steps=paths if mode=='ordered' else paths.sum(axis=1)[:,None,:]
                    for step in range(steps.shape[1]):
                        ids=np.flatnonzero(active)
                        if not len(ids):break
                        position[ids]=move(position[ids],steps[ids,step],boundary,rhos)
                        active[ids]=np.all(abs(position[ids,:2])<100,axis=1)&(position[ids,2]>=0)&(position[ids,2]<length)
                    w=weights*loss/birth_points;escaped+=w[~active].sum()
                    xyz=np.floor((position[active]+np.array([100,100,0]))/np.array([2,2,.5])).astype(int)
                    z=xyz[:,2];r2=radial2[xyz[:,0]]+radial2[xyz[:,1]]
                    deposited+=np.bincount(z,weights=w[active],minlength=nz)
                    moment+=np.bincount(z,weights=w[active]*r2,minlength=nz)
                    outer+=np.bincount(z,weights=w[active]*(r2>100),minlength=nz)
            closure=(deposited.sum()+escaped)/energy.sum()-1
            if abs(closure)>1e-10:raise ValueError('Projection energy closure')
            z=(np.arange(nz)+.5)*.5-boundary;windows=[]
            for w in reference['cases'][name]['windows']:
                low,high=w['relative_depth_mm'];take=(z>=low)&(z<high)
                windows.append(dict(relative_depth_mm=[low,high],
                    dose_relative_error=float((deposited[take]/factor[take]).sum()/w['topas_Gy_per_primary']-1),
                    radial_rms_mm=float(np.sqrt((moment[take]/factor[take]).sum()/(deposited[take]/factor[take]).sum())),
                    outer_10mm_fraction=float((outer[take]/factor[take]).sum()/(deposited[take]/factor[take]).sum())))
            variants[mode]=dict(windows=windows,closure=float(closure),escaped_MeV_per_primary=float(escaped))
        out[name]=variants
    return dict(status='OFFLINE_MECHANISM_COMPARISON_ONLY',cases=out,
        limitations=['mean-energy pencil source approximation, not full 1% spread/source-MCS',
                     'bulk source-material electron scattering retained after crossing',
                     'finite-box photon escape retained local','no GPU or patient validation'],
        response_manifest_sha256=sha(response/'manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('response',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=project(a.root,a.response)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    for case,variants in report['cases'].items():
        for mode,v in variants.items():print(case,mode,v['windows'],flush=True)
