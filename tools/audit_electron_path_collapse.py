"""Read-only geometric counterexample using direct-electron TOPAS trajectories.

Not a dose model or fitted correction. Compare a density-weighted collapsed
vector with the actual piecewise track. Uniform density must be an identity.
Descendants excluded explicitly: this diagnostic does not claim family closure.
"""
import argparse,json
from pathlib import Path
import numpy as np
from compare_electron_response_geometries import load_v2_binary,build_roots
from analyze_longitudinal_holdout import sha

def endpoint(birth,vector,boundary,rhos):
    # vector units are rho*mm, not geometric displacement. A straight ray can
    # cross a single planar interface at most once; exact analytic integration.
    side=int(birth[2]>=boundary)
    result=birth+vector/rhos[side]
    if vector[2]==0 or int(result[2]>=boundary)==side:return result
    t=(boundary-birth[2])*rhos[side]/vector[2]
    if t<0 or t>1:raise ValueError('Invalid planar crossing')
    crossing=birth+t*vector/rhos[side]
    return crossing+(1-t)*vector/rhos[1-side]

def audit(path,boundary,rhos):
    d=load_v2_binary(path);tracks,root=build_roots(d)
    selected={k for k in tracks if root(k)==k}
    cumulative={};previous={};energy=0.;square=0.;maximum=0.;uniform_max=0.;crossed=0.;count=0
    for row in d:
        k=tuple(row[:3].astype(int))
        if k not in selected:continue
        birth=row[6:9];pre=row[10:13];post=row[13:16];rho=row[20]
        if k in previous and np.linalg.norm(previous[k]-pre)>1e-5:raise ValueError('Discontinuous electron track')
        before=cumulative.get(k,np.zeros(3));delta=post-pre
        # Midpoint for continuous charged-particle step deposition. Apply the
        # SAME midpoint for actual and counterfactual, so no interpolation bias.
        if row[16]>0:
            actual=(pre+post)/2
            collapsed=endpoint(birth,before+.5*rho*delta,boundary,rhos)
            error=float(np.linalg.norm(collapsed-actual));weight=row[16]*row[19]
            energy+=weight;square+=weight*error**2;maximum=max(maximum,error);count+=1
            if int(collapsed[2]>=boundary)!=int(actual[2]>=boundary):crossed+=weight
            uniform=endpoint(birth,rho*(actual-birth),boundary,(rho,rho))
            uniform_max=max(uniform_max,float(np.linalg.norm(uniform-actual)))
        cumulative[k]=before+rho*delta;previous[k]=post
    if energy<=0 or uniform_max>1e-9:raise ValueError('Uniform geometry identity failed')
    return dict(direct_electron_deposit_MeV=energy,deposit_steps=count,
        energy_weighted_collapse_position_rms_mm=float(np.sqrt(square/energy)),
        max_collapse_position_error_mm=maximum,wrong_material_deposit_fraction=crossed/energy,
        uniform_identity_max_mm=uniform_max,raw_sha256=sha(path),
        scope='direct root electrons only; geometry diagnostic, not predicted dose')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();air=.01131606474518776;tissue=1.0787997245788574;report={}
    for name,boundary,rhos in [('air_to_tissue',40,(air,tissue)),('tissue_to_air',5,(tissue,air))]:
        for seed in ('s1','s2'):
            report[name+'/'+seed]=audit(a.root/name/seed/'steps.phsp',boundary,rhos)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
