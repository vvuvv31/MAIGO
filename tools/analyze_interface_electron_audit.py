"""Read-only cross-interface electron ancestry/energy census; not a LUT fit."""
import argparse,json,re
from pathlib import Path
import numpy as np
from analyze_air_tissue_interface_pilot import topas
from analyze_longitudinal_holdout import sha
from compare_electron_response_geometries import load_v2_binary,build_roots

def census(folder,name):
    d=load_v2_binary(folder/'steps.phsp')
    if len(np.unique(d[:,[0,1,2,5]],axis=0))!=len(d):raise ValueError('Duplicate track steps')
    if set(d[:,1].astype(int))!=set(range(4)):raise ValueError('Incomplete histories')
    tracks,root=build_roots(d)
    first={}
    for i,row in enumerate(d):
        first.setdefault(tuple(row[:3].astype(int)),i)
    families={k for k in tracks if root(k)==k}
    for k in families:
        row=d[first[k]]
        if row[22]!=1 or row[29]<1:raise ValueError('Missing generating parent step')
    birth_sum={'air':0.,'tissue':0.};nroots={'air':0,'tissue':0}
    def material(rho):
        if abs(rho/.01131606474518776-1)<1e-4:return 'air'
        if abs(rho/1.0787997245788574-1)<1e-4:return 'tissue'
        return 'vacuum'
    for k in families:
        row=d[first[k]];m=material(row[28]);birth_sum[m]+=row[9]*row[19];nroots[m]+=1
    interface=40 if name=='air_to_tissue' else 5
    windows=[(-5,-2),(-2,0),(0,2),(2,5)]
    scores=[dict(primary=0.,electron_born_air=0.,electron_born_tissue=0.) for _ in windows]
    all_dep={'air':0.,'tissue':0.};primary_dep={'air':0.,'tissue':0.}
    transfer={'air_to_tissue':0.,'tissue_to_air':0.};vacuum_deposit=0.
    for row in d:
        target=material(row[20]);edep=row[16]*row[19]
        if target=='vacuum':
            vacuum_deposit+=edep
            continue
        all_dep[target]+=edep
        rk=root(tuple(row[:3].astype(int)))
        if rk is None:
            category='primary';primary_dep[target]+=edep
        else:
            born=material(d[first[rk],28]);category='electron_born_'+born
            if born!=target:transfer[born+'_to_'+target]+=edep
        # Geant4 material boundaries are assigned by pre-step material. Use
        # midpoint only for this census window (not a replacement dose scorer).
        z=(row[12]+row[15])/2-interface
        for (lo,hi),score in zip(windows,scores):
            if lo<=z<hi:score[category]+=edep
    dims=(80,50) if name=='air_to_tissue' else (10,120)
    mats=('air','tissue') if name=='air_to_tissue' else ('tissue','air')
    rho={'air':.01131606474518776,'tissue':1.0787997245788574}
    dose_energy={m:float(topas(folder/f'dose{i}.csv',nz).sum()*rho[m]*2e-6/1.602176634e-13)
                 for i,(m,nz) in enumerate(zip(mats,dims))}
    residual={m:dose_energy[m]/all_dep[m]-1 for m in mats}
    if max(abs(x) for x in residual.values())>1e-4:raise ValueError(f'3D/ntuple energy mismatch: {residual}')
    if vacuum_deposit>1e-12:raise ValueError('Non-negligible vacuum deposition')
    return dict(roots=nroots,vacuum_deposit_MeV=vacuum_deposit,direct_electron_birth_MeV=birth_sum,primary_local_deposit_MeV=primary_dep,
        cross_material_electron_deposit_MeV=transfer,
        windows=[dict(relative_depth_mm=w,energy_MeV=s) for w,s in zip(windows,scores)],
        dose_ntuple_relative_residual=residual,
        artifacts={str(p):sha(p) for p in [folder/'steps.phsp',folder/'steps.header',folder/'run.txt']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text())
    for path,pin in manifest['inputs'].items():
        if sha(Path(path))!=pin:raise ValueError('Input changed: '+path)
    r={name:{seed:census(a.root/name/seed,name) for seed in ('s1','s2')} for name in ('air_to_tissue','tissue_to_air')}
    text=json.dumps(dict(scope='4 histories/seed, diagnosis only, not package compilation',cases=r),indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(text)
    else:print(text)
