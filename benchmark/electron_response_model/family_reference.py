#!/usr/bin/env python3
"""Streaming native full electron/photon-family reference, no transport changes.

One 672-byte row per primary replaces the diagnostic raw step output. The
charged and photon-descendant profiles remain separately auditable. This
runner is deliberately restricted to the existing homogeneous lung fixture.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from prototype import save, sha
from voxel_profile import summarize, CENTRES

LEGACY_DTYPE = np.dtype([('run','<i4'),('event','<i4'),('roots','<i4'),('steps','<i4'),('tracks','<i4'),
                  ('incident','<f8'),('charged','<f8'),('sink','<f8'),('outside','<f8'),
                  ('max_residual','<f8'),('max_step','<f8'),('photon','<f8'),('escape','<f8'),
                  ('profile','<f8',(36,)),('photon_profile','<f8',(36,))])
DTYPE=np.dtype(LEGACY_DTYPE.descr+[('navigation','<i4'),('navigation_path','<f8')])


def read_summary(path, histories):
    assert DTYPE.itemsize == 672 and LEGACY_DTYPE.itemsize==660
    dtype=LEGACY_DTYPE if path.stat().st_size==histories*LEGACY_DTYPE.itemsize else DTYPE
    assert path.stat().st_size==histories*dtype.itemsize
    r = np.fromfile(path, dtype); r = r[np.argsort(r['event'])]
    assert np.array_equal(r['event'],np.arange(histories)) and len(np.unique(r['run']))==1
    assert np.all(r['roots']==1) and np.all(r['steps']>0) and np.all(r['tracks']>0)
    assert np.all(r['incident']==r['incident'][0])
    for name in ['profile','photon_profile','charged','photon','sink','escape']:
        assert np.isfinite(r[name]).all() and np.all(r[name]>=0)
    assert np.max(abs(r['incident']-r['charged']-r['sink']))<1e-8
    assert np.max(abs(r['sink']-r['photon']-r['escape']))<1e-8
    assert np.max(abs(r['profile'].sum(axis=1)-r['charged']))<1e-10
    assert np.max(abs(r['photon_profile'].sum(axis=1)-r['photon']))<1e-10
    assert np.max(r['max_residual'])<1e-8 and np.max(r['outside'])<1e-12
    assert np.max(r['max_step'])<=.25*(1+1e-6)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,default=Path('out/electron_voxel_lung065_smoke_v2_20261002'))
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--histories',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1002320001)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--raw',action='store_true')
    p.add_argument('--homogeneous-box',action='store_true',help='Matched source/material/box without internal voxel faces')
    p.add_argument('--max-step',type=float,default=.25,help='Native physical control, mm; does not change the official CT reference')
    p.add_argument('--energy',type=float,help='Independent mono-electron energy in MeV; never retunes the reference physics')
    a=p.parse_args(); a.out=a.out.resolve(); a.out.mkdir(parents=True,exist_ok=False)
    a.binary=a.binary.resolve(); inputs=json.loads((a.template/'manifest.json').read_text())
    assert len(inputs)==2 and a.histories>0 and a.threads>0
    assert 0<a.max_step<=.25
    assert a.energy is None or 0<a.energy<=.65
    assert all(s['table']=='s1_mid9' and s['geometry']=='voxel_box_in_vacuum' for s in inputs)
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
             LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    results=[]; cases=[]
    for i,metadata in enumerate(inputs):
        d=a.out/metadata['name']; d.mkdir()
        source=(a.template/metadata['name']/'topas.txt').read_text()
        source=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {a.seed+i}',source)
        source=re.sub(r'i:Ts/NumberOfThreads = \d+',f'i:Ts/NumberOfThreads = {a.threads}',source)
        source=re.sub(r'i:So/Beam/NumberOfHistoriesInRun = \d+',f'i:So/Beam/NumberOfHistoriesInRun = {a.histories}',source)
        if a.energy is not None:
            source,n=re.subn(r'd:So/Beam/BeamEnergy = .*',f'd:So/Beam/BeamEnergy = {a.energy:.17g} MeV',source)
            assert n==1
        source,n=re.subn(r'd:Ge/ROI/MaxStepSize = .*',f'd:Ge/ROI/MaxStepSize = {a.max_step:.17g} mm',source)
        assert n==1
        if a.homogeneous_box:
            material=re.search(r'sv:Ge/ROI/MaterialNames = 1 "([^"]+)"',source)[1]
            remove=['InputDirectory','InputFile','ImagingToMaterialConverter','NumberOfVoxelsX','NumberOfVoxelsY','NumberOfVoxelsZ',
                    'VoxelSizeX','VoxelSizeY','VoxelSizeZ','MaterialTagNumbers','MaterialNames']
            source='\n'.join(l for l in source.splitlines() if not any(':Ge/ROI/'+key+' ' in l for key in remove))+'\n'
            source=source.replace('s:Ge/ROI/Type = "TsImageCube"','s:Ge/ROI/Type = "TsBox"')
            source+=f's:Ge/ROI/Material = "{material}"\n'+''.join(f'd:Ge/ROI/HL{axis} = 20 mm\n' for axis in 'XYZ')
        if not a.raw:
            source='\n'.join(l for l in source.splitlines() if ':Sc/Steps/' not in l)+'\n'
        source+='''
s:Sc/Family/Quantity = "ElectronFamilyProfile"
s:Sc/Family/Component = "Envelope"
s:Sc/Family/OutputType = "binary"
s:Sc/Family/OutputFile = "family"
s:Sc/Family/IfOutputFileAlreadyExists = "Exit"
b:Sc/Family/PropagateToChildren = "True"
'''
        (d/'topas.txt').write_text(source); start=time.monotonic()
        with (d/'topas.log').open('x') as log:
            code=subprocess.run([str(a.binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        text=(d/'topas.log').read_text(); counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',text)
        unscored=sum(map(float,re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',text)))
        case=dict(metadata,seed=a.seed+i,histories=a.histories,threads=a.threads,
                  energy_MeV=a.energy if a.energy is not None else metadata['energy_MeV'],
                  requested_max_step_mm=a.max_step,
                  geometry='homogeneous_box_in_vacuum' if a.homogeneous_box else metadata['geometry'],
                  accepted=bool(code==0 and counts and int(counts[-1])==a.histories and unscored==0),
                  binary_sha256=sha(a.binary),input_sha256=sha(d/'topas.txt'),exit_code=code,
                  seconds=time.monotonic()-start,unscored_MeV=unscored,path=str(d/'steps.phsp'))
        case.pop('sha256',None)
        if a.raw and (d/'steps.phsp').exists():case['sha256']=sha(d/'steps.phsp')
        save(d/'status.json',case);cases.append(case);save(a.out/'manifest.json',cases)
        assert case['accepted'],case
        r=read_summary(d/'family.phsp',a.histories)
        summaries={key:summarize(profile,float(r['incident'][0])) for key,profile in
                   [('charged',r['profile']),('photon',r['photon_profile']),('full',r['profile']+r['photon_profile'])]}
        for s in summaries.values():
            for k,v in s.items():
                if isinstance(v,np.ndarray):s[k]=v.tolist()
        result=dict(name=case['name'],summaries=summaries,summary_sha256=sha(d/'family.phsp'),
                    steps=int(r['steps'].sum()),max_step_mm=float(r['max_step'].max()),
                    certified_internal_null_navigation_steps=int(r['navigation'].sum()) if 'navigation' in r.dtype.names else 0,
                    max_internal_navigation_path_mm=float(r['navigation_path'].max()) if 'navigation_path' in r.dtype.names else 0.,
                    fractions={key:float(r[key].sum()/r['incident'].sum()) for key in ['charged','photon','sink','escape']},
                    max_full_energy_residual_MeV=float(np.max(abs(r['incident']-r['charged']-r['photon']-r['escape']))))
        results.append(result);print(json.dumps(result),flush=True)
    save(a.out/'report.json',dict(status='NATIVE_FULL_FAMILY_REFERENCE',ct_acceptance=False,results=results,
         settings=cases[0],profile_method='Exact lab-x chord overlaps for charged particles; photon transfer only at interaction endpoint',
         source_manifest_sha256=sha(a.out/'manifest.json'),script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
