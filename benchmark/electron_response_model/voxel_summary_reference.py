#!/usr/bin/env python3
"""Native voxel fixture with one audited x-profile row per primary history."""
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

DTYPE=np.dtype([('run','<i4'),('event','<i4'),('roots','<i4'),('steps','<i4'),('tracks','<i4'),
                ('incident','<f8'),('charged','<f8'),('sink','<f8'),('outside','<f8'),
                ('max_residual','<f8'),('max_step','<f8'),('profile','<f8',(36,))])


def read_summary(path, histories):
    assert path.stat().st_size == histories*DTYPE.itemsize
    rows=np.fromfile(path, DTYPE); rows=rows[np.argsort(rows['event'])]
    assert np.array_equal(rows['event'],np.arange(histories))
    assert len(np.unique(rows['run'])) == 1 and np.all(rows['roots']==1)
    assert np.all(rows['steps']>0) and np.all(rows['tracks']>0)
    assert np.all(rows['incident']==rows['incident'][0])
    assert np.all(rows['profile']>=0) and np.all(rows['sink']>=0)
    assert np.max(abs(rows['incident']-rows['charged']-rows['sink']))<1e-8
    assert np.max(abs(rows['profile'].sum(axis=1)-rows['charged']))<1e-10
    assert np.max(rows['max_residual'])<1e-8 and np.max(rows['outside'])<1e-12
    assert np.max(rows['max_step'])<=.25*(1+1e-6)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,default=Path('out/electron_voxel_lung065_smoke_v2_20261002'))
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--histories',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1002320001)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--raw',action='store_true',help='Also record full raw steps for paired scorer verification')
    a=p.parse_args(); a.out=a.out.resolve(); a.out.mkdir(parents=True,exist_ok=False)
    a.binary=a.binary.resolve(); inputs=json.loads((a.template/'manifest.json').read_text())
    assert len(inputs)==2
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
             LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    results=[]; cases=[]
    for i,metadata in enumerate(inputs):
        name=metadata['name']; d=a.out/name; d.mkdir()
        source=(a.template/name/'topas.txt').read_text()
        source=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {a.seed+i}',source)
        source=re.sub(r'i:Ts/NumberOfThreads = \d+',f'i:Ts/NumberOfThreads = {a.threads}',source)
        source=re.sub(r'i:So/Beam/NumberOfHistoriesInRun = \d+',f'i:So/Beam/NumberOfHistoriesInRun = {a.histories}',source)
        if not a.raw:
            source='\n'.join(line for line in source.splitlines() if ':Sc/Steps/' not in line)+'\n'
        source+='''
s:Sc/Profile/Quantity = "ElectronHistoryProfile"
s:Sc/Profile/Component = "Envelope"
s:Sc/Profile/OutputType = "binary"
s:Sc/Profile/OutputFile = "profile"
s:Sc/Profile/IfOutputFileAlreadyExists = "Exit"
b:Sc/Profile/PropagateToChildren = "True"
'''
        (d/'topas.txt').write_text(source)
        started=time.monotonic()
        with (d/'topas.log').open('x') as log:
            code=subprocess.run([str(a.binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        elapsed=time.monotonic()-started
        text=(d/'topas.log').read_text()
        counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',text)
        unscored=sum(map(float,re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',text)))
        case=dict(metadata,name=name,seed=a.seed+i,histories=a.histories,threads=a.threads,
                  accepted=bool(code==0 and counts and int(counts[-1])==a.histories and unscored==0),
                  binary_sha256=sha(a.binary),input_sha256=sha(d/'topas.txt'),exit_code=code,
                  seconds=elapsed,unscored_MeV=unscored,path=str(d/'steps.phsp'))
        if a.raw and (d/'steps.phsp').exists(): case['sha256']=sha(d/'steps.phsp')
        else: case.pop('sha256',None)
        save(d/'status.json',case); cases.append(case); save(a.out/'manifest.json',cases)
        if not case['accepted']: raise RuntimeError(f'Native summary reference failed: {d}')
        rows=read_summary(d/'profile.phsp',a.histories)
        result=summarize(rows['profile'],float(rows['incident'][0])); results.append(result)
        case.update(summary_sha256=sha(d/'profile.phsp'),summary_bytes=(d/'profile.phsp').stat().st_size,
                    actual_energy_MeV=float(rows['incident'][0]),max_step_mm=float(rows['max_step'].max()),
                    raw_steps=int(rows['steps'].sum()),max_energy_residual_MeV=float(rows['max_residual'].max()))
        save(d/'status.json',case); save(a.out/'manifest.json',cases)
        print(name,result['roi_mean'],result['roi_sem'],elapsed,flush=True)
    train,test=results
    np.savez_compressed(a.out/'local_profile.npz',x_mm=CENTRES,reference=test['mean'],test_sem=test['sem'],
                        training=train['mean'],training_sem=train['sem'])
    save(a.out/'report.json',dict(status='NATIVE_VOXEL_HISTORY_SUMMARY',ct_acceptance=False,
         sources=cases,train_roi=train['roi_mean'].tolist(),test_roi=test['roi_mean'].tolist(),
         train_roi_sem=train['roi_sem'].tolist(),test_roi_sem=test['roi_sem'].tolist(),
         scope='Homogeneous lung, .65 MeV slit; explicit first-photon energy sink, not full CT acceptance',
         script_sha256=sha(Path(__file__))))


if __name__=='__main__':
    main()
