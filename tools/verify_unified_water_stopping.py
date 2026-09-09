"""Verify candidate water data integrity, NOT authorize production use."""
import hashlib
import json
from pathlib import Path
import numpy as np

def verify(repo):
    repo=Path(repo)
    meta=json.loads((repo/'data/water_unified/g4_water78_stopping_v1.metadata.json').read_text())
    if meta['material']!='G4_WATER' or meta['mean_excitation_energy_eV']!=78:
        raise ValueError('Wrong water material')
    expected={f'data/water_unified/g4_water78_{name}_v1.csv' for name in ('primary','ions')}
    if set(meta['files'])!=expected:raise ValueError('Wrong file set')
    for name,pin in meta['files'].items():
        p=repo/name
        if p.stat().st_size!=pin['size_bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']:
            raise ValueError('Water stopping size/SHA mismatch: '+name)
    primary=np.loadtxt(repo/'data/water_unified/g4_water78_primary_v1.csv',delimiter=',')
    ions=np.loadtxt(repo/'data/water_unified/g4_water78_ions_v1.csv',delimiter=',')
    if primary.shape!=(4001,2) or ions.shape!=(128032,8):raise ValueError('Wrong table shape')
    if not np.isfinite(primary).all() or not np.isfinite(ions).all():raise ValueError('Nonfinite payload')
    if np.any(primary[:,1]<=0) or np.any(ions[:,3]<=0):raise ValueError('Nonpositive stopping')
    if not np.allclose(primary[:,0],.01+.1*np.arange(4001),rtol=0,atol=1e-9):raise ValueError('Wrong domain/grid')
    keys=set()
    for block in ions.reshape(32,4001,8):
        if not np.array_equal(block[:,2],primary[:,0]) or not np.all(block[:,:2]==block[0,:2]):raise ValueError('Ion grid/identity')
        keys.add(tuple(block[0,:2]))
    if len(keys)!=32:raise ValueError('Duplicate species')
    c=ions[(ions[:,0]==6)&(ions[:,1]==12),3]
    if c.shape!=(4001,) or not np.allclose(c,primary[:,1],rtol=2e-5,atol=0):raise ValueError('C12 mismatch')
    return meta

if __name__=='__main__':
    verify(Path(__file__).resolve().parents[1])
    print('Candidate G4_WATER 78eV integrity PASS: 4001 nodes, 32 species, max 400.01 MeV/u. Not production acceptance.')
