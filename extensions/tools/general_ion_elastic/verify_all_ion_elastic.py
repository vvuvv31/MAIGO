#!/usr/bin/env python3
"""Verify runtime elastic/recoil packages; optionally audit external extraction provenance."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from compile_all_ion_elastic import SPECIES
REPO=Path(__file__).resolve().parents[3]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=REPO/'data/schneider');ap.add_argument('--audit-provenance',action='store_true');a=ap.parse_args()
 p=a.data/'all_ion_elastic_v1.bin';m=json.loads(p.with_suffix('.metadata.json').read_text())
 assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'];assert m['projectiles']==[list(x) for x in SPECIES]
 with p.open('rb') as f:
  assert f.read(8)==b'ELBANK01';version,np_,ns,nt,ne,nq=struct.unpack('<6I',f.read(24));assert (version,np_,ns,nt)==(1,18,26,13)
  masses=np.fromfile(f,'<f8',18);e=np.fromfile(f,'<f4',ne);r=np.fromfile(f,'<f4',18*26*13*ne).reshape(18,26,13,ne)
  samples=np.fromfile(f,dtype=[('fraction','<f4'),('mass','<f4'),('a','<u4')],count=18*13*ne*nq).reshape(18,13,ne,nq);assert not f.read(1)
 assert np.all(np.isfinite(masses)&(masses>0)) and np.all(np.diff(e)>0) and np.allclose(e,m['energies_MeVu'])
 assert np.all(np.isfinite(r)&(r>=0));assert np.all(np.isfinite(samples['fraction'])&(samples['fraction']>=0)&(samples['fraction']<=1))
 for i in range(18):
  for t in range(13):
   active=np.any(r[i,:,t,:]>0,axis=0);assert np.all(samples['a'][i,t,active]>0)
 q=a.data/'elastic_recoil_stopping_v1.bin';qm=json.loads(q.with_suffix('.metadata.json').read_text());assert sha(q)==qm['sha256'] and q.stat().st_size==qm['bytes']
 with q.open('rb') as f:
  assert f.read(8)==b'ELRSP001';np_,ns,ne=struct.unpack('<3I',f.read(12));assert ns==26
  keys=np.fromfile(f,'<i4',2*np_).reshape(np_,2);se=np.fromfile(f,'<f4',ne);sp=np.fromfile(f,'<f4',np_*26*ne);assert not f.read(1)
 assert len(set(map(tuple,keys)))==np_ and np.all(np.diff(se)>0) and np.all(np.isfinite(sp)&(sp>0))
 for t,z in enumerate(m['target_z']):
  for A in np.unique(samples['a'][:,t]):
   if A:assert (z,int(A)) in set(map(tuple,keys)),(z,A)
 if a.audit_provenance:
  for key,h in m['input_pins'].items():assert sha(m.get('source_snapshots',{}).get(key,key))==h,key
  for key,h in m['raw_sha256'].items():assert sha(key)==h,key
  for key,h in qm['input_sha256'].items():assert sha(qm.get('source_snapshots',{}).get(key,key))==h,key
 print(f'PASS: 18 elastic projectiles, 26 materials, 13 targets, {len(e)} energy nodes, {nq} samples/node; {np_} recoil isotopes; SHA and coverage verified')
if __name__=='__main__':main()
