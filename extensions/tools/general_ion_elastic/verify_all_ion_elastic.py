#!/usr/bin/env python3
"""Audit elastic/recoil binary layout, physical values and isotope coverage."""
import argparse,struct
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=REPO/'data/elastic/schneider/carbon');a=ap.parse_args()
 p=a.data/'all_ion.bin'
 with p.open('rb') as f:
  assert f.read(8)==b'ELBANK01';version,np_,ns,nt,ne,nq=struct.unpack('<6I',f.read(24));assert (version,np_,ns,nt)==(1,18,26,13)
  masses=np.fromfile(f,'<f8',18);e=np.fromfile(f,'<f4',ne);r=np.fromfile(f,'<f4',18*26*13*ne).reshape(18,26,13,ne)
  samples=np.fromfile(f,dtype=[('fraction','<f4'),('mass','<f4'),('a','<u4')],count=18*13*ne*nq).reshape(18,13,ne,nq);assert not f.read(1)
 assert np.all(np.isfinite(masses)&(masses>0)) and np.all(np.diff(e)>0)
 assert np.all(np.isfinite(r)&(r>=0));assert np.all(np.isfinite(samples['fraction'])&(samples['fraction']>=0)&(samples['fraction']<=1))
 for i in range(18):
  for t in range(13):
   active=np.any(r[i,:,t,:]>0,axis=0);assert np.all(samples['a'][i,t,active]>0)
 q=a.data/'recoil_stopping.bin'
 with q.open('rb') as f:
  assert f.read(8)==b'ELRSP001';np_,ns,ne=struct.unpack('<3I',f.read(12));assert ns==26
  keys=np.fromfile(f,'<i4',2*np_).reshape(np_,2);se=np.fromfile(f,'<f4',ne);sp=np.fromfile(f,'<f4',np_*26*ne);assert not f.read(1)
 assert len(set(map(tuple,keys)))==np_ and np.all(np.diff(se)>0) and np.all(np.isfinite(sp)&(sp>0))
 for t,z in enumerate((1,6,7,8,12,15,16,17,18,20,11,19,22)):
  for A in np.unique(samples['a'][:,t]):
   if A:assert (z,int(A)) in set(map(tuple,keys)),(z,A)
 print(f'PASS: 18 elastic projectiles, 26 materials, 13 targets, {len(e)} energy nodes, {nq} samples/node; {np_} recoil isotopes; layout, finite values and coverage verified')
if __name__=='__main__':main()
