#!/usr/bin/env python3
"""Compile explicitly pinned TOPAS elastic extraction; no synthesized channels."""
import argparse,json,struct,hashlib
from pathlib import Path
import numpy as np
SPECIES=[(1,1),(1,2),(1,3),(2,3),(2,4),(2,6),(3,6),(3,7),(4,7),(4,9),(4,10),(5,8),(5,10),(5,11),(6,10),(6,11),(6,12),(4,6)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
 assert not a.output.exists();pins=json.loads((a.campaign/'provenance.json').read_text())
 resolution_path=a.campaign/"provenance_resolution.json"
 resolution=json.loads(resolution_path.read_text()) if resolution_path.exists() else {}
 for p,h in pins.items():assert sha(resolution.get(p,p))==h,p
 masses=[];rates=[];samples=[];sources={};grid=None;nq=None
 dtype=np.dtype([('fraction','<f4'),('mass','<f4'),('a','<u4')])
 for z,A in SPECIES:
  p=a.campaign/'raw'/f'z{z}a{A}.bin';meta=p.with_suffix('.json');sources[str(p)]=sha(p);sources[str(meta)]=sha(meta)
  with p.open('rb') as f:
   assert f.read(8)==b'ELRAW001';zz,aa,ns,nt,ne,q=struct.unpack('<6I',f.read(24));assert (zz,aa,ns,nt)==(z,A,26,13)
   m=struct.unpack('<d',f.read(8))[0];energies=np.fromfile(f,'<f8',ne);r=np.fromfile(f,'<f8',26*13*ne).reshape(26,13,ne);s=np.fromfile(f,dtype,13*ne*q).reshape(13,ne,q);assert not f.read(1)
  if grid is None:grid=energies;nq=q
  assert np.array_equal(grid,energies) and nq==q
  assert np.all(np.isfinite(r)&(r>=0)) and np.all(np.isfinite(s['fraction'])&(s['fraction']>=0)&(s['fraction']<=1))
  for t in range(13):
   for e in range(ne):
    if np.any(r[:,t,e]>0):assert np.all(s['a'][t,e]>0),('positive rate without samples',z,A,t,e)
  masses.append(m);rates.append(r.astype('<f4'));samples.append(s)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('wb') as f:
  f.write(b'ELBANK01');f.write(struct.pack('<6I',1,18,26,13,len(grid),nq));np.array(masses,'<f8').tofile(f);grid.astype('<f4').tofile(f)
  for r in rates:r.tofile(f)
  for s in samples:s.tofile(f)
 metadata={'schema':1,'projectiles':SPECIES,'materials':'25 Schneider sections plus G4_WATER','target_z':[1,6,7,8,12,15,16,17,18,20,11,19,22],'samples_per_node':nq,'energies_MeVu':grid.tolist(),'sha256':sha(a.output),'bytes':a.output.stat().st_size,'input_pins':pins,'source_snapshots':resolution,'raw_sha256':sources,'sampling':'Actual attached TOPAS elastic model ApplyYourself; target isotopes sampled by cross-section store. Joint target isotope/mass and t/tmax samples; rates per Schneider section; final-state samples on pure elements with original isotope compositions.','limitation':'Candidate finite energy/sample bank; pure-element angular response requires material-independence validation. Not a production acceptance claim.'}
 a.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata['sha256'])
if __name__=='__main__':main()
