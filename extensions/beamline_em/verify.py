#!/usr/bin/env python3
"""Check native held-out full responses; stamp immutable package provenance."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path);p.add_argument('heldout',type=Path);p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
 b=a.package.read_bytes();magic,version,nm,ns,nn=struct.unpack_from('<8s4I',b)
 if (magic,version,nm)!=(b'BLEM0001',1,2) or ns==0:raise ValueError('Invalid full response header')
 offset=24+nm*12;species=np.frombuffer(b,dtype=np.dtype([('z','<u4'),('a','<u4'),('mass','<f4')]),count=ns,offset=offset)
 offset+=ns*12;records=np.frombuffer(b,dtype='<u4',count=nm*ns*2,offset=offset).reshape(nm,ns,2);offset+=nm*ns*8
 nodes=np.frombuffer(b,dtype='<f4',count=nn*4,offset=offset).reshape(nn,4)
 if offset+nn*16!=len(b) or not np.isfinite(nodes).all() or np.any(nodes[:,:2]<=0) or np.any(nodes[:,2:]<0):raise ValueError('Invalid nodes/size')
 maxima=np.zeros(3);worst=[None]*3;n=0
 for r in csv.DictReader(a.heldout.open()):
  m,i=int(r['material']),int(r['projectile']);start,count=records[m,i];v=nodes[start:start+count];e=float(r['energy'])
  j=int(np.searchsorted(v[:,0],e,side='right')-1);j=max(0,min(j,len(v)-2));w=(e-v[j,0])/(v[j+1,0]-v[j,0]);estimated=v[j,1:].astype(float)+w*(v[j+1,1:].astype(float)-v[j,1:])
  ref=np.array([float(r[k]) for k in ('stopping','ion_variance','universal_variance')]);err=np.abs(estimated-ref)/np.maximum(ref,1e-30)
  for k in range(3):
   if err[k]>maxima[k]:maxima[k]=err[k];worst[k]=dict(material=m,z=int(species[i]['z']),a=int(species[i]['a']),energy=e,relative_error=err[k])
  n+=1
 result=dict(schema='BLEM0001',geant4='11.3.2',materials=['G4_Cu','G4_AIR'],loss='unrestricted electronic',variance='native Dispersion(tcut=Tmax), MeV^2/mm',species=[[int(x['z']),int(x['a'])] for x in species],nodes=nn,heldout_points=n,max_relative_errors=maxima.tolist(),worst=worst,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),low_energy='sqrt(E) stopping continuation below first node; R=2E/S',range='analytic integral of reciprocal piecewise-linear stopping',extractor_sha256=hashlib.sha256(Path(__file__).with_name('extract.cpp').read_bytes()).hexdigest())
 a.metadata.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['species']},indent=2))
 if maxima.max()>5e-4:raise ValueError('Full response held-out error exceeds 5e-4')
if __name__=='__main__':main()
