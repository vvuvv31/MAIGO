#!/usr/bin/env python3
"""Separate operator implementation checks from independent native comparisons."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE,sha,save
from compile_photon_response import OUTCOME

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--energy',type=float,default=.1);a=p.parse_args();d=a.operator
 m=json.loads((d/'manifest.json').read_text());assert sha(d/'operator.bin')==m['sha256']
 with (d/'operator.bin').open('rb') as f:
  magic,version,section,rho,nrate,nbin,nout=struct.unpack('<8sIidIIQ',f.read(40));assert magic==b'MAIPHN01' and version==1
  rates=np.fromfile(f,'<f8',nrate*4).reshape(-1,4);edges=np.fromfile(f,'<f8',nbin+1);channels=np.fromfile(f,'<u8',nbin*6).reshape(-1,2);outcomes=np.fromfile(f,OUTCOME,nout)
 energy=a.energy;coeff=np.array([np.interp(energy,rates[:,0],rates[:,i]) for i in range(1,4)]);prob=coeff/coeff.sum();binid=min(nbin-1,np.searchsorted(edges,energy,side='right')-1)
 x=np.fromfile(d/'samples.bin','<f8').reshape(-1,6);n=len(x);assert np.isfinite(x).all() and np.all(x[:,0]<3)
 assert np.max(abs(np.linalg.norm(x[x[:,0]>0,2:5],axis=1)-1),initial=0)<1e-10
 expected={};sample={};checks={}
 for name,field in [('probability',None),('energy',1),('cosine',4)]:
  ex=np.zeros(3);second=np.zeros(3)
  for proc in range(3):
   offset,count=channels[binid*3+proc];rows=outcomes[int(offset):int(offset+count)];assert len(rows)>0 or prob[proc]==0
   weight=np.diff(np.r_[0.,rows['cdf']])*prob[proc]
   kind=np.where(rows['pdg']==0,0,np.where(rows['pdg']==11,1,2))
   values=np.ones(len(rows)) if field is None else (rows['fraction']*energy if field==1 else np.where(kind==0,0,rows['cosine']))
   ex+=np.bincount(kind,weights=weight*values,minlength=3);second+=np.bincount(kind,weights=weight*values**2,minlength=3)
  obs=np.bincount(x[:,0].astype(int),weights=None if field is None else x[:,field],minlength=3)/n
  se=np.sqrt(np.maximum(0,second-ex**2)/n);assert np.all(abs(obs-ex)<=6*se+1e-12),(name,obs,ex,se)
  checks[name]=dict(observed=obs.tolist(),operator_expectation=ex.tolist(),SE=se.tolist());expected[name]=ex;sample[name]=obs
 # All primary first interactions, each contributes fractions summing to one.
 raw=np.fromfile(a.native/'photons.phsp',DTYPE);c=raw[(raw['kind']==0)&(raw['step']==1)];ch=raw[(raw['kind']==1)&(raw['step']==1)]
 H=len(c);assert H>100 and np.array_equal(np.sort(c['event']),np.arange(H))
 assert np.max(abs(c['energy']-energy))<1e-7*energy
 birth_cos=np.sum(ch['pre']*ch['post'],axis=1);gamma_cos=np.sum(c['pre']*c['post'],axis=1)
 key=c['event'];native={}
 for name in expected:
  sums=np.zeros((H,3))
  branches=[(key,np.zeros(H,int),c['deposit'],np.zeros(H)),(key,np.full(H,2),c['remaining'],gamma_cos),(ch['event'],np.where(ch['pdg']==11,1,2),ch['remaining'],birth_cos)]
  for ev,kind,e,cos in branches:
   factor=np.ones(len(ev)) if name=='probability' else (e if name=='energy' else cos)
   np.add.at(sums,(ev,kind),e/c['energy'][0]*factor)
  mean=sums.mean(axis=0);se=sums.std(axis=0,ddof=1)/np.sqrt(H)
  native[name]=dict(mean=mean.tolist(),SE=se.tolist(),operator_minus_native=(expected[name]-mean).tolist(),note='Independent native first-collision distribution; finite-bin/source statistics remain, no automatic CT dose acceptance')
 report=dict(status='SAMPLER_IMPLEMENTATION_PASS_NATIVE_COMPARISON_REPORTED',operator_sha256=m['sha256'],native_sha256=sha(a.native/'photons.phsp'),samples=n,native_histories=H,implementation_checks=checks,native_comparison=native)
 save(d/'sampling_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
