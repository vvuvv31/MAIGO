#!/usr/bin/env python3
"""Compare native conditional draws with independent analog first collisions."""
import argparse,json
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE,sha,save


def features(raw,proc,analog):
 c=raw[(raw['kind']==0)&(raw['process']==proc)&((raw['step']==1) if analog else True)]
 h=raw[(raw['kind']==1)&(raw['process']==proc)&((raw['step']==1) if analog else True)&(raw['remaining']>0)]
 assert len(c)>100
 order=np.argsort(c['event']);c=c[order];indices=np.searchsorted(c['event'],h['event']);assert np.array_equal(c['event'][indices],h['event'])
 n=len(c);x=np.zeros((n,9+3*64));x[:,0]=c['deposit']/c['energy'];x[:,1]=c['remaining']/c['energy']
 cos=np.sum(c['pre']*c['post'],axis=1);x[:,2]=x[:,1]*cos;x[:,3]=x[:,1]*cos*cos
 frac=h['remaining']/h['energy'];hc=np.sum(h['pre']*h['post'],axis=1);electron=h['pdg']==11
 np.add.at(x[:,4],indices[electron],frac[electron]);np.add.at(x[:,5],indices[electron],frac[electron]*hc[electron]);np.add.at(x[:,6],indices[electron],frac[electron]*hc[electron]**2)
 np.add.at(x[:,7],indices[~electron],frac[~electron]);np.add.at(x[:,8],indices[~electron],frac[~electron]*hc[~electron])
 # Event-level energy-weighted joint histograms retain correlations and
 # correctly handle multiple children from one photoelectric interaction.
 for kind,events,f,d in [(0,np.arange(n),x[:,1],cos),(1,indices[electron],frac[electron],hc[electron]),(2,indices[~electron],frac[~electron],hc[~electron])]:
  eb=np.minimum((np.maximum(f,0)*8).astype(int),7);ab=np.minimum((np.maximum(d+1,0)*4).astype(int),7)
  np.add.at(x,(events,9+kind*64+eb*8+ab),f)
 assert np.max(abs(x[:,[0,1,4,7]].sum(axis=1)-1))<1e-8
 return x


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analog',type=Path,required=True);p.add_argument('--conditional',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',type=int,nargs='+',default=[57,12,32]);a=p.parse_args();reports=[]
 for node in a.nodes:
  analog=a.analog/f'node{node:02d}/photons.phsp';conditional=a.conditional/f'node{node:02d}/conditional.phsp';r=np.fromfile(analog,DTYPE);s=np.fromfile(conditional,DTYPE);checks=[]
  for proc in range(3):
   x=features(r,proc,True);y=features(s,proc,False);mx=x.mean(axis=0);my=y.mean(axis=0)
   # Under the hypothesis being tested the entire event distributions are
   # identical, so estimate their common variance from both samples. A zero
   # empirical variance in a small analog rare-tail cell is not certainty of
   # zero probability (e.g.0/1463 vs145/100000 is compatible with~2 expected).
   pooled=((len(x)-1)*x.var(axis=0,ddof=1)+(len(y)-1)*y.var(axis=0,ddof=1))/(len(x)+len(y)-2)
   se=np.sqrt(pooled*(1/len(x)+1/len(y)));err=abs(mx-my);limit=6*se+2e-7
   # Also test histogram occupancy with an exact test, independent of weights.
   from scipy.stats import fisher_exact
   occupancy=[]
   for j in range(9,x.shape[1]):
    nx=int(np.count_nonzero(x[:,j]));ny=int(np.count_nonzero(y[:,j]));pv=float(fisher_exact([[nx,len(x)-nx],[ny,len(y)-ny]]).pvalue)
    if pv<1e-6/(3*len(a.nodes)*(x.shape[1]-9)):occupancy.append(dict(feature=j,p_value=pv,counts=[nx,ny]))
   checks.append(dict(process=['phot','compt','Rayl'][proc],analog_collisions=len(x),conditional_collisions=len(y),features=x.shape[1],passed=bool(np.all(err<=limit) and not occupancy),max_standardized_difference=float(np.max(err/np.maximum(se,2e-7))),failed_features=np.flatnonzero(err>limit).tolist(),failed_occupancy=occupancy,analog_means=mx[:9].tolist(),conditional_means=my[:9].tolist(),SE=se[:9].tolist(),statistical_scope='Common-variance null test plus exact Fisher histogram occupancy with family-wise threshold1e-6; finite samples do not prove arbitrary rare-tail equivalence'))
  reports.append(dict(node=node,analog_sha256=sha(analog),conditional_sha256=sha(conditional),checks=checks))
 result=dict(status='NATIVE_CONDITIONAL_VS_ANALOG_FIRST_COLLISION',passed=all(c['passed'] for r in reports for c in r['checks']),results=reports,scope='Independent monoenergetic process-conditioned energy/angle moments and8x8 energy-weighted branch histograms; does not certify continuous-energy interpolation or CT dose')
 save(a.out,result);print(json.dumps({k:result[k] for k in ['status','passed']},indent=2))
 for r in reports:
  for c in r['checks']:print(r['node'],c['process'],c['max_standardized_difference'],c['failed_features'])
 assert result['passed']
if __name__=='__main__':main()
