#!/usr/bin/env python3
from pathlib import Path
import argparse,json,numpy as np
from scipy.stats import binomtest,poisson
p=argparse.ArgumentParser(description='Exact sparse-count screens for fixed-energy multiple scattering.');p.add_argument('root',type=Path);p.add_argument('--samples',type=int,default=131072);a=p.parse_args()
cases=json.loads((a.root/'cases.json').read_text());N=a.samples;DT=np.dtype([('position','<f4',3),('direction','<f4',3),('events','<u4'),('mode','<u4')]);rows=[];fail=[];family=len(cases)*2*35
for arm in ['reference','candidate']:
 raw=np.memmap(a.root/f'packets.{arm}.bin',DT,mode='r').reshape(len(cases),N)
 for i,c in enumerate(cases):
  counts=raw[i]['events'];mu=float(np.float32(c['mu']));tests=[]
  for k in range(34):
   probability=float(poisson.pmf(k,mu)) if k<33 else float(poisson.sf(32,mu));observed=int(np.count_nonzero(counts==k)) if k<33 else int(np.count_nonzero(counts>32));pv=float(binomtest(observed,N,probability).pvalue)
   if pv<.01/family:fail.append([arm,i,k,pv])
   tests.append(dict(count=k if k<33 else '>32',observed=observed,probability=probability,p=pv))
  condensed=int(np.count_nonzero(raw[i]['mode']==7));eligible=float(np.float32(mu)*np.float32(c['mean_q']))<=.02
  expected=float(poisson.sf(4,mu)) if eligible and arm=='candidate' else 0.
  pv=float(binomtest(condensed,N,expected).pvalue) if expected else float(condensed==0)
  if pv<.01/family:fail.append([arm,i,'condensed',pv])
  rows.append(dict(arm=arm,case=i,node=c['node'],energy=c['energy'],mu=mu,maximum_count=int(counts.max()),condensed=condensed,expected_condensed=expected,condensed_test_p=pv,tests=tests))
report=dict(scope='Fixed-energy homogeneous counts only; exact binomial family alpha0.01, not normal Z for sparse tails.',passed=not fail,failures=fail,cases=len(cases),histories_per_case_per_arm=N,results=rows)
(a.root/'count_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
raise SystemExit(0 if report['passed'] else 1)
