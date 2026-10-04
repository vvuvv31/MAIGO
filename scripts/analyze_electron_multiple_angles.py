#!/usr/bin/env python3
"""Independent theory/moment/tail and reference-distribution checks."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import ks_2samp,binomtest
from prepare_electron_multiple_cdf import cdf_from_coefficients
p=argparse.ArgumentParser(description=__doc__);p.add_argument('tables',type=Path);p.add_argument('samples',type=Path);p.add_argument('--histories',type=int,default=32768);a=p.parse_args()
r=json.loads((a.tables/'report.json').read_text());laws=r['cases'];counts=r['max_count']+2;N=a.histories
coefs=np.fromfile(a.tables/'legendre_coefficients.bin','<f8').reshape(len(laws),r['degree']+1)
qtable=np.fromfile(a.tables/'quantiles.bin','<f4').reshape(len(laws),r['max_count']-1,r['knots']);prob=np.fromfile(a.tables/'probabilities.bin','<f8')
data={k:np.memmap(a.samples/f'angles.{k}.bin','<f4',mode='r').reshape(len(laws),counts,N) for k in ['reference','candidate']}
rows=[];moment_fail=[];tail_fail=[];ks_fail=[];tail_family=len(laws)*(r['max_count']-1)*2*7;ks_family=len(laws)*(counts-1)
for law,case in enumerate(laws):
 for n in range(counts):
  f=coefs[law];expect=np.array([-np.expm1(n*np.log(f[1])),-np.expm1(n*np.log(f[2]))]);l1,l2,l4=f[1]**n,f[2]**n,f[4]**n
  variance=np.maximum(0,np.array([(1+2*l2)/3-l1*l1,.2+2*l2/7+18*l4/35-l2*l2]));se=np.sqrt(variance/N)
  row=dict(law=law,node=case['node'],energy=case['energy'],count=n,expected=expect.tolist(),theoretical_standard_error=se.tolist(),arms={})
  if 2<=n<=r['max_count']:
   tails=np.array([.0001,.001,.01,.5,.99,.999,.9999]);cut=np.interp(tails,prob,qtable[law,n-2]).astype(np.float32).astype(float);cdf=cdf_from_coefficients(f,cut,[n])[:,0]
   assert np.all((cdf>0)&(cdf<1))
  for arm,raw in data.items():
   q=raw[law,n].astype(float);assert np.all(np.isfinite(q)) and np.all((q>=0)&(q<=2))
   means=np.array([q.mean(),(3*q-1.5*q*q).mean()]);delta=means-expect;tolerance=7*se+1e-4*expect
   passed=bool(np.all(abs(delta)<=tolerance)) if n else bool(np.all(q==0))
   if not passed:moment_fail.append([law,n,arm])
   info=dict(means=means.tolist(),relative_error=(delta/np.maximum(expect,1e-300)).tolist(),moment_pass=passed)
   if 2<=n<=r['max_count']:
    observed=(q[:,None]<=cut).sum(axis=0);pv=[float(binomtest(int(k),N,float(p)).pvalue) for k,p in zip(observed,cdf)];bad=[j for j,v in enumerate(pv) if v<.01/tail_family]
    for j in bad:tail_fail.append([law,n,arm,j])
    info['tail_checks']=dict(cut_q=cut.tolist(),expected_cdf=cdf.tolist(),observed_counts=observed.tolist(),exact_binomial_p=pv)
   row['arms'][arm]=info
  ks=ks_2samp(data['reference'][law,n],data['candidate'][law,n]);row['two_sample_KS']=float(ks.statistic);row['two_sample_p']=float(ks.pvalue)
  if n and ks.pvalue<.01/ks_family:ks_fail.append([law,n])
  rows.append(row)
report=dict(scope='Fixed-count angular component only; no spatial/energy/interpolation/CT acceptance.',laws=len(laws),conditions=len(rows),samples_per_condition_per_arm=N,total_samples_per_arm=len(rows)*N,moment_failures=moment_fail,tail_failures=tail_fail,KS_failures=ks_fail,maximum_two_sample_KS=max(x['two_sample_KS'] for x in rows),family_alpha=.01,passed=not(moment_fail or tail_fail or ks_fail),results=rows)
(a.samples/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
raise SystemExit(0 if report['passed'] else 1)
