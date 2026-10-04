from pathlib import Path
import argparse,json,numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import ks_2samp
parser=argparse.ArgumentParser(description='Isolated condensed soft-elastic distribution screen; not CT acceptance.')
parser.add_argument('root',type=Path)
parser.add_argument('--samples',type=int,default=131072)
args=parser.parse_args();B=args.root;cases=json.loads((B/'cases.json').read_text());N=args.samples
DT=np.dtype([('position','<f4',3),('direction','<f4',3),('events','<u4'),('mode','<u4')]);data={arm:np.memmap(B/f'packets.{arm}.bin',DT,mode='r').reshape(len(cases),N) for arm in ['reference','candidate']};nodes,weights=leggauss(64);t=(nodes+1)/2;w=weights/2
rows=[]
for i,case in enumerate(cases):
 a=case['mu']*case['mean_q'];b=case['mu']*case['second_q'];c=3*a-1.5*b;L=case['length']
 theory={'q':float(-np.expm1(-a)),'transport2':float(-np.expm1(-c)),'x2':float(2*L*L/(3*a)*np.dot(w,(-np.expm1(-c*t))*(-np.expm1(-a*(1-t))))),'x_ux':float(L/3*np.dot(w,(-np.expm1(-c*t))*np.exp(-a*(1-t))))}
 row=dict(case=i,**case,theory=theory,arms={});radii={};angles={}
 for arm,x in data.items():
  d=x[i]['direction'].astype(float);p=x[i]['position'].astype(float);assert np.all(x[i]['mode']>0) and np.isfinite(p).all() and np.isfinite(d).all();q=1-d[:,2];angles[arm]=q
  obs={'q':q,'transport2':3*q-1.5*q*q,'x2':p[:,0]**2,'x_ux':p[:,0]*d[:,0]};metrics={}
  for k,y in obs.items():
   se=float(y.std(ddof=1)/np.sqrt(N));delta=float(y.mean()-theory[k]);metrics[k]=dict(mean=float(y.mean()),standard_error=se,delta_percent=100*delta/theory[k],z=abs(delta)/se,screen_pass=bool(abs(delta)<=7*se+1e-4*theory[k]))
  radii[arm]=np.linalg.norm(p[:,:2],axis=1);zero=(x[i]['events']==0) if arm=='reference' else ((x[i]['mode']==2)|((x[i]['mode']==1)&(x[i]['events']==0)))
  expected=np.exp(-case['mu']);metrics['zero_scatter']=dict(fraction=float(zero.mean()),expected=float(expected),difference=float(zero.mean()-expected))
  row['arms'][arm]=dict(metrics=metrics,max_norm_error=float(abs(np.linalg.norm(d,axis=1)-1).max()),max_chord_over_length=float(np.linalg.norm(p,axis=1).max()/L),radial_quantiles_mm={str(v):float(np.percentile(radii[arm],v)) for v in [50,90,95,99,99.9]})
 row['angular_q_ks']=float(ks_2samp(angles['reference'],angles['candidate']).statistic)
 row['radial_ks']=float(ks_2samp(radii['reference'],radii['candidate']).statistic)
 row['radial_quantile_change_percent']={k:100*(row['arms']['candidate']['radial_quantiles_mm'][k]/v-1) if v>0 else None for k,v in row['arms']['reference']['radial_quantiles_mm'].items()}
 rows.append(row)
report=dict(scope='Fixed energy homogeneous SOFT-elastic segment only, 0.05mm; no hard events/energy loss/interfaces/CT or throughput acceptance.',cases=len(rows),samples_per_case_per_arm=N,angular_moment_failures=[(r['case'],arm,k) for r in rows for arm in ['reference','candidate'] for k in ['q','transport2'] if not r['arms'][arm]['metrics'][k]['screen_pass']],spatial_moment_failures=[(r['case'],arm,k) for r in rows for arm in ['reference','candidate'] for k in ['x2','x_ux'] if not r['arms'][arm]['metrics'][k]['screen_pass']],maximum_radial_KS=max(r['radial_ks'] for r in rows),maximum_angular_q_KS=max(r['angular_q_ks'] for r in rows),geometry_failures=[(r['case'],arm) for r in rows for arm in ['reference','candidate'] if r['arms'][arm]['max_chord_over_length']>1+8*np.finfo(np.float32).eps or r['arms'][arm]['max_norm_error']>8*np.finfo(np.float32).eps],results=rows)
(B/'comparison.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='results'})
