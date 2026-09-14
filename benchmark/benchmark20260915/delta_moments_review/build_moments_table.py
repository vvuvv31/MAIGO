"""Derived delta first and raw second moments at the verified EM package's existing nodes."""
from pathlib import Path
import hashlib,struct,json,sys
import numpy as np
R=Path(__file__).resolve().parent;repo=R.parents[2];sys.path.insert(0,str(R))
# Reuse the binary reader and preserve its data consistency report.
import check_mean_data as data
records=data.records;nodes=data.nodes;nn=data.nn
out=repo/'scratch/delta_moments_current_20260915';out.mkdir(exist_ok=True,parents=True)
path=out/'delta_moments.bin';assert not path.exists()
x,w=np.polynomial.legendre.leggauss(32)
def mean_values(r,n,order=32,moment=1):
 xx,ww=(x,w) if order==32 else np.polynomial.legendre.leggauss(order)
 k=n[:,0]*r['a'];mass=float(r['mass']);u=k/mass;ratio=.51099891/mass;tm=2*.51099891*u*(u+2)/(1+2*(u+1)*ratio+ratio*ratio);cut=float(r['cut']);out=np.zeros(len(n));valid=(tm>cut*1.0000001)&(n[:,5]>0)
 if not valid.any():return out
 u=u[valid];k=k[valid];tm=tm[valid];beta=u*(u+2)/(u+1)**2;total2=(k+mass)**2;spin=float(r['spin'])
 den=1/cut-1/tm-beta/tm*np.log(tm/cut)+(0.5*(tm-cut)/total2 if spin>0 else 0)
 half=.5*np.log(tm/cut);mid=np.log(cut)+half;e=np.exp(mid[:,None]+half[:,None]*xx)
 f1=.5*e*e/total2[:,None] if spin>0 else np.zeros_like(e);f=1-beta[:,None]*e/tm[:,None]+f1;ff=float(r['ff'])*e
 acc=np.ones_like(e);use=ff>1e-6;veto=1/(1+ff)**2
 if spin>0:
  x2=.5*.51099891*e/(mass*mass);veto*=1+float(r['mm2'])*(x2-f1/f)/(1+x2)
 acc[use]=np.clip(veto[use],0,1)
 m=np.sum(ww[None,:]*f*acc*half[:,None]*e**(moment-1),axis=1)/den
 out[valid]=n[valid,5]*m;return out
with path.open('wb') as f:f.write(struct.pack('<8sII',b'EMDMOMT2',2,nn));f.truncate(16+nn*8)
values=np.memmap(path,dtype='<f4',mode='r+',offset=16,shape=(nn,2));maxerr=0;checked=0
for idx,r in enumerate(records):
 n=np.asarray(nodes[r['off']:r['off']+r['count']],dtype='f8')
 for moment in (1,2):
  v=mean_values(r,n,moment=moment);assert np.isfinite(v).all() and (v>=0).all();values[r['off']:r['off']+r['count'],moment-1]=v
  pick=np.linspace(0,len(n)-1,20).astype(int);ref=mean_values(r,n[pick],64,moment);mask=ref>1e-12
  if mask.any():maxerr=max(maxerr,float(np.max(abs(v[pick][mask]/ref[mask]-1))));checked+=int(mask.sum())
 if idx%500==0:print('records',idx,'/',len(records),flush=True)
values.flush();assert maxerr<1e-4,maxerr
m=dict(schema='EMDMOMT2',version=2,nodes=nn,records=len(records),source_package_sha256=hashlib.sha256(data.p.read_bytes()).hexdigest(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,quadrature_order=32,check_order=64,checked_nodes=checked,max_relative_quadrature_error=maxerr,model='native proposal rate times accepted delta first and raw second moments; same spin/form-factor/magnetic veto as GPU sampler; all existing material/density/projectile nodes',data_path=str(path))
(R/'moments_table_manifest.json').write_text(json.dumps(m,indent=2));print(m)
