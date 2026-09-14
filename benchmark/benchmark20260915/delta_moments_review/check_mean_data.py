from pathlib import Path
import numpy as np,struct,json
repo=Path(__file__).resolve().parents[3];p=repo/'data/em/unified_em_v1.bin'
with p.open('rb') as f:magic,v,nm,ns,nr,nn,nv=struct.unpack('<8s6I',f.read(32))
assert magic==b'EMJOINT1'
rd=np.dtype([('mass','<f4'),('ratio','<f4'),('cut','<f4'),('excitation','<f4'),('e0','<f4'),('spin','<f4'),('sf','<f4'),('fr','<f4'),('ll','<f4'),('ion','<f4'),('ff','<f4'),('mm2','<f4'),('low','<f4'),('peak','<f4'),('off','<u4'),('count','<u4'),('offsets','<u4',(4,)),('counts','<u4',(4,)),('z','<u4'),('a','<u4')]);assert rd.itemsize==104
records=np.memmap(p,dtype=rd,mode='r',offset=32+8*(nm+ns),shape=(nr,));nodes=np.memmap(p,dtype='<f4',mode='r',offset=32+8*(nm+ns)+104*nr,shape=(nn,13))
x,w=np.polynomial.legendre.leggauss(64);errors=[];details=[];negative=0
for idx,r in enumerate(records):
 n=np.asarray(nodes[r['off']:r['off']+r['count']],dtype='f8');delta=n[:,1]-n[:,2];negative+=int(np.count_nonzero(delta< -1e-6*np.maximum(n[:,1],1)))
 # All material/projectile records, spread through the transported energy range.
 mask=(n[:,0]>=.01)&(n[:,0]<=6000/r['a'])&(delta>1e-6)&(n[:,5]>0)
 valid=np.flatnonzero(mask)
 if not len(valid):continue
 ii=valid[np.linspace(0,len(valid)-1,min(40,len(valid))).astype(int)];k=n[ii,0]*r['a'];mass=float(r['mass']);u=k/mass;ratio=.51099891/mass;tm=2*.51099891*u*(u+2)/(1+2*(u+1)*ratio+ratio*ratio);cut=float(r['cut']);keep=tm>cut*1.0001;ii=ii[keep];tm=tm[keep];k=k[keep];u=u[keep]
 if not len(ii):continue
 beta=u*(u+2)/(u+1)**2;total2=(k+mass)**2;spin=float(r['spin']);den=1/cut-1/tm-beta/tm*np.log(tm/cut)+(0.5*(tm-cut)/total2 if spin>0 else 0)
 half=.5*np.log(tm/cut);mid=np.log(cut)+half;e=np.exp(mid[:,None]+half[:,None]*x)
 f1=.5*e*e/total2[:,None] if spin>0 else np.zeros_like(e);f=1-beta[:,None]*e/tm[:,None]+f1;ff=float(r['ff'])*e;acc=1/(1+ff)**2
 if spin>0:
  x2=.5*.51099891*e/(mass*mass);acc*=1+float(r['mm2'])*(x2-f1/f)/(1+x2)
 acc=np.clip(acc,0,1)
 moment=np.sum(w[None,:]*f*acc*half[:,None],axis=1)/den
 predicted=n[ii,5]*moment;actual=delta[ii];rel=(actual-predicted)/np.maximum(predicted,1e-20)
 errors.extend(rel.tolist())
 if r['z']==6 and r['a']==12:details.append(dict(record=int(idx),z=int(r['z']),a=int(r['a']),max_abs_relative=float(abs(rel).max()),median_ratio=float(np.median(actual/predicted))))
result=dict(records=nr,tested_nodes=len(errors),negative_difference_nodes=negative,relative_difference_percentiles=np.percentile(np.abs(errors)*100,[50,90,99,100]).tolist(),c12_records=details,comparison='(full_dedx-restricted_dedx) versus GetLambda * accepted delta-spectrum first moment at material density nodes; quadrature includes spin/form-factor veto')
Path(__file__).with_name('mean_data_check.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='c12_records'})
