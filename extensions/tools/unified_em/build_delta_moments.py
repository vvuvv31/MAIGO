#!/usr/bin/env python3
"""Offline M1/M2 generation from archived native EM; compress the matching pair for runtime.
Requires NumPy. Existing valid output is verified; invalid output is never overwritten.
"""
import argparse,json,os,struct
from pathlib import Path
import numpy as np
from verify_delta_moments import ROOT,verify
from verify_unified_em_data import verify as verify_source
x,w=np.polynomial.legendre.leggauss(32)
rd=np.dtype([('mass','<f4'),('ratio','<f4'),('cut','<f4'),('excitation','<f4'),('e0','<f4'),('spin','<f4'),('sf','<f4'),('fr','<f4'),('ll','<f4'),('ion','<f4'),('ff','<f4'),('mm2','<f4'),('low','<f4'),('peak','<f4'),('off','<u4'),('count','<u4'),('offsets','<u4',(4,)),('counts','<u4',(4,)),('z','<u4'),('a','<u4')])
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

def build(source,output):
    verify_source(source)
    if output.exists():verify(output,source);return
    with source.open('rb') as f:magic,version,nm,ns,nr,nn,nv=struct.unpack('<8s6I',f.read(32))
    if magic!=b'EMJOINT1' or version!=1:raise ValueError('Delta generation requires native EMJOINT1; use archived source, then factor_density.py')
    records=np.memmap(source,dtype=rd,mode='r',offset=32+8*(nm+ns),shape=(nr,))
    nodes=np.memmap(source,dtype='<f4',mode='r',offset=32+8*(nm+ns)+104*nr,shape=(nn,13))
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name(output.name+f'.tmp.{os.getpid()}')
    try:
        with tmp.open('xb') as f:f.write(struct.pack('<8sII',b'EMDMOMT2',2,nn));f.truncate(16+8*nn)
        values=np.memmap(tmp,dtype='<f4',mode='r+',offset=16,shape=(nn,2));maxerr=0.
        for idx,r in enumerate(records):
            n=np.asarray(nodes[r['off']:r['off']+r['count']],dtype='f8')
            for moment in (1,2):
                v=mean_values(r,n,moment=moment)
                if not (np.isfinite(v).all() and (v>=0).all()):raise ValueError('Invalid derived moment')
                values[r['off']:r['off']+r['count'],moment-1]=v
                pick=np.linspace(0,len(n)-1,20).astype(int);ref=mean_values(r,n[pick],64,moment);mask=ref>1e-12
                if mask.any():maxerr=max(maxerr,float(np.max(abs(v[pick][mask]/ref[mask]-1))))
            if idx%500==0:print(f'{idx}/{nr} material-ion records',flush=True)
        values.flush();del values
        if maxerr>=1e-4:raise ValueError(f'Quadrature check failed: {maxerr}')
        meta=dict(schema='EMDMOMT2',version=2,nodes=nn,records=nr,bytes=16+8*nn,
                  source=str(source.resolve()),quadrature_order=32,check_order=64,
                  max_relative_quadrature_error=maxerr,units=['MeV/mm','MeV^2/mm'],
                  data_file=output.name,generator='extensions/tools/unified_em/build_delta_moments.py')
        verify(tmp,source,metadata=meta);os.replace(tmp,output)
        output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'Generated {output}; max quadrature discrepancy={maxerr:.3g}')
    finally:
        if tmp.exists():tmp.unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.source,a.output)
