"""26-composition analytical-density experiment; never changes active packages.

Only reference composition rows are used to predict other densities. Target EM
rows are consulted afterwards for the component comparison. Formula metadata
is independently computed from composition, density and the 0.05 mm range cut.
No target residual fit or dose calibration is performed.
"""
from pathlib import Path
import argparse, json, struct, time
import numpy as np

FIELDS = ['energy','full','restricted','stopping','range','lambda','factor',
          'correction','dispersion','universal_dispersion','q2','model','fluctuation']
EXPONENTS=np.array([0,1,1,1,-1,1,0,1,1,1,0,0,0])
RD=np.dtype([('f','<f4',(14,)),('u','<u4',(12,))])

def read(path):
    with path.open('rb') as f:h=struct.unpack('<8s6I',f.read(32))
    assert h[0]==b'EMJOINT1' and h[1]==1
    nm,ns,nr,nn,nv=h[2:]
    mats=np.memmap(path,mode='r',offset=32,dtype=[('s','<i4'),('d','<f4')],shape=(nm,))
    rec=np.memmap(path,mode='r',offset=32+8*(nm+ns),dtype=RD,shape=(nr,))
    nodes=np.memmap(path,mode='r',offset=32+8*(nm+ns)+104*nr,dtype='<f4',shape=(nn,13))
    vec=np.memmap(path,mode='r',offset=32+8*(nm+ns)+104*nr+52*nn,dtype='<f4',shape=(nv,6))
    return h,mats,rec,nodes,vec

def density_effect(x,meta):
    c,m,a,x0,x1,d0=meta[5:11]
    return np.where(x>=x1,2*np.log(10)*x-c,
           np.where(x>=x0,2*np.log(10)*x-c+a*np.maximum(x1-x,0)**m,
                    d0*10**(2*(x-x0))))

def kinematics(e,r):
    mass,spin=float(r['f'][0]),float(r['f'][5]);k=e*int(r['u'][-1]);u=k/mass
    bg2=u*(u+2);beta=bg2/(u+1)**2
    ratio=.51099891/mass
    tm=2*.51099891*bg2/(1+2*(u+1)*ratio+ratio*ratio)
    return beta,tm,(k+mass)**2,bg2,spin

def delta_integral(e,r,cut):
    beta,tm,total2,bg2,spin=kinematics(e,r);c=np.minimum(cut,tm)
    den=1/c-1/tm-beta/tm*np.log(tm/c)
    if spin>0:den+=.5*(tm-c)/total2
    return np.maximum(den,0)

def predict(base,r,mb,mt,method,clamp_stopping=True):
    ratio=mt[2]/mb[2];p=base.astype(float)*ratio**EXPONENTS
    if method=='linear':return p
    beta,tm,total2,bg2,spin=kinematics(base[:,0].astype(float),r)
    cb=np.minimum(mb[3],tm);ct=np.minimum(mt[3],tm)
    x=.5*np.log10(bg2);dd=density_effect(x,mt)-density_effect(x,mb)
    # G4 alpha tables use q2 normalized to alpha's nominal charge, whereas
    # generic-ion/proton tables store the absolute effective charge square.
    alpha=int(r['u'][-2])==2 and int(r['u'][-1])==4
    coeff=mt[4]*base[:,10]*(4 if alpha else 1)/beta
    # Same-composition shell/Barkas/effective-charge data remain in the base.
    change=coeff*(np.log(ct/cb)-beta*(ct-cb)/tm)
    if spin>0:change+=coeff*.25*(ct*ct-cb*cb)/total2
    density_change=-coeff*dd
    p[:,1]+=density_change
    p[:,2]+=change+density_change
    p[:,3]+=change+density_change
    # Born delta rate recalculated from the target cut, including zero support.
    p[:,5]=coeff*delta_integral(base[:,0].astype(float),r,mt[3])
    # IonFluc's material-only extra charge factor cancels between densities.
    vr=(tm/beta-.5*ct)/(tm/beta-.5*cb)
    p[:,8]*=vr;p[:,9]*=vr
    if clamp_stopping:p[:,[1,2,3]]=np.maximum(p[:,[1,2,3]],0)
    return p

def metrics(pred,ref,energy_min):
    rows=ref[:,0]>=energy_min;result={}
    for j in range(1,13):
        a,b=pred[rows,j],ref[rows,j].astype(float)
        active=abs(b)>max(float(np.max(abs(b)))*1e-10,1e-25)
        err=abs(a[active]-b[active])/abs(b[active])
        result[FIELDS[j]]=dict(max_relative=float(err.max()) if len(err) else 0.,
             p99_relative=float(np.quantile(err,.99)) if len(err) else 0.,
             zero_support_mismatches=int(np.count_nonzero((a==0)!=(b==0))))
    return result

def probe(source,metadata,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    h,mats,records,nodes,vectors=read(source);ns=h[3]
    meta=np.loadtxt(metadata);assert len(meta)==len(mats)
    bases={int(s):int(np.flatnonzero(mats['s']==s)[len(np.flatnonzero(mats['s']==s))//2]) for s in np.unique(mats['s'])}
    assert len(bases)==26
    cases=[]
    for mid in range(len(mats)):
        bid=bases[int(mats[mid]['s'])]
        if mid==bid:continue
        for ion in range(ns):
            rb=records[bid*ns+ion];rt=records[mid*ns+ion]
            ob,nb=map(int,rb['u'][:2]);ot,nt=map(int,rt['u'][:2]);assert nb==nt
            b=nodes[ob:ob+nb];target=nodes[ot:ot+nt]
            assert np.array_equal(b[:,0],target[:,0])
            for method in ['linear','cut_density']:
                p=predict(b,rb,meta[bid],meta[mid],method)
                for low in [0.,1.,10.,50.]:
                    cases.append(dict(material=mid,base_material=bid,section=int(mats[mid]['s']),
                         density=float(mats[mid]['d']),z=int(rt['u'][-2]),a=int(rt['u'][-1]),
                         method=method,minimum_MeVu=low,fields=metrics(p,target,low)))
        if mid%20==0:print(f'{mid}/{len(mats)} densities, {time.monotonic()-start:.1f}s',flush=True)
    summary={}
    for method in ['linear','cut_density']:
        summary[method]={}
        for low in [0.,1.,10.,50.]:
            subset=[c for c in cases if c['method']==method and c['minimum_MeVu']==low]
            fields={}
            for f in FIELDS[1:]:
                worst=max(subset,key=lambda c:c['fields'][f]['max_relative'])
                fields[f]=dict(max_relative=worst['fields'][f]['max_relative'],
                     worst={k:worst[k] for k in ['material','section','density','z','a']},
                     median_case_p99=float(np.median([c['fields'][f]['p99_relative'] for c in subset])),
                     zero_support_mismatches=sum(c['fields'][f]['zero_support_mismatches'] for c in subset))
            summary[method][str(low)]=fields
    report=dict(scope='component probe; range/spline and finite-step synthesis pending',
         source=str(source),formula_metadata=str(metadata),composition_bases=bases,
         base_response_tables=26,density_residual_tables=0,
         cut_max_relative_error=float(np.max(abs(meta[:,3]/records['f'][::ns,2]-1))),
         elapsed_seconds=time.monotonic()-start,summary=summary,cases=cases)
    (out/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();probe(a.source,a.metadata,a.output)
