"""Offline 26-base analytical-density generation for the existing GPU interface.

Extraction saves only 26 composition responses. Synthesis consumes that file
and formula-derived material metadata, never target EM responses. It expands
to the existing density registry solely to avoid changing GPU transport during
the existing runtime interface. No density residuals, target splines or cuts are kept.
"""
from pathlib import Path
import argparse,json,struct,time
import numpy as np
from probe_analytic_density import read,predict,RD
from build_delta_moments import mean_values,rd as DELTA_RD

def extract(source,output,lossless=False):
    h,m,r,n,v=read(source);ns=h[3];data={'materials':np.array(m),'species':np.array(np.memmap(source,mode='r',offset=32+8*h[2],dtype='<u4',shape=(ns,2)))}
    for s in np.unique(m['s']):
        ids=np.flatnonzero(m['s']==s);mid=int(ids[len(ids)//2]);data[f'meta_{s}']=np.array([mid,float(m[mid]['d'])])
        for ion in range(ns):
            rec=r[mid*ns+ion];tag=f'{s}_{ion}'
            data[f'r_{tag}']=np.array(rec)
            off,count=map(int,rec['u'][:2]);data[f'n_{tag}']=np.array(n[off:off+count])
            for k in range(4):
                off,count=int(rec['u'][2+k]),int(rec['u'][6+k]);data[f'v_{tag}_{k}']=np.array(v[off:off+count])
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    (np.savez_compressed if lossless else np.savez)(output,**data)
    print(f'26 composition bases: {output.stat().st_size} bytes',flush=True)

def material_inputs(bases,section_metadata,output):
    bank=np.load(bases);mats=bank['materials']
    sections=json.loads(section_metadata.read_text())['section_manifest']
    elements=['H','C','N','O','Mg','P','S','Cl','Ar','Ca','Na','K','Ti'];rows=[]
    for mid,m in enumerate(mats):
        s=int(m['s']);rho=float(m['d'])
        weights=[('H',.111894),('O',.888106)] if s==-1 else [(e,w) for e,w in zip(elements,sections[s]['element_weights']) if w>0]
        excitation=75 if s==-1 else sections[s]['mean_excitation_energy_eV']
        rows.append(f'{mid} {s} {rho:.17g} {excitation:.17g} {len(weights)} '+' '.join(f'{e} {w:.17g}' for e,w in weights))
    if output.exists():raise FileExistsError(output)
    output.write_text('\n'.join(rows)+'\n')

def eval_segments(segments,x,kind=0):
    s=segments.astype(float);x=np.asarray(x,dtype=float)
    idx=np.clip(np.searchsorted(s[:,0],x,side='right')-1,0,len(s)-1);row=s[idx]
    w=np.clip((x-row[...,0])/(row[...,1]-row[...,0]),0,1)
    a=w-1/3;b=w-2/3;c=w-1
    y=(-4.5*a*b*c*row[...,2]+13.5*w*b*c*row[...,3]-13.5*w*a*c*row[...,4]+4.5*w*a*b*row[...,5])
    f=np.maximum(x/s[0,0],0)
    return np.where(x<s[0,0],0 if kind==3 else s[0,2]*(f*f if kind==2 else np.sqrt(f)),y)

class G4Spline:
    """Geant4 11.3.2 G4PhysicsVector::ComputeSecDerivative1 / Interpolation.

    Preserve its actual recurrence, including endpoint arithmetic, rather than
    replacing it with a library's mathematically standard not-a-knot spline.
    Algorithm reference: Geant4 v11.3.2, G4PhysicsVector.cc and .icc;
    Geant4 Software License: https://geant4.web.cern.ch/license/LICENSE.html
    """
    def __init__(self,x,y):
        self.x=np.asarray(x,dtype=float);self.y=np.asarray(y,dtype=float)
        x,y=self.x,self.y;n=len(x)-1;s=np.zeros(n+1);u=np.zeros(n)
        assert n>=4 and np.all(np.diff(x)>0)
        u[1]=6*((y[2]-y[1])/(x[2]-x[1])-(y[1]-y[0])/(x[1]-x[0]))*(x[2]-x[1])/(x[2]-x[0])**2
        s[1]=(2*x[1]-x[0]-x[2])/(2*x[2]-x[0]-x[1])
        for i in range(2,n-1):
            sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);p=sig*s[i-1]+2;s[i]=(sig-1)/p
            u[i]=6*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1]/p
        sig=(x[n-1]-x[n-2])/(x[n]-x[n-2]);p=sig*s[n-3]+2
        u[n-1]=6*sig*((y[n]-y[n-1])/(x[n]-x[n-1])-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))/(x[n]-x[n-2])-(2*sig-1)*u[n-2]/p
        p=1+sig+(2*sig-1)*s[n-2];s[n-1]=u[n-1]/p
        for k in range(n-2,1,-1):s[k]*=s[k+1]-u[k]*(x[k+1]-x[k-1])/(x[k+1]-x[k])
        s[n]=(s[n-1]-(1-sig)*s[n-2])/sig
        sig=1-(x[2]-x[1])/(x[2]-x[0]);s[1]*=s[2]-u[1]/(1-sig);s[0]=(s[1]-sig*s[2])/(1-sig)
        self.second=s
    def __call__(self,t):
        x,y,s=self.x,self.y,self.second;t=np.clip(t,x[0],x[-1])
        i=np.clip(np.searchsorted(x,t,side='right')-1,0,len(x)-2);dl=x[i+1]-x[i];w=(t-x[i])/dl
        return y[i]+w*(y[i+1]-y[i])+w*(w-1)*((2-w)*s[i]+(1+w)*s[i+1])*dl*dl/6

def segments_from_spline(x,y):
    spline=G4Spline(x,y)
    sample=x[:-1,None]+np.diff(x)[:,None]*np.array([0,1/3,2/3,1])
    seg=np.column_stack([x[:-1],x[1:],spline(sample)])
    return seg,spline

def reference_record(r):
    q=r.copy();alpha=int(r['u'][-2])==2 and int(r['u'][-1])==4
    q['f'][0]=3727.379 if alpha else 938.27208816
    q['f'][1]=1.;q['f'][5]=.5 if int(r['u'][-2])==1 else 0.
    q['u'][-2]=2 if alpha else 1;q['u'][-1]=4 if alpha else 1
    return q

def rebuild(r,bn,bv,mb,mt):
    ratio=mt[2]/mb[2];rr=reference_record(r)
    # Rebuild the restricted DEDX spline at the base's original energy knots.
    x=np.r_[bv[0][:,0],bv[0][-1,1]].astype(float)
    old=np.r_[bv[0][:,2],bv[0][-1,5]].astype(float)
    pseudo=np.zeros((len(x),13));pseudo[:,0]=x/int(rr['u'][-1]);pseudo[:,10]=1
    p=predict(pseudo,rr,mb,mt,'cut_density',clamp_stopping=False)
    # Bragg's cut floor is lowestKinEnergy * mass/proton_mass, not the
    # material production cut alone. Alpha's floor is about 0.993 keV.
    floor=.00025*float(rr['f'][0])/938.272013
    mb_floor=mb.copy();mt_floor=mt.copy();mb_floor[3]=max(mb[3],floor);mt_floor[3]=max(mt[3],floor)
    plow=predict(pseudo,rr,mb_floor,mt_floor,'cut_density',clamp_stopping=False)
    low=x*(938.272013/float(rr['f'][0]))<=2
    p[low,2]=plow[low,2]
    stop=old*ratio+p[:,2]
    if not np.all(stop>0):raise ValueError('Nonpositive reconstructed DEDX')
    vs,ss=segments_from_spline(x,stop)
    # Follow G4LossTableBuilder: 100 midpoint samples per original interval.
    de=np.diff(x)/100
    sample=x[1:,None]-de[:,None]*(np.arange(100)[None,:]+.5)
    ranges=np.r_[2*x[0]/stop[0],2*x[0]/stop[0]+np.cumsum(np.sum(de[:,None]/ss(sample),axis=1))]
    if not np.all(np.diff(ranges)>0):raise ValueError('Nonmonotone reconstructed range')
    vr,sr=segments_from_spline(x,ranges);vi,si=segments_from_spline(ranges,x)
    # Rebuild G4's target-cut log lambda grid; the base grid cannot be scaled.
    mass=float(rr['f'][0]);cut=mt[3];c=.5*cut/.51099891
    if int(r['u'][-2])==1:
        ratio_e=.51099891/mass
        minimum=mass*(c*ratio_e+np.sqrt((1+c)*(1+c*ratio_e*ratio_e))-1)
    else:minimum=mass*(np.sqrt(1+c)-1)
    maximum=float(bv[3][-1,1]);minimum=max(minimum,float(bv[0][0,0]))
    bins=max(5,int(np.rint(20/np.log(10)*np.log(maximum/minimum))))
    lx=np.geomspace(minimum,maximum,bins+1)
    lp=np.zeros((len(lx),13));lp[:,0]=lx/int(rr['u'][-1]);lp[:,10]=1
    rates=predict(lp,rr,mb,mt,'cut_density')[:,5]
    low=lx*(938.272013/float(rr['f'][0]))<=2
    rates[low]=predict(lp,rr,mb_floor,mt_floor,'cut_density')[:,5][low]
    rates[0]=0
    vl,sl=segments_from_spline(lx,rates)
    # Store queried table fields, including the updated spline and range.
    pn=predict(bn,r,mb,mt,'cut_density')
    kin=bn[:,0].astype(float)*int(r['u'][-1]);raw=kin*float(r['f'][1]);factor=bn[:,6].astype(float)
    pn[:,3]=eval_segments(vs,raw)*factor
    pn[:,4]=eval_segments(vr,raw,1)/(factor*float(r['f'][1]))
    pn[:,5]=np.maximum(eval_segments(vl,raw,3)*factor,0)
    if int(r['u'][-2])>2:
        low=kin*(938.272013/float(r['f'][0]))<=2
        pn[low,7]=(bn[low,3].astype(float)+bn[low,7])*ratio-pn[low,3]
    rec=r.copy();rec['f'][2]=mt[3]
    rec['f'][13]=lx[np.argmax(rates)]/(int(r['u'][-1])*float(r['f'][1]))
    return rec,pn.astype('<f4'),[v.astype('<f4') for v in [vs,vr,vi,vl]]

def synthesize(bases,metadata,output,delta_output):
    start=time.monotonic()
    # Inflate each composition array once, not once per target density.
    with np.load(bases) as archive:bank={k:archive[k] for k in archive.files}
    mats=bank['materials'];species=bank['species'];ns=len(species)
    meta=np.loadtxt(metadata)
    if meta.shape!=(len(mats),13) or not np.isfinite(meta).all():raise ValueError('Invalid formula metadata shape/values')
    if not (np.array_equal(meta[:,0],np.arange(len(mats))) and np.array_equal(meta[:,1],mats['s']) and
            np.allclose(meta[:,2],mats['d'],rtol=1e-7,atol=0) and np.all(meta[:,3:5]>0)):
        raise ValueError('Formula metadata registry/cut/electron-density mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or delta_output.exists():raise FileExistsError('Use new candidate output paths')
    nr=len(mats)*ns;nn=sum(len(bank[f'n_{int(m["s"])}_{i}']) for m in mats for i in range(ns))
    # Temporary node payload is mmap-backed; preserve partial evidence on error.
    payload=output.with_suffix('.nodes.tmp');nodes=np.memmap(payload,mode='w+',dtype='<f4',shape=(nn,13))
    delta=np.memmap(delta_output,mode='w+',offset=16,dtype='<f4',shape=(nn,2))
    with delta_output.open('r+b') as f:f.write(struct.pack('<8sII',b'EMDMOMT2',2,nn))
    records=[];vectors=[];off=voff=0
    for mid,m in enumerate(mats):
        s=int(m['s']);bid=int(bank[f'meta_{s}'][0])
        for ion in range(ns):
            tag=f'{s}_{ion}';r=bank[f'r_{tag}'];bn=bank[f'n_{tag}'];bv=[bank[f'v_{tag}_{k}'] for k in range(4)]
            rec,n,v=rebuild(r,bn,bv,meta[bid],meta[mid])
            rec['u'][0]=off;rec['u'][1]=len(n)
            for k,a in enumerate(v):rec['u'][2+k]=voff;rec['u'][6+k]=len(a);voff+=len(a);vectors.append(a)
            records.append(rec);nodes[off:off+len(n)]=n
            dr=np.frombuffer(rec.tobytes(),dtype=DELTA_RD)[0]
            for moment in [1,2]:
                values=mean_values(dr,n.astype(float),moment=moment)
                if not (np.isfinite(values).all() and np.all(values>=0)):raise ValueError('Invalid reconstructed delta moments')
                delta[off:off+len(n),moment-1]=values
            off+=len(n)
        if mid%20==0:print(f'{mid}/{len(mats)} densities, {time.monotonic()-start:.1f}s',flush=True)
    nodes.flush();delta.flush()
    with output.open('xb') as f:
        f.write(struct.pack('<8s6I',b'EMJOINT1',1,len(mats),ns,nr,nn,voff))
        f.write(mats.tobytes());f.write(species.tobytes());f.write(np.array(records,dtype=RD).tobytes())
        # Stream without duplicating the expanded node matrix in RAM.
        for begin in range(0,nn,100000):f.write(nodes[begin:begin+100000].tobytes())
        for a in vectors:f.write(a.tobytes())
    del nodes,delta;payload.unlink()
    report=dict(scope='26 response bases + formula-derived cut/density corrections; offline-expanded runtime compatibility interface',
        base_file=str(bases),metadata=str(metadata),base_bytes=bases.stat().st_size,
        response_compositions=26,density_residual_tables=0,materials=len(mats),species=ns,nodes=nn,segments=voff,
        em_package=str(output),delta_package=str(delta_output),elapsed_seconds=time.monotonic()-start,
        no_target_response_input=True,default_promoted=False)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract');e.add_argument('--source',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--lossless',action='store_true',help='Lossless ZIP compression; no change to stored FP32 arrays')
    m=sub.add_parser('material-inputs');m.add_argument('--bases',type=Path,required=True)
    m.add_argument('--section-metadata',type=Path,default=Path('data/schneider/schneider_stopping_v1.metadata.json'));m.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('synthesize');s.add_argument('--bases',type=Path,default=Path('data/archive/ion_em_analytic_expanded_20261005/ion_em_26_composition_bases.npz'));s.add_argument('--metadata',type=Path,default=Path('data/archive/ion_em_analytic_expanded_20261005/ion_em_analytic_density_metadata.txt'))
    s.add_argument('--output',type=Path,required=True);s.add_argument('--delta-output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='extract':extract(a.source,a.output,a.lossless)
    elif a.command=='material-inputs':material_inputs(a.bases,a.section_metadata,a.output)
    else:synthesize(a.bases,a.metadata,a.output,a.delta_output)
