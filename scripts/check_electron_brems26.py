#!/usr/bin/env python3
"""Independent native correlated Seltzer-Berger spectrum and recoil checks.

This qualifies kinematics only, not rates, a runtime package or GPU dose.
Uncertainties use complete hard events, preserving branch covariance.
"""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE
ROOT=Path(__file__).resolve().parents[1]

def weighted_event_moments(pair,energy):
    return np.sum(pair[:,:,0]/energy*pair,axis=1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-samples',action='store_true');p.add_argument('--spectrum',type=Path,required=True);p.add_argument('--samples',type=int,default=100000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=a.reuse_samples);cases=[];recoil=[]
    spectra={}
    with a.spectrum.open('rb') as f:
        magic,version,count=struct.unpack('<8sII',f.read(16))
        if (magic,version,count)!=(b'MAISB026',2,26):raise ValueError('26 spectrum schema')
        for i in range(count):
            section,ne,nx,nz=struct.unpack('<4I',f.read(16))
            spectra[section]=(ne,nx,nz,f.read(4*(ne+nx+ne*nx+nz+nz*ne*nx)))
        if f.read(1):raise ValueError('Trailing spectrum bytes')
    for d in sorted(a.native.glob('composition*')):
        audit=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
        raw=np.memmap(d/'conditional.phsp',DTYPE,mode='r')
        _,indices=np.unique(raw['run'],return_index=True)
        ranges=dict(zip(raw['run'][indices],zip(indices,np.r_[indices[1:],len(raw)])))
        for row in audit:
            if row['kind']!=0:continue
            lo,hi=ranges[int(row['cell'])];r=raw[lo:hi];events=r[r['kind']==0];children=r[r['kind']==1]
            energy=float(row['energy_MeV'])
            if row['model']!='eBremSB':continue
            if len(events)!=len(children) or np.any(children['pdg']!=22) or not np.array_equal(events['event'],children['event']):
                raise ValueError('Brems must have exactly one correlated photon child')
            pair=np.stack([np.c_[events['remaining'],events['post'][:,2]],np.c_[children['remaining'],children['post'][:,2]]],axis=1)
            cases.append(dict(section=int(d.name[-2:]),energy=energy,cut=float(row['photon_cut_MeV']),cell=int(row['cell']),pair=pair))
    if not cases:raise ValueError('No independently sampled native brems cases')
    if not a.reuse_samples:
        with (a.out/'fixture.bin').open('xb') as f:
            f.write(struct.pack('<I',len(cases)))
            for c in cases:
                ne,nx,nz,table=spectra[c['section']]
                f.write(struct.pack('<ffIIII',c['energy'],c['cut'],a.samples,ne,nx,nz));f.write(table)
        binary=a.out/'check'
        subprocess.run(['g++','-O3','-std=c++20','-DCARBON_DOSE_FP32=1','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_brems26.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'sampled.bin')],check=True)
    samples=np.memmap(a.out/'sampled.bin','<f4',mode='r',shape=(len(cases),a.samples,2,2));rows=[]
    for i,c in enumerate(cases):
        native=c.pop('pair');model=samples[i].astype(float);energy=c['energy'];moments={}
        for field,name in enumerate(['energy','cosine']):
            values=[np.sum(pair[:,:,0]/energy*pair[:,:,field],axis=1) for pair in [native,model]]
            means=[float(x.mean()) for x in values];se=np.sqrt(sum(x.var(ddof=1)/len(x) for x in values))
            moments[name]=dict(native=means[0],candidate=means[1],difference=means[1]-means[0],z=float((means[1]-means[0])/se) if se else 0.)
        thresholds=np.unique(np.quantile(native[:,:,0],np.linspace(.05,.95,21)))
        angles=np.linspace(0,1,25)
        def cdf_stats(pair):
            # Complete-event second moment retains both branch covariances.
            nt,na=len(thresholds),len(angles)
            def cumulative(e,mu,w):
                hist=np.zeros((nt+1,na+1))
                ti=np.searchsorted(thresholds,e,side='left');ai=np.searchsorted(angles,mu,side='left')
                np.add.at(hist,(ti,ai),w)
                return hist.cumsum(0).cumsum(1)[:nt,:na]/len(pair)
            w=pair[:,:,0]/energy
            mean=sum(cumulative(pair[:,k,0],pair[:,k,1],w[:,k]) for k in range(2))
            second=sum(cumulative(pair[:,k,0],pair[:,k,1],w[:,k]**2) for k in range(2))
            second+=cumulative(np.max(pair[:,:,0],axis=1),np.max(pair[:,:,1],axis=1),2*w[:,0]*w[:,1])
            variance=np.maximum(0.,second-mean**2)/(len(pair)-1)
            return mean,variance
        nm,nv=cdf_stats(native);cm,cv=cdf_stats(model)
        difference=abs(cm-nm);se=np.sqrt(nv+cv)
        maximum=float(difference.max());max_z=float(np.max(np.divide(difference,se,out=np.zeros_like(se),where=se>0)))
        rows.append(dict(c,moments=moments,joint_cdf_difference=maximum,joint_cdf_max_z=max_z,native_events=len(native),candidate_events=len(model)))
    worst=max(abs(r['moments'][f]['z']) for r in rows for f in ['energy','cosine']);worst_cdf=max(r['joint_cdf_max_z'] for r in rows)
    result=dict(status='NATIVE_KINEMATICS_ONLY_NOT_RATE_PACKAGE_OR_GPU_ACCEPTANCE',cases=len(rows),maximum_moment_z=worst,maximum_joint_z=worst_cdf,maximum_joint_cdf=max(r['joint_cdf_difference'] for r in rows),
                component_pass=bool(worst<7 and worst_cdf<8 and all(r['maximum_cosine_residual']<1e-7 for r in recoil)),rows=rows,brems_recoil=recoil)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','brems_recoil']},indent=2),flush=True)
    if not result['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
