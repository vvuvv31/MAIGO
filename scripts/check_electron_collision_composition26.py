#!/usr/bin/env python3
"""Native complete Penelope event check including prompt and atomic children.

Checks complete-event energy-weighted correlations and deposited energy.
This is a component check, not GPU dose acceptance.
"""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE
ROOT=Path(__file__).resolve().parents[1]
OSC=np.dtype([(s,'<f4') for s in ['strength','ionization','resonance','recoil_cut','binding']]+[('z','<u4'),('shell_flag','<u4')])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--functions',type=Path,required=True)
    p.add_argument('--atomic',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=100000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for d in sorted(a.native.glob('composition*')):
        functions=a.functions/d.name
        if not functions.exists():continue
        osc=np.atleast_1d(np.genfromtxt(functions/'penelope_oscillators.csv',delimiter=',',names=True));o=np.empty(len(osc),OSC)
        for target,source in [('strength','strength'),('ionization','ionization_MeV'),('resonance','resonance_MeV'),('recoil_cut','recoil_cut_MeV'),('binding','binding_MeV'),('z','z'),('shell_flag','shell_flag')]:o[target]=osc[source]
        w=np.genfromtxt(functions/'penelope_weights.csv',delimiter=',',names=True);delta=w[w['oscillator']==0]['density_correction'][:34].astype('<f4')
        audit=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
        for r in audit:
            if r['model']=='PenIoni':cases.append(dict(section=int(d.name[-2:]),energy=float(r['energy_MeV']),cut=float(r['electron_cut_MeV']),photon_cut=float(r['photon_cut_MeV']),cell=int(r['cell']),directory=d,oscillators=o,delta=delta))
    if not cases:raise ValueError('No native Penelope conditions')
    with (a.out/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<I',len(cases)))
        for c in cases:
            f.write(struct.pack('<fffIII',c['energy'],c['cut'],c['photon_cut'],a.samples,len(c['oscillators']),len(c['delta'])));f.write(c['oscillators'].tobytes());f.write(c['delta'].tobytes())
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-DCARBON_DOSE_FP32=1','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_collision_composition26.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'sampled.bin'),str(a.atomic)],check=True)
    samples=np.memmap(a.out/'sampled.bin','<f4',mode='r',shape=(len(cases),a.samples,16,3));rows=[];current=None
    for i,c in enumerate(cases):
        if current!=c['directory']:
            current=c['directory'];raw=np.memmap(current/'conditional.phsp',DTYPE,mode='r')
        r=raw[raw['run']==c['cell']];events=r[r['kind']==0];children=r[r['kind']==1]
        # Native primary and deposit share the parent event; children are
        # grouped before computing standard errors, preserving covariance.
        ids=events['event'];positions=np.searchsorted(ids,children['event'])
        if np.any(ids[positions]!=children['event']):raise ValueError('Child event identity')
        model=samples[i].astype(float);energy=c['energy'];moments={}
        def native_sum(values,child_values):
            out=values.copy();np.add.at(out,positions,child_values);return out
        reference=[native_sum(events['remaining']**2/energy,children['remaining']**2/energy),
                   native_sum(events['remaining']*events['post'][:,2]/energy,children['remaining']*children['post'][:,2]/energy),
                   events['deposit']/energy,
                   native_sum(np.zeros(len(events)),children['remaining']/energy*(children['pdg']==22))]
        candidate=[np.sum(model[:,:,1]**2/energy*(model[:,:,0]!=0),axis=1),
                   np.sum(model[:,:,1]*model[:,:,2]/energy*(model[:,:,0]!=0),axis=1),
                   np.sum(model[:,:,1]/energy*(model[:,:,0]==0),axis=1),
                   np.sum(model[:,:,1]/energy*(model[:,:,0]==22),axis=1)]
        for name,x,y in zip(['energy','cosine','local','photon'],reference,candidate):
            diff=float(y.mean()-x.mean());se=np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
            moments[name]=dict(native=float(x.mean()),candidate=float(y.mean()),difference=diff,z=float(diff/se) if se else (0. if abs(diff)<1e-7 else float('inf')))
        thresholds=np.unique(np.quantile(np.r_[events['remaining'],children['remaining']],np.linspace(.05,.95,13)))
        maximum=0.;max_z=0.
        for t in thresholds:
            threshold=float(np.float32(t))+4*np.finfo(np.float32).eps*energy
            for mu in np.linspace(-1,1,17):
                x=native_sum(events['remaining']/energy*((events['remaining']<=threshold)&(events['post'][:,2]<=mu)),
                             children['remaining']/energy*((children['remaining']<=threshold)&(children['post'][:,2]<=mu)))
                y=np.sum(model[:,:,1]/energy*((model[:,:,1]<=threshold)&(model[:,:,2]<=mu)&(model[:,:,0]!=0)),axis=1)
                diff=float(y.mean()-x.mean());se=np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
                maximum=max(maximum,abs(diff));max_z=max(max_z,abs(diff)/se if se else (0. if abs(diff)<1e-7 else float('inf')))
        row={k:v for k,v in c.items() if k not in ['directory','oscillators','delta']}
        rows.append(dict(row,moments=moments,maximum_joint_cdf=maximum,maximum_joint_z=max_z))
    worst=max(abs(r['moments'][f]['z']) for r in rows for f in ['energy','cosine','local','photon']);worst_joint=max(r['maximum_joint_z'] for r in rows)
    result=dict(status='FULL_PENELOPE_EVENT_COMPONENT_NOT_GPU_DOSE_ACCEPTANCE',cases=len(rows),maximum_moment_z=worst,maximum_joint_z=worst_joint,maximum_joint_cdf=max(r['maximum_joint_cdf'] for r in rows),component_pass=bool(worst<7 and worst_joint<8),rows=rows)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not result['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
