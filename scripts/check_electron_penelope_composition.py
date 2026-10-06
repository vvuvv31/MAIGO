#!/usr/bin/env python3
"""Native Penelope primary-state check; relaxation is NOT yet implemented.

Checks actual-cut shell selection and the correlated primary energy/angle,
including null events. Does not qualify atomic relaxation or a full package.
"""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE
ROOT=Path(__file__).resolve().parents[1]
OSC=np.dtype([(s,'<f4') for s in ['strength','ionization','resonance','recoil_cut','binding']]+[('z','<u4'),('shell_flag','<u4')])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--functions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=100000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for d in sorted(a.native.glob('composition*')):
        functions=a.functions/d.name
        if not functions.exists():continue
        osc=np.atleast_1d(np.genfromtxt(functions/'penelope_oscillators.csv',delimiter=',',names=True));o=np.empty(len(osc),OSC)
        for target,source in [('strength','strength'),('ionization','ionization_MeV'),('resonance','resonance_MeV'),('recoil_cut','recoil_cut_MeV'),('binding','binding_MeV'),('z','z'),('shell_flag','shell_flag')]:o[target]=osc[source]
        w=np.genfromtxt(functions/'penelope_weights.csv',delimiter=',',names=True);delta=w[w['oscillator']==0]['density_correction'][:34].astype('<f4')
        audit=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
        for r in audit:
            if r['model']=='PenIoni':cases.append(dict(section=int(d.name[-2:]),energy=float(r['energy_MeV']),cut=float(r['electron_cut_MeV']),cell=int(r['cell']),directory=d,oscillators=o,delta=delta))
    if not cases:raise ValueError('No native Penelope conditions')
    with (a.out/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<I',len(cases)))
        for c in cases:
            f.write(struct.pack('<ffIII',c['energy'],c['cut'],a.samples,len(c['oscillators']),len(c['delta'])));f.write(c['oscillators'].tobytes());f.write(c['delta'].tobytes())
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-DCARBON_DOSE_FP32=1','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_penelope_composition.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'sampled.bin')],check=True)
    samples=np.memmap(a.out/'sampled.bin','<f4',mode='r',shape=(len(cases),a.samples,7));rows=[];current=None
    for i,c in enumerate(cases):
        if current!=c['directory']:
            current=c['directory'];raw=np.memmap(current/'conditional.phsp',DTYPE,mode='r');raw=raw[raw['kind']==0]
        r=raw[raw['run']==c['cell']];native=np.c_[r['remaining'],r['post'][:,2]];model=samples[i,:,:2].astype(float)
        moments={}
        for field,name in enumerate(['energy','cosine']):
            values=[pair[:,field] for pair in [native,model]];means=[float(x.mean()) for x in values];se=np.sqrt(sum(x.var(ddof=1)/len(x) for x in values))
            moments[name]=dict(native=means[0],candidate=means[1],difference=means[1]-means[0],z=float((means[1]-means[0])/se) if se else 0.)
        thresholds=np.unique(np.quantile(native[:,0],np.linspace(.05,.95,21)));angles=np.linspace(-1,1,33);maximum=0.;max_z=0.
        for t in thresholds:
            for mu in angles:
                # Quantile tests on atom-valued endpoints are one-sided with
                # an FP32-sized tolerance, shared by native and candidate.
                tol=4*np.finfo(np.float32).eps*c['energy'];threshold=float(np.float32(t))+tol
                values=[(pair[:,0]<=threshold)&(pair[:,1]<=mu) for pair in [native,model]]
                diff=float(values[1].mean()-values[0].mean());se=np.sqrt(sum(x.var(ddof=1)/len(x) for x in values))
                maximum=max(maximum,abs(diff));max_z=max(max_z,abs(diff)/se if se else 0.)
        row={k:v for k,v in c.items() if k not in ['directory','oscillators','delta']}
        rows.append(dict(row,moments=moments,maximum_joint_cdf=maximum,maximum_joint_z=max_z,native_null_fraction=float(np.mean((r['remaining']==c['energy'])&(r['deposit']==0)&(r['children']==0))),candidate_null_fraction=float(samples[i,:,6].mean())))
    worst=max(abs(r['moments'][f]['z']) for r in rows for f in ['energy','cosine']);worst_joint=max(r['maximum_joint_z'] for r in rows)
    result=dict(status='PENELOPE_PRIMARY_KINEMATICS_NOT_RELAXATION_OR_FULL_PACKAGE',cases=len(rows),maximum_moment_z=worst,maximum_joint_z=worst_joint,maximum_joint_cdf=max(r['maximum_joint_cdf'] for r in rows),component_pass=bool(worst<7 and worst_joint<8),rows=rows)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not result['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
