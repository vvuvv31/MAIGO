#!/usr/bin/env python3
"""Joint energy/angle tests versus independent native fixed-energy births."""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE
ROOT=Path(__file__).resolve().parents[1]
SHELL=np.dtype([('cdf','<u8'),('binding','<f4'),('z','<u4')])
def shells(path):
    r=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True));w=r['electron_fraction']*r['shell_fraction'];w/=w.sum()
    out=np.empty(len(r),SHELL);cdf=np.cumsum(w.astype(np.longdouble));out['cdf']=np.minimum(cdf*np.longdouble(2)**64,np.longdouble(2)**64-1).astype('<u8');out['cdf'][-1]=np.iinfo(np.uint64).max
    out['binding']=r['binding_MeV'];out['z']=r['Z'];return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=100000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    spec=json.loads((a.native/'report.json').read_text());cases=[]
    with (a.out/'fixture.bin').open('xb') as f:
        count=sum(len(s['energies']) for s in spec['results']);f.write(struct.pack('<I',count))
        for s in spec['results']:
            sh=shells(Path(s['directory'])/'atomic_shells.csv')
            for i,e in enumerate(s['energies']):
                f.write(struct.pack('<IffII',s['projectile_a'],e,s['actual_cut'],len(sh),a.samples));f.write(sh.tobytes());cases.append((s,i,e))
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-DCARBON_DOSE_FP32=1','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_cut_birth26.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'sampled.bin')],check=True)
    model=np.memmap(a.out/'sampled.bin','<f4',mode='r',shape=(len(cases),a.samples,2));rows=[];cache=None;raw=None
    for index,(s,cell,e) in enumerate(cases):
        if cache!=s['directory']:
            cache=s['directory'];raw=np.fromfile(Path(cache)/'conditional.phsp',DTYPE);raw=raw[raw['kind']==1]
        r=raw[raw['run']==cell];t=r['remaining'];mu=r['post'][:,2];w=t/t.sum();se={}
        values=model[index].astype(float)
        for name,ref,actual in [('energy',t,values[:,0]),('cosine',mu,values[:,1])]:
            mean=float(w@ref);native_var=float(np.sum(w*w*(ref-mean)**2)*len(w)/(len(w)-1))
            candidate=float(actual.mean());error=np.sqrt(native_var+actual.var(ddof=1)/len(actual))
            se[name]=dict(native=mean,candidate=candidate,difference=candidate-mean,combined_se=float(error),z=float((candidate-mean)/error) if error else 0.)
        thresholds=np.quantile(t,np.linspace(.05,.95,13));angles=np.linspace(-1,1,25)
        joint=0.;cdf_z=0.
        for threshold in thresholds:
            for angle in angles:
                native=(t<=threshold)&(mu<=angle);candidate=(values[:,0]<=threshold)&(values[:,1]<=angle)
                nr=float(w@native);cr=float(candidate.mean());variance=float(np.sum(w*w*(native-nr)**2)*len(w)/(len(w)-1)+candidate.var(ddof=1)/len(candidate))
                joint=max(joint,abs(cr-nr));cdf_z=max(cdf_z,abs(cr-nr)/np.sqrt(variance) if variance else 0.)
        row=dict(section=s['section'],density=s['density'],projectile_a=s['projectile_a'],parent=e,cut=s['actual_cut'],moments=se,joint_cdf_absolute=joint,joint_cdf_max_z=cdf_z,native_events=len(t),model_samples=len(values))
        rows.append(row);print(json.dumps(row),flush=True)
    # Wide, predeclared screening threshold accounts for many correlated CDF
    # probes. Native consistency still does not imply GPU dose qualification.
    worst=max(max(abs(r['moments'][k]['z']) for k in ['energy','cosine']) for r in rows)
    result=dict(status='FIXED_EVENT_NATIVE_COMPONENT_NOT_GPU_DOSE_ACCEPTANCE',cases=len(rows),maximum_moment_z=worst,maximum_joint_cdf=max(r['joint_cdf_absolute'] for r in rows),maximum_joint_z=max(r['joint_cdf_max_z'] for r in rows),component_pass=bool(worst<7 and max(r['joint_cdf_max_z'] for r in rows)<8),rows=rows)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
