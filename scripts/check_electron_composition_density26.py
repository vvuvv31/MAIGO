#!/usr/bin/env python3
"""Independent test of one composition model with explicit native cut.

Native lambda tables are rebuilt per cut; compare scaled composition rates to
fresh off-grid-density native process splines. This is a component screen,
not fitting/reusing density-node response states in a GPU package.
"""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
import numpy as np
from build_electron_response26 import raw_rate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--offgrid',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);tasks=[]
    for section in [0,1,15,16,17,18,19,23]:
        ref=a.reference/f'composition{section:02d}';r=json.loads((ref/'report.json').read_text());d=a.out/f'composition{section:02d}';d.mkdir()
        (d/'topas.txt').write_text((ref/'topas.txt').read_text());requests=[];cases=[]
        for directory in sorted(a.offgrid.glob(f'c{section:02d}_off*')):
            meta=json.loads((directory/'report.json').read_text());cut=meta['cut']
            # Disjoint from the first off-grid checker; also resolve onset.
            energy=np.unique(np.r_[np.geomspace(1e-5,.65,512),2*cut[0]*np.array([.9999,1,1.0001,1.01,1.05,1.1])]);energy=energy[energy<=.65].astype(np.float32)
            begin=len(requests)
            for process in [0,1]:
                for e in energy:requests.append([len(requests),0,process,float(e),*cut,0])
            cases.append(dict(**meta,begin=begin,energies=energy.tolist()))
        (d/'requests.txt').write_text(''.join(' '.join(f'{v:.17g}' if isinstance(v,float) else str(v) for v in row)+'\n' for row in requests))
        tasks.append((d,r['density'],cases))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    def run(task):
        d,density,cases=task;start=time.monotonic()
        with (d/'topas.log').open('x') as log:r=subprocess.run([str(a.binary.resolve()),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode or 'ELECTRON_COMPOSITION_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        actual=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'));rows=[]
        for case in cases:
            native=np.loadtxt(Path(case['directory'])/'hard_rate_cubics.csv',delimiter=',',skiprows=1);energy=np.array(case['energies'])
            for process in [0,1]:
                i=case['begin']+process*len(energy);raw_candidate=actual['table_rate_per_mm'][i:i+len(energy)]*case['density']/density
                # Both current transport and the native-reference evaluator
                # use max(raw spline lambda,0). Retain negative raw counts.
                candidate=np.maximum(0,raw_candidate)
                expected=np.maximum(0,raw_rate(native[native[:,0]==process,1:7],energy));peak=max(expected.max(),1e-30);active=expected>peak*1e-4
                error=abs(candidate-expected)
                rows.append(dict(section=int(d.name[-2:]),density=case['density'],process=process,queries=len(energy),
                                 peak_error=float(error.max()/peak),active_relative=float(np.max(error[active]/expected[active])) if active.any() else 0,
                                 negative_raw_queries=int((raw_candidate<0).sum()),
                                 zero_support_leak=float(candidate[expected==0].max()) if np.any(expected==0) else 0))
        print(json.dumps(dict(section=int(d.name[-2:]),seconds=time.monotonic()-start)),flush=True);return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=sum(list(pool.map(run,tasks)),[])
    report=dict(status='NATIVE_COMPOSITION_WITH_ACTUAL_CUT_OFFGRID_COMPONENT_NOT_GPU',cases=len(rows),queries=sum(r['queries'] for r in rows),
                maximum_peak_error=max(r['peak_error'] for r in rows),maximum_active_relative=max(r['active_relative'] for r in rows),
                component_pass=all(r['peak_error']<.0025 and r['active_relative']<.01 for r in rows),rows=rows)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
