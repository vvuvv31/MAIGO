#!/usr/bin/env python3
"""FP32 actual-cut native-grid functions versus native density/cut references.

Legacy material-density banks are explicit offline references only. This does
not load them in GPU transport or qualify terminal-state sampling or dose.
"""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes,raw_rate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--offgrid',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    data={};rows_meta=[]
    for model in a.models:
        with np.load(model/'rates26.npz') as f:
            overlap=data.keys()&f.files
            if overlap:raise ValueError('Duplicate composition functions')
            data.update({k:f[k] for k in f.files})
        rows_meta.extend(json.loads((model/'report.json').read_text())['rows'])
    if len(rows_meta)!=52 or len({(r['section'],r['process']) for r in rows_meta})!=52:raise ValueError('Need exactly26 compositions/two processes')
    lookup={(r['section'],r['process']):r for r in rows_meta};source=ROOT/'data/archive/electron_density_nodes_20261006/shared/charged';cuts={}
    for line in (source/'index.txt').read_text().splitlines()[1:]:
        s,rho,name=line.split()
        with (source/name).open('rb') as f:h=struct.unpack('<8sIid4IQ4d',f.read(80))
        cuts[int(s),float(rho)]=h[9:11]
    cases=[]
    for n in collision_nodes(source,'charged'):
        for proc in [0,1]:cases.append(dict(section=n['section'],density=n['density'],cut=cuts[n['section'],n['density']][proc],process=proc,native=n['rates'][proc],scope='native_source_node'))
    if a.offgrid:
        for d in sorted(a.offgrid.glob('c*_off*')):
            meta=json.loads((d/'report.json').read_text());native=np.loadtxt(d/'hard_rate_cubics.csv',delimiter=',',skiprows=1)
            for proc in [0,1]:cases.append(dict(section=meta['section'],density=meta['density'],cut=meta['cut'][proc],process=proc,native=native[native[:,0]==proc,1:7],scope='independent_native_offgrid'))
    groups=[];references=[];offset=0
    with (a.out/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<I',len(cases)))
        for c in cases:
            section,process=c['section'],c['process'];r=lookup[section,process];prefix=f'c{section:02d}_p{process}_'
            bounds=data[prefix+'bounds'];bins=data[prefix+'bins'];intervals=data[prefix+'intervals'];offsets=data[prefix+'offsets']
            coefficients=np.concatenate([data[prefix+f'coeff{i:03d}'].ravel() for i in range(len(bins))]);degree=data[prefix+'coeff000'].shape[0]-1
            cut=float(np.float32(c['cut']));factor=2 if process==0 else 1
            energy=np.unique(np.r_[np.geomspace(1e-5,.65,1024),factor*cut*np.array([.9999,1,1.0001,1.01,1.1])]).astype('<f4');energy=energy[energy<=np.float32(.65)]
            query=np.c_[np.full(len(energy),c['density']),np.full(len(energy),cut),energy].astype('<f4')
            f.write(struct.pack('<5I3f',len(bins),degree,process,len(coefficients),len(energy),*r['cut_domain'],float(data[prefix+'majorant'][0])))
            for values in [bounds,bins,intervals,offsets,data[prefix+'pole'],coefficients,query]:f.write(values.tobytes())
            expected=np.maximum(0.,raw_rate(c['native'],energy.astype(float)))
            groups.append(dict(**{k:v for k,v in c.items() if k!='native'},offset=offset,count=len(energy),energy=energy));references.extend(expected);offset+=len(energy)
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_native_grid_rates.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'actual.bin')],check=True)
    actual=np.fromfile(a.out/'actual.bin','<f4').astype(float);reference=np.array(references);rows=[]
    if len(actual)!=len(reference):raise ValueError('FP32 query completion')
    for g in groups:
        sl=slice(g['offset'],g['offset']+g['count']);expected=reference[sl];error=abs(actual[sl]-expected);peak=max(expected.max(),1e-30);active=expected>peak*1e-4
        rows.append(dict(**{k:v for k,v in g.items() if k!='energy'},peak_error=float(error.max()/peak),active_relative=float(np.max(error[active]/expected[active])) if active.any() else 0.,zero_support_peak_leak=float(actual[sl][expected==0].max()/peak) if np.any(expected==0) else 0.))
    result=dict(status='FP32_ACTUAL_CUT_FUNCTION_COMPONENT_NOT_OUTCOME_OR_GPU_DOSE',compositions=26,cases=len(rows),queries=len(actual),maximum_peak_error=max(r['peak_error'] for r in rows),maximum_active_relative=max(r['active_relative'] for r in rows),maximum_zero_support_peak_leak=max(r['zero_support_peak_leak'] for r in rows),component_pass=all(r['peak_error']<.0025 and r['active_relative']<.01 for r in rows),rows=rows)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not result['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
