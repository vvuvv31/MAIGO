#!/usr/bin/env python3
"""FP32 rate-function queries against explicit native splines/collision probes."""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes,raw_rate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    data=np.load(a.model/'rates26.npz');nodes=collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged');cuts={}
    for line in (ROOT/'data/archive/electron_density_nodes_20261006/shared/charged/index.txt').read_text().splitlines()[1:]:
        s,r,file=line.split()
        with (ROOT/'data/archive/electron_density_nodes_20261006/shared/charged'/file).open('rb') as f:h=struct.unpack('<8sIid4IQ4d',f.read(80))
        cuts[int(s),float(r)]=h[9:11]
    native=json.loads((a.native/'result.json').read_text());observations=[r for result in native['results'] for r in result['rows'] if r['kind']=='charged']
    expected=[];groups=[];offset=0
    with (a.out/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<I',52))
        for section in range(26):
            for proc in range(2):
                prefix=f'c{section:02d}_p{proc}_';edges=data[prefix+'edges'];bounds=data[prefix+'cut_piece_bounds'];coeff=data[prefix+'coeff'];slope=data[prefix+'density_slope'];low,high=data[prefix+'density_domain'];majorant=data[prefix+'coefficient_majorant'][0]
                probes=edges[:-1,None].astype(float)+np.diff(edges.astype(float))[:,None]*np.array([.01,.125,.375,.5,.625,.875,.99]);queries=[];ref=[]
                for n in nodes:
                    if n['section']!=section:continue
                    valid=(probes>=n['rates'][proc][0,0])&(probes<=n['rates'][proc][-1,1]);e=probes[valid].astype('<f4');c=cuts[section,n['density']][proc]
                    queries.append(np.column_stack([np.full(len(e),n['density']),np.full(len(e),c),e]).astype('<f4'));ref.extend(np.maximum(0.,raw_rate(n['rates'][proc],e.astype(float)))/n['density'])
                source_count=len(ref)
                for r in observations:
                    if r['section']==section and r['process']==proc:
                        queries.append(np.array([[r['density'],cuts[section,r['density']][proc],r['energy']]],'<f4'));ref.append(r['native_rate']/r['density'])
                query=np.concatenate(queries);f.write(struct.pack('<4I3f',len(edges)-1,len(bounds)-1,coeff.shape[1],len(query),low,high,majorant))
                for x in [edges,bounds,coeff,slope,query]:f.write(x.astype('<f4').tobytes())
                expected.extend(ref);groups.append(dict(section=section,process=proc,offset=offset,count=len(query),source_count=source_count,density=query[:,0].astype(float)));offset+=len(query)
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_cut_rates26.cpp'),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'actual.bin')],capture_output=True,text=True,check=True);(a.out/'runtime.json').write_text(result.stdout)
    actual=np.fromfile(a.out/'actual.bin','<f4').astype(float);expected=np.array(expected);rows=[]
    for g in groups:
        sl=slice(g['offset'],g['offset']+g['count']);ref=expected[sl];mass=actual[sl]/g.pop('density');peak=max(ref.max(),1e-30);error=abs(mass-ref)
        source=slice(0,g['source_count']);probe=slice(g['source_count'],g['count']);active=ref>peak*1e-4
        native_active=ref[probe]>0
        row=dict(g,peak_normalized_error=float(error.max()/peak),source_peak_normalized_error=float(error[source].max()/peak),native_probe_peak_normalized_error=float(error[probe].max()/peak) if len(error[probe]) else None,relative_active=float(np.max(error[active]/ref[active])) if active.any() else 0.,native_probe_relative=float(np.max(error[probe][native_active]/ref[probe][native_active])) if native_active.any() else None)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='FP32_SOURCE_AND_NATIVE_RATE_CHECK_NOT_CONDITIONAL_STATE_OR_GPU_DOSE_ACCEPTANCE',runtime=json.loads(result.stdout),maximum_peak_normalized_error=max(r['peak_normalized_error'] for r in rows),maximum_native_probe_peak_error=max(r['native_probe_peak_normalized_error'] for r in rows if r['native_probe_peak_normalized_error'] is not None),maximum_active_relative=max(r['relative_active'] for r in rows),maximum_native_probe_relative=max(r['native_probe_relative'] for r in rows if r['native_probe_relative'] is not None),active_rate_floor_of_peak=1e-4,component_pass=all(r['peak_normalized_error']<.0025 and r['relative_active']<.01 and (r['native_probe_relative'] is None or r['native_probe_relative']<.01) for r in rows),rows=rows)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if not report['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
