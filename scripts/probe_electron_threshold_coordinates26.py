#!/usr/bin/env python3
"""Independent native screen of threshold-aligned rate conditioning.

Offline source nodes are references, never a runtime directory. This screen
is not a package or a complete26-transport acceptance. Aligning the physical
E/cut argument is a materially different retry from fitting fixed-E mixtures.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

from build_electron_response26 import collision_nodes, raw_rate


def beta2(e):
    return e*(e+2*.51099895)/(e+.51099895)**2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('data/archive/electron_density_nodes_20261006/shared/charged'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    nodes=collision_nodes(a.source,'charged')
    for n in nodes:
        file=next(line.split()[2] for line in (a.source/'index.txt').read_text().splitlines()[1:]
                  if (int(line.split()[0]),float(line.split()[1]))==(n['section'],n['density']))
        with (a.source/file).open('rb') as f:n['cuts']=struct.unpack('<8sIid4IQ4d',f.read(80))[9:11]
    rows=[]
    for directory in sorted(a.native.glob('c*_off*')):
        report=json.loads((directory/'report.json').read_text())
        native=np.loadtxt(directory/'hard_rate_cubics.csv',delimiter=',',skiprows=1)
        group=sorted((n for n in nodes if n['section']==report['section']),key=lambda n:n['density'])
        densities=np.array([n['density'] for n in group])
        hi=int(np.searchsorted(densities,report['density']));lo=hi-1
        for process in [0,1]:
            cut=report['cut'][process];lcut,hcut=[group[i]['cuts'][process] for i in [lo,hi]]
            mix=(cut-lcut)/(hcut-lcut) if hcut!=lcut else (report['density']-densities[lo])/(densities[hi]-densities[lo])
            energy=np.geomspace(1e-5,.65,2048)
            values=[];valid=np.ones(len(energy),bool)
            for i,source_cut in [(lo,lcut),(hi,hcut)]:
                transformed=energy*source_cut/cut
                reference=group[i]['rates'][process]
                valid &= (transformed>=reference[0,0]) & (transformed<=reference[-1,1])
                value=np.maximum(0,raw_rate(reference,transformed))/group[i]['density']
                if process==0:value*=source_cut*beta2(transformed)/(cut*beta2(energy))
                values.append(value)
            candidate=((1-mix)*values[0]+mix*values[1])*report['density']
            if process==0:candidate[energy<=2*cut]=0
            expected=np.maximum(0,raw_rate(native[native[:,0]==process,1:7],energy))
            energy,candidate,expected=[x[valid] for x in [energy,candidate,expected]]
            peak=max(expected.max(),1e-30);active=expected>peak*1e-4
            error=abs(candidate-expected);index=int(np.argmax(error))
            rows.append(dict(section=report['section'],density=report['density'],process=process,
                             queries=len(energy),excluded_outside_source_domain=int((~valid).sum()),
                             peak_error=float(error.max()/peak),
                             active_relative=float(np.max(error[active]/expected[active])) if active.any() else 0,
                             zero_support_leak=float(candidate[expected==0].max()) if np.any(expected==0) else 0,
                             hotspot_energy=float(energy[index]),native=float(expected[index]),candidate=float(candidate[index])))
    result=dict(status='THRESHOLD_COORDINATE_SCREEN_NOT_RUNTIME_OR_FULL_DOMAIN_ACCEPTANCE',
                maximum_peak_error=max(r['peak_error'] for r in rows),
                maximum_active_relative=max(r['active_relative'] for r in rows),
                component_pass=all(r['peak_error']<.0025 and r['active_relative']<.01 for r in rows),rows=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
