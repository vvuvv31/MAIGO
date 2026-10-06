#!/usr/bin/env python3
"""Pack experimental26 actual-cut rate functions, excluding terminal states."""
import argparse,json,struct
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    arrays={};rows={}
    for model in a.models:
        with np.load(model/'rates26.npz') as f:arrays.update({k:f[k] for k in f.files})
        for r in json.loads((model/'report.json').read_text())['rows']:
            key=r['section'],r['process']
            if key in rows:raise ValueError('Duplicate composition')
            rows[key]=r
    if len(rows)!=52:raise ValueError('Incomplete26')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('xb') as f:
        f.write(struct.pack('<8sII',b'MAIERG26',1,26))
        for section in range(26):
            for process in [0,1]:
                prefix=f'c{section:02d}_p{process}_';bins=arrays[prefix+'bins'];bounds=arrays[prefix+'bounds'];r=rows[section,process]
                coeff=[arrays[prefix+f'coeff{i:03d}'] for i in range(len(bins))];degree=coeff[0].shape[0]-1
                if any(c.shape[0]!=degree+1 for c in coeff):raise ValueError('Polynomial degree')
                majorants=np.array([np.nextafter(np.float32(6*np.max(np.sum(abs(c.astype(float)),axis=0))*1.02),np.float32(np.inf)) for c in coeff],'<f4')
                c=np.concatenate([c.ravel() for c in coeff]);f.write(struct.pack('<5I3f',section,process,len(bins),degree,len(c),*r['cut_domain'],float(arrays[prefix+'majorant'][0])))
                for v in [bounds,bins,arrays[prefix+'intervals'],arrays[prefix+'offsets'],arrays[prefix+'pole'],majorants,c]:f.write(v.tobytes())
    print(json.dumps(dict(schema='MAIERG26',compositions=26,density_nodes=0,bytes=a.out.stat().st_size,status='EXPERIMENTAL_RATES_ONLY_REQUIRES_TERMINALS_AND_GPU_DOSE')))
if __name__=='__main__':main()
