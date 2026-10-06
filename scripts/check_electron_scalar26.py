#!/usr/bin/env python3
"""Check the FP32 C++ prototype at off-grid queries against native interpolation."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from build_electron_scalar26 import load_charged


def native(node,energy):
    values=node['mass'];grid=node['energy']
    # Native linear interpolation is in mfp, stopping and range.
    return np.array([1/np.interp(energy,grid,values[:,0]),
                     np.interp(energy,grid,values[:,1]),np.interp(energy,grid,values[:,2])])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('data/archive/electron_density_nodes_20261006/shared/charged'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--queries',type=int,default=1024)
    args=p.parse_args()
    if not 1<=args.queries<=1000000:p.error('Invalid query count')
    groups={s:[] for s in range(26)}
    for line in (args.source/'index.txt').read_text().splitlines()[1:]:
        section,density,relative=line.split()
        groups[int(section)].append(load_charged(args.source/relative))
    arrays=np.load(args.model,allow_pickle=False);rng=np.random.default_rng(20261005)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<8sI',b'ECS26T01',26))
        for section in range(26):
            prefix=f'c{section:02d}_';energy=arrays[prefix+'energy']
            scalar=arrays[prefix+'scalar_coeff'];cut=arrays[prefix+'cut_coeff']
            domain=arrays[prefix+'log_domain'];nodes=sorted(groups[section],key=lambda x:x['density'])
            rho=np.array([n['density'] for n in nodes])
            f.write(struct.pack('<5I2f',len(energy),len(scalar),len(cut),cut.shape[1],args.queries,*domain))
            for array in [energy,scalar,cut]:f.write(array.astype('<f4').tobytes())
            for _ in range(args.queries):
                # Interior log samples avoid mistaking FP32 endpoint rounding
                # for extrapolation; water singleton is tested at exactly1.
                density=float(np.float32(np.exp(np.log(rho[0])+rng.uniform(.0001,.9999)*np.log(rho[-1]/rho[0]))))
                kinetic=float(np.float32(np.exp(rng.uniform(np.log(energy[0]*1.0001),np.log(energy[-1]*.9999)))))
                hi=min(int(np.searchsorted(rho,density)),len(rho)-1);lo=max(0,hi-1)
                fraction=0 if hi==lo else (density-rho[lo])/(rho[hi]-rho[lo])
                left=native(nodes[lo],kinetic);right=native(nodes[hi],kinetic)
                mass=left+fraction*(right-left)
                cuts=nodes[lo]['cut']+fraction*(nodes[hi]['cut']-nodes[lo]['cut'])
                expected=np.r_[density*mass[0],density*mass[1],mass[2]/density,cuts]
                f.write(struct.pack('<2f5d',density,kinetic,*expected))
    root=Path(__file__).resolve().parents[1];binary=args.output/'check'
    subprocess.run(['g++','-O3','-std=c++20','-I'+str(root/'include'),
                    str(root/'scripts/check_electron_scalar26.cpp'),'-o',str(binary)],check=True)
    run=subprocess.run([str(binary),str(args.output/'fixture.bin')],capture_output=True,text=True)
    (args.output/'check.log').write_text(run.stdout+run.stderr)
    if run.stdout:
        result=json.loads(run.stdout);result['returncode']=run.returncode
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    if run.returncode:raise SystemExit(run.returncode)


if __name__=='__main__':main()
