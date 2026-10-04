#!/usr/bin/env python3
"""Probe native GS/energy-loss primitives; not a charged boundary operator."""
import argparse, concurrent.futures, hashlib, json, os, re, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'out/electron_standalone_20261003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(task):
    out,node,binary=task;d=out/f'node{node:02d}';d.mkdir()
    source=BASE/'photon_conditional_bank512'/f'node{node:02d}'
    text=(source/'topas.txt').read_text().replace('PhotonConditionalDump','ElectronBoundaryProbe')
    (d/'topas.txt').write_text(text)
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    with (d/'topas.log').open('x') as log:r=subprocess.run([str(binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
    (d/'execution.json').write_text(json.dumps(dict(returncode=r.returncode,binary_sha256=sha(binary)))+'\n')
    if r.returncode or 'ELECTRON_BOUNDARY_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(f'native boundary probe failed: {d}')
    table=np.loadtxt(d/'boundary_parameters.csv',delimiter=',',skiprows=1)
    assert table.shape==(257,12) and np.isfinite(table).all() and np.all(table[:,:10]>=0)
    error=float(np.max(abs(table[:,1+1]/table[:,3]-1)))
    assert error<1e-10 and np.all(np.diff(table[:,0])>0)
    options=dict(line.split('=',1) for line in (d/'boundary_options.txt').read_text().splitlines())
    identity=json.loads((BASE/'birth_bank64'/f'node{node:02d}'/'complete.json').read_text())['material_identity']
    assert np.float32(float(options['density_g_cm3']))==np.float32(identity['runtime_density_g_cm3'])
    result=dict(node=node,material_identity=identity,options=options,maximum_transport_mfp_relative_error=error,
        negative_raw_interpolated_rates=int(np.count_nonzero(table[:,10:]<0)),
        binary_sha256=sha(binary),input_sha256=sha(d/'topas.txt'),table_sha256=sha(d/'boundary_parameters.csv'),
        hard_rate_cubics_sha256=sha(d/'hard_rate_cubics.csv') if (d/'hard_rate_cubics.csv').exists() else None,
        elastic_mfp_mm_at_400keV=float(np.interp(.4,table[:,0],table[:,1])),
        transport_mfp_mm_at_400keV=float(np.interp(.4,table[:,0],table[:,2])))
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',nargs='+',type=int,default=[57,12,32]);p.add_argument('--workers',type=int,default=3);p.add_argument('--binary',type=Path,default=BASE/'electron_boundary_native_v3/topas');a=p.parse_args()
    assert len(set(a.nodes))==len(a.nodes) and all(0<=n<82 for n in a.nodes)
    out=a.out.resolve();out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:r=list(pool.map(one,[(out,n,a.binary.resolve()) for n in a.nodes]))
    (out/'report.json').write_text(json.dumps(dict(status='NATIVE_BOUNDARY_PARAMETERS_ONLY_NOT_TRANSPORT_OR_DOSE_ACCEPTANCE',results=r),indent=2)+'\n')
if __name__=='__main__':main()
