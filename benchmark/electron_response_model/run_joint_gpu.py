#!/usr/bin/env python3
"""Run and audit the FP32 research operator; never grants CT acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from prototype import save, sha
from voxel_profile import PEAK, VALLEY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--operator',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--histories',type=int,default=2000000)
    p.add_argument('--seed',type=int,default=100306701);a=p.parse_args()
    assert not a.out.exists()
    info=json.loads((a.operator/'manifest.json').read_text());binary=a.binary.resolve()
    assert sha(a.operator/'operator.bin')==info['payload_sha256']
    binary_hash=sha(binary)
    env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+
             '/usr/local/cuda/lib64:'+os.environ.get('LD_LIBRARY_PATH',''))
    cmd=[str(binary),str((a.operator/'operator.bin').resolve()),info['payload_sha256'],str(a.histories),str(a.seed),str(a.out.resolve())]
    with a.out.with_suffix('.log').open('x') as f:
        subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert sha(binary)==binary_hash
    run=json.loads((a.out/'run.json').read_text())
    counts=np.fromfile(a.out/'counts.u64','<u8').reshape(20,6)
    blocks=np.fromfile(a.out/'profile.f32','<f4').reshape(20,2,36).astype(float)
    assert np.all(counts.sum(axis=1)==a.histories//20)
    assert not np.any(counts[:,3:]) and run['scorer']=='FP32'
    assert np.array_equal(blocks.sum(axis=(1,2)),counts[:,0]+counts[:,2])
    blocks/=a.histories//20;summaries={}
    for name,values in [('charged',blocks[:,0]),('photon',blocks[:,1]),('full',blocks.sum(axis=1))]:
        rois=np.column_stack([values[:,PEAK].sum(axis=1),values[:,VALLEY].sum(axis=1)])
        summaries[name]=dict(mean=values.mean(axis=0).tolist(),sem=(values.std(axis=0,ddof=1)/np.sqrt(20)).tolist(),
                            roi_mean=rois.mean(axis=0).tolist(),roi_sem=(rois.std(axis=0,ddof=1)/np.sqrt(20)).tolist())
    save(a.out/'report.json',dict(run,summaries=summaries,counts=counts.sum(axis=0).tolist(),packet_energy_closure=True,
         binary_sha256=binary_hash,operator_manifest_sha256=sha(a.operator/'manifest.json'),command=cmd,
         scope='Homogeneous mono-electron diagnostic only; not proton primary/s or CT gamma acceptance',
         script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(kernel_seconds=run['kernel_seconds'],histories=a.histories,counts=counts.sum(axis=0).tolist())),flush=True)


if __name__=='__main__':main()
