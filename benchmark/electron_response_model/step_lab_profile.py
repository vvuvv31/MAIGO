#!/usr/bin/env python3
"""Score a frozen joint-step model in laboratory x for geometry diagnostics.

Convolve the translationally invariant candidate with the source slit exactly.
Do not rotate, smooth or refit the anisotropic voxel reference. The candidate
still has a photon sink; report that limitation instead of adding local dose.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from step_joint import transport
from prototype import save, sha
from voxel_profile import chord_profile, PEAK, VALLEY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--histories',type=int,default=2000000)
    p.add_argument('--seed',type=int,default=100303701)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert a.histories>=200 and a.histories%20==0
    report=json.loads((a.model/'report.json').read_text())
    assert sha(a.model/'model.npz')==report['model_sha256']
    with np.load(a.model/'model.npz') as saved:model={k:saved[k].copy() for k in saved.files}
    entry=None
    if report.get('native_entry_model_sha256'):
        assert sha(a.model/'entry_model.npz')==report['native_entry_model_sha256']
        with np.load(a.model/'entry_model.npz') as saved:entry={k:saved[k].copy() for k in saved.files}
    energy=report['source']['actual_energy_MeV'];rng=np.random.default_rng(a.seed)
    blocks=[];counts=np.zeros(3,np.int64);work=[];start=time.monotonic()
    for i in range(20):
        pos,status,stats=transport(model,energy,a.histories//20,rng,entry)
        x=pos[status==0,0]
        h=chord_profile(x-.25,x+.25,np.full(len(x),1/(a.histories//20)),np.zeros(len(x),np.int64),1)[0]
        assert abs(h.sum()-len(x)/(a.histories//20))<1e-10
        blocks.append(h);counts+=np.bincount(status,minlength=3);work.append(stats)
        print('block',i+1,time.monotonic()-start,flush=True)
    profiles=np.array(blocks)
    rois=np.column_stack([profiles[:,PEAK].sum(axis=1),profiles[:,VALLEY].sum(axis=1)])
    result=dict(status='CHARGED_MODEL_LAB_X_SCREEN',ct_acceptance=False,production_promoted=False,
                scope='Frozen homogeneous joint-step model; photon branch sink, no voxel geometry navigation',
                source_model_sha256=report['model_sha256'],histories=a.histories,energy_MeV=energy,seed=a.seed,
                entry_model_sha256=report.get('native_entry_model_sha256'),
                mean=profiles.mean(axis=0).tolist(),sem=(profiles.std(axis=0,ddof=1)/np.sqrt(20)).tolist(),
                roi_mean=rois.mean(axis=0).tolist(),roi_sem=(rois.std(axis=0,ddof=1)/np.sqrt(20)).tolist(),
                counts=dict(deposit=int(counts[0]),photon_sink=int(counts[2])),work=work,
                packet_energy_closure=bool(counts[0]+counts[2]==a.histories),seconds=time.monotonic()-start,
                script_sha256=sha(Path(__file__)),transport_sha256=sha(Path(__file__).with_name('step_joint.py')))
    save(a.out/'report.json',result);np.savez(a.out/'blocks.npz',profile=profiles)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
