#!/usr/bin/env python3
"""Research: retain native electron birth, resample its later continuing steps.

Photon trajectories remain exact. By default photon-descendant electrons also
remain exact; an explicit option tests the same electron continuation operator
for them. No photon or descendant energy is discarded. Both modes require
independent dose validation; neither is lossless graph compression.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from joint_family import load_model
from macro_model import DTYPE
from prototype import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resample-photon-electrons',action='store_true');a=p.parse_args()
    models,report=load_model(a.model);source=report['source'];raw_path=Path(source['path'])
    assert sha(raw_path)==source['sha256']
    # The reference physics has not been modified. This approximation is
    # motivated by isolated training with homogeneous displacement restored.
    raw=np.memmap(raw_path,DTYPE,mode='r');native=np.flatnonzero(raw['parent']!=0)
    g=models['graph'];assert len(native)==len(g['pdg'])
    photon=np.zeros(len(native),bool);active=np.flatnonzero(g['pdg']==22)
    while len(active):
        photon[active]=True
        following=np.unique(np.r_[g['next'][active],g['child'][active].ravel()]);following=following[following>=0]
        active=following[~photon[following]]
    resample=g['inside']&(g['pdg']==11)&(g['fraction'][:,1]>0)
    if not a.resample_photon_electrons:resample &= ~photon
    assert np.all(g['next'][resample]>=0)
    g['next'][resample]=-2
    reached=np.zeros(len(native),bool)
    active=np.unique(np.r_[models['continuation']['child'].ravel(),models['entry']['child'].ravel()])
    active=active[active>=0]
    while len(active):
        reached[active]=True
        following=np.unique(np.r_[g['next'][active],g['child'][active].ravel()]);following=following[following>=0]
        active=following[~reached[following]]
    selected=np.flatnonzero(reached);remap=np.full(len(native),-1,np.int32);remap[selected]=np.arange(len(selected))
    g={k:v[selected].copy() for k,v in g.items()}
    g['postdir']=raw['post_direction'][native[selected]].copy()
    for key in ['next','child']:
        r=g[key];g[key]=np.where(r>=0,remap[np.maximum(r,0)],r)
    for key in ['continuation','entry']:
        r=models[key]['child'];models[key]['child']=np.where(r>=0,remap[np.maximum(r,0)],r)
    models['graph']=g;a.out.mkdir(parents=True,exist_ok=False);hashes={}
    for name,m in models.items():
        path=a.out/(name+'.npz');np.savez(path,**m);hashes[name]=sha(path)
    save(a.out/'report.json',dict(report,status='NEW_SECONDARY_CONTINUATION_APPROXIMATION_NOT_VALIDATED',
        source_model_report_sha256=sha(a.model/'report.json'),hashes=hashes,
        original_graph_rows=len(native),graph_rows=len(selected),
        photon_graph_rows=int(photon[selected].sum()),photon_rows=int((g['pdg']==22).sum()),
        resampled_electron_nodes=int((g['next']==-2).sum()),resample_photon_electrons=a.resample_photon_electrons,
        payload_bytes=sum(v.nbytes for m in models.values() for v in m.values()),
        secondary_continuation_resampling=True,script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
