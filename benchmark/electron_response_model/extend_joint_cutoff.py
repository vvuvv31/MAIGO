#!/usr/bin/env python3
"""Attach measured native 100 eV terminal responses below a sampled table.

This explicitly records the sub-table endpoint approximation: kinetic energy
is not clamped and packet energy is unchanged; terminal displacement is drawn
from native cutoff histories. Independent 50/100 eV controls bound its spatial
extent. It does not turn arbitrary missing coverage into local deposition.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import DTYPE,extract
from macro_compose import basis
from joint_family import load_model
from prototype import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--cutoff',type=Path,required=True);p.add_argument('--subcut',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    models,report=load_model(a.model);audits=[];training=None
    for folder in [a.subcut,a.cutoff]:
        sources=json.loads((folder/'manifest.json').read_text())
        assert len(sources)==2 and sources[0]['seed']!=sources[1]['seed']
        for source in sources:
            for k in ['table','density_g_cm3','production_cut_mm','binary_sha256']:
                assert source[k]==report['source'][k],k
            audit=extract(source)['audit'];r=np.memmap(source['path'],DTYPE,mode='r')
            assert len(r)==source['histories'] and np.all(r['pdg']==11) and np.all(r['parent']==0)
            assert np.all(r['step']==1) and np.all(r['remaining']==0) and np.all(r['dep']==r['energy'])
            extent=float(np.linalg.norm(r['post']-r['pre'],axis=1).max(initial=0))
            # Explicit geometry screen: 1e-4 of the 0.1 mm scoring bin.
            # This is NOT a dose-error guarantee; independent full profiles
            # and the number of actual endpoint draws remain mandatory.
            assert extent<.1*1e-4 and r['length'].max()<.1*1e-4
            audits.append(dict(audit,maximum_chord_mm=extent,source=source))
            if folder==a.cutoff and source['name']=='train':training=np.array(r)
    r=training;n=len(r);m=models['continuation'];energy=float(r['energy'][0])
    assert np.all(r['energy']==energy) and energy<m['edges'][0]
    u,v=basis(r['direction']);frame=np.stack([u,v,r['direction']],axis=1)
    values=dict(cdf=np.arange(1,n+1)/n,delta=np.einsum('nij,nj->ni',frame,r['post']-r['pre']),
                postdir=np.einsum('nij,nj->ni',frame,r['post_direction']),length=r['length'],
                source_direction=r['direction'],source_energy=r['energy'],
                child=np.full((n,2),-1,np.int32),childpdg=np.zeros((n,2),np.int32),childdir=np.zeros((n,2,3)),
                fraction=np.column_stack([np.ones(n),np.zeros((n,3))]),channel_cdf=np.ones((n,4)))
    for k in values:m[k]=np.concatenate([values[k],m[k]])
    m['edges']=np.r_[energy,m['edges']];m['offsets']=np.r_[0,m['offsets']+n];m['nearest']=np.r_[0,m['nearest']+1]
    a.out.mkdir(parents=True,exist_ok=False);hashes={}
    for name,model in models.items():
        path=a.out/(name+'.npz');np.savez(path,**model);hashes[name]=sha(path)
    save(a.out/'report.json',dict(report,status='NATIVE_CUTOFF_EXTENSION_NOT_CT_VALIDATED',hashes=hashes,
        cutoff_endpoint=dict(approximation='Measured 100 eV terminal displacement below measured continuation floor; no KE clamp or dose scaling',
                             source_model_report_sha256=sha(a.model/'report.json'),native_controls=audits,
                             anchor_energy_MeV=energy,maximum_control_chord_mm=max(x['maximum_chord_mm'] for x in audits)),
        payload_bytes=sum(v.nbytes for model in models.values() for v in model.values()),script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
