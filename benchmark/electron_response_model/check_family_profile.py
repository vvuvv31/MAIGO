#!/usr/bin/env python3
"""Independently check streaming full-family statistics against raw graphs."""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import extract, DTYPE
from family_reference import read_summary
from voxel_profile import chord_profile
from prototype import save, sha


def full_raw(source):
    charged=extract(source)  # Complete existing genealogy/continuity/energy audits.
    raw=np.memmap(source['path'],DTYPE,mode='r')
    order=np.lexsort((raw['step'],raw['track'],raw['event']))
    change=np.r_[True,(raw['track'][order[1:]]!=raw['track'][order[:-1]]) |
                 (raw['event'][order[1:]]!=raw['event'][order[:-1]])]
    starts=np.flatnonzero(change);first=order[starts]
    groups=np.empty(len(raw),np.int32);groups[order]=np.cumsum(change)-1
    lookup={(int(raw['event'][i]),int(raw['track'][i])):g for g,i in enumerate(first)}
    parent=np.array([lookup[(int(raw['event'][i]),int(raw['parent'][i]))] if raw['parent'][i] else -1 for i in first])
    photon=np.full(len(first),-1,np.int8)
    for g in range(len(first)):
        chain=[];j=g
        while j>=0 and photon[j]<0:
            assert j not in chain
            chain.append(j);j=parent[j]
        flag=bool(photon[j]) if j>=0 else False
        for k in reversed(chain):
            flag=flag or raw['pdg'][first[k]]==22;photon[k]=flag
    material=raw['pre_material'][first[parent<0]][0]
    inside=raw['pre_material']==material
    rows=np.flatnonzero(inside & (photon[groups]!=0) & (raw['dep']>0))
    pre=raw['pre'][rows,0].copy();post=raw['post'][rows,0].copy()
    pre[raw['pdg'][rows]==22]=post[raw['pdg'][rows]==22]
    pp=chord_profile(pre,post,raw['dep'][rows],raw['event'][rows],source['histories'])
    cp=chord_profile(charged['pre'][:,0],charged['post'][:,0],charged['weight'],charged['event'],source['histories'])
    outside=order[~inside[order]];_,indices=np.unique(groups[outside],return_index=True)
    exits=outside[indices];assert np.all(photon[groups[exits]])
    escaped=np.bincount(raw['event'][exits],weights=raw['energy'][exits],minlength=source['histories'])
    assert np.max(abs(pp.sum(axis=1)+escaped-charged['ledger'][:,2]))<1e-8
    return cp,pp,escaped,charged['audit'],raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();checks=[]
    sources=json.loads((a.input/'manifest.json').read_text())
    originals=json.loads((a.baseline/'manifest.json').read_text())
    # Includes exact points, boundary ownership, and multi-period chords.
    x=np.array([0.,-.2,1.7,-5.4]);y=np.array([0.,.2,1.9,5.4])
    test=chord_profile(x,y,np.ones(4),np.arange(4),4)
    expected=np.zeros((4,36));expected[0,18]=1;expected[1,16:20]=.25;expected[2,[35,0]]=.5;expected[3]=1/36
    assert np.max(abs(test-expected))<1e-13
    for s,old in zip(sources,originals):
        assert s['seed']==old['seed'] and s['histories']==old['histories']
        cp,pp,escape,audit,raw=full_raw(s)
        native=read_summary(a.input/s['name']/'family.phsp',s['histories'])
        differences={'charged':float(np.max(abs(cp-native['profile']))),
                     'photon':float(np.max(abs(pp-native['photon_profile']))),
                     'escape':float(np.max(abs(escape-native['escape'])))}
        assert max(differences.values())<1e-12,differences
        assert int(native['steps'].sum())==len(raw)
        assert sha(Path(old['path']))==old['sha256']
        baseline=np.memmap(old['path'],DTYPE,mode='r')
        def canonical(r):return r[np.lexsort((r['step'],r['track'],r['event'],r['run']))].view(np.uint8)
        identical=np.array_equal(canonical(raw),canonical(baseline));assert identical
        checks.append(dict(name=s['name'],histories=s['histories'],rows=len(raw),differences_MeV=differences,
                           all_transport_bytes_identical=identical,raw_audit=audit))
    save(a.out,dict(status='PASS_FULL_FAMILY_SCORER_RAW_AND_BASELINE',checks=checks,
         analytic_chord_cases=4,script_sha256=sha(Path(__file__))))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
