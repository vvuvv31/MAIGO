#!/usr/bin/env python3
"""Check a complete rare history and scorer neutrality, including navigation."""
import argparse
from pathlib import Path
import numpy as np
from macro_model import DTYPE
from family_reference import read_summary
from voxel_profile import chord_profile
from prototype import sha, save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();r=a.work
    raw=np.fromfile(r/'rare_with/steps.phsp',DTYPE);old=np.fromfile(r/'rare_without/steps.phsp',DTYPE)
    assert np.array_equal(raw.view(np.uint8),old.view(np.uint8))
    assert len(raw)>0 and np.all(raw['event']==21602) and np.all(raw['version']==4)
    order=np.lexsort((raw['step'],raw['track']));raw=raw[order]
    starts=np.flatnonzero(np.r_[True,np.diff(raw['track'])!=0]);first=raw[starts]
    lookup={int(t):i for i,t in enumerate(first['track'])}
    parent=np.array([lookup[int(t)] if t else -1 for t in first['parent']])
    groups=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(raw)]))
    assert np.all(first['step']==1)
    photon=np.zeros(len(starts),bool)
    for g in range(len(starts)):
        chain=[];j=g
        while j>=0:
            assert j not in chain;chain.append(j);j=parent[j]
        photon[g]=np.any(first['pdg'][chain]==22)
    child=np.flatnonzero(parent>=0);child_first=starts[child]
    rows={(int(t),int(s)):i for i,(t,s) in enumerate(zip(raw['track'],raw['step']))}
    producing=np.array([rows[(int(raw['parent'][i]),int(raw['parent_step'][i]))] for i in child_first],np.int64)
    assert np.all(raw['step'][producing]==raw['parent_step'][child_first])
    birth=np.bincount(producing,weights=raw['birth'][child_first],minlength=len(raw))
    assert np.max(abs(raw['energy']-raw['remaining']-raw['dep']-birth))<1e-8
    within=raw['track'][1:]==raw['track'][:-1]
    gaps=np.flatnonzero(within & (np.diff(raw['step'])!=1))
    for i in gaps:
        assert raw['step'][i+1]-raw['step'][i]==2
        assert np.array_equal(raw['post'][i],raw['pre'][i+1])
        assert np.array_equal(raw['post_direction'][i],raw['direction'][i+1])
        assert raw['remaining'][i]==raw['energy'][i+1]
        assert raw['pre_material'][i]==raw['post_material'][i] and raw['pre_material'][i+1]!=raw['post_material'][i]
    assert np.max(abs(raw['remaining'][:-1][within]-raw['energy'][1:][within]))<1e-8
    assert np.max(abs(raw['post'][:-1][within]-raw['pre'][1:][within]))<1e-6
    material=first['pre_material'][parent<0][0];inside=raw['pre_material']==material
    nav=np.flatnonzero(~inside & (abs(raw['pre']).max(axis=1)<20.-1e-6))
    assert len(nav)>0
    assert np.all(np.isin(gaps+1,nav)), 'Uncertified interior/source gap'
    for i in nav:
        assert raw['length'][i]<=1e-6 and np.linalg.norm(raw['post'][i]-raw['pre'][i])<=1e-6
        spacing=np.array([.1,2.,.5]);face=np.rint(raw['pre'][i]/spacing)*spacing
        assert np.any((abs(raw['pre'][i]-face)<=1e-9)&(abs(raw['post'][i]-face)<=1e-9))
        assert raw['energy'][i]==raw['remaining'][i] and raw['dep'][i]<1e-12 and birth[i]==0
        assert np.array_equal(raw['direction'][i],raw['post_direction'][i])
        assert i+1<len(raw) and raw['track'][i+1]==raw['track'][i] and inside[i+1]
        assert np.array_equal(raw['post'][i],raw['pre'][i+1])
        assert np.array_equal(raw['post_direction'][i],raw['direction'][i+1])
        assert raw['remaining'][i]==raw['energy'][i+1]
    physical_outside=np.flatnonzero(~inside & (abs(raw['pre']).max(axis=1)>=20.-1e-6))
    _,idx=np.unique(raw['track'][physical_outside],return_index=True)
    escape=float(raw['energy'][physical_outside[idx]].sum())
    profiles=[]
    for flag in [False,True]:
        ids=np.flatnonzero(inside & (photon[groups]==flag) & (raw['dep']>0))
        pre=raw['pre'][ids,0].copy();post=raw['post'][ids,0].copy();pre[raw['pdg'][ids]==22]=post[raw['pdg'][ids]==22]
        profiles.append(chord_profile(pre,post,raw['dep'][ids],np.zeros(len(ids),np.int64),1)[0])
    family=read_summary(r/'rare_with/family.phsp',200000)
    earlier=read_summary(r/'voxel200k_v3/train/family.phsp',200000)
    assert np.array_equal(family,earlier), 'Selected raw recorder changed history summaries'
    selected=family[21602]
    assert np.max(abs(profiles[0]-selected['profile']))<1e-12
    assert np.max(abs(profiles[1]-selected['photon_profile']))<1e-12
    assert abs(escape-selected['escape'])<1e-12
    assert selected['navigation']==len(nav)
    save(r/'navigation_check.json',dict(status='PASS_RAW_NAVIGATION_AND_SCORER_NEUTRALITY',event=21602,
        raw_records=len(raw),full_summary_histories=200000,all_selected_transport_bytes_identical=True,
        all_history_summaries_identical=True,navigation_records=nav.tolist(),
        certified_zero_state_gap_records=gaps.tolist(),
        navigation_lengths_mm=raw['length'][nav].tolist(),navigation_deposit_MeV=raw['dep'][nav].tolist(),
        raw_sha256=sha(r/'rare_with/steps.phsp'),script_sha256=sha(Path(__file__))))
    print('PASS',len(raw),'records',len(nav),'certified navigation steps')


if __name__=='__main__':main()
