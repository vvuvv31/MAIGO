#!/usr/bin/env python3
"""Verify native per-history scorer against fully audited raw trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import extract, DTYPE
from prototype import save, sha
from voxel_profile import chord_profile
from voxel_summary_reference import read_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    # Independent hand-computable boundary and multi-pitch chord cases.
    x=np.array([0.,-.2,1.7,-5.4]); y=np.array([0.,.2,1.9,5.4]); w=np.ones(4)
    check=chord_profile(x,y,w,np.arange(4),4)
    expected=np.zeros((4,36)); expected[0,18]=1.; expected[1,16:20]=.25
    expected[2,[35,0]]=.5; expected[3,:]=1/36
    assert np.max(abs(check-expected))<1e-13
    sources=json.loads((a.input/'manifest.json').read_text())
    originals=json.loads((a.baseline/'manifest.json').read_text())
    checks=[]
    for source,original in zip(sources,originals):
        assert source['seed']==original['seed'] and source['histories']==original['histories']
        data=extract(source)
        profile=chord_profile(data['pre'][:,0],data['post'][:,0],data['weight'],data['event'],source['histories'])
        native=read_summary(a.input/source['name']/'profile.phsp',source['histories'])
        pdiff=float(np.max(abs(profile-native['profile'])))
        ld=np.column_stack([native['charged'],np.zeros(len(native)),native['sink']])
        ldiff=float(np.max(abs(ld-data['ledger'])))
        assert pdiff<1e-12 and ldiff<1e-12
        assert int(native['steps'].sum())==data['audit']['rows']
        # Native transport must be unchanged by the extra scorer. MT row order
        # may differ; sort by complete history/track/step key before comparison.
        raw=np.memmap(source['path'],DTYPE,mode='r'); old=np.memmap(original['path'],DTYPE,mode='r')
        assert sha(Path(original['path']))==original['sha256']
        def canonical(r):
            return r[np.lexsort((r['step'],r['track'],r['event'],r['run']))].view(np.uint8)
        identical=np.array_equal(canonical(raw),canonical(old)); assert identical
        checks.append(dict(name=source['name'],histories=source['histories'],raw_records=len(raw),
                           max_profile_absolute_MeV=pdiff,max_ledger_absolute_MeV=ldiff,
                           all_transport_records_bitwise_equal=identical,audit=data['audit'],
                           summary_sha256=sha(a.input/source['name']/'profile.phsp')))
    save(a.out,dict(status='PASS_NATIVE_HISTORY_SCORER_RAW_AND_BASELINE',checks=checks,
                   analytic_chord_cases=4,script_sha256=sha(Path(__file__))))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
