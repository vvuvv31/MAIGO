#!/usr/bin/env python3
"""Compare compact collision candidates to every original runtime field/probability."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from compact_electron_runtime import NATIVE,COMPACT,CHANNEL,MAX,convert

def layout(path,kind,compact):
    with path.open('rb') as f:
        if kind=='charged':
            h=struct.unpack('<8sIid4IQ4d',f.read(80));n=h[8];bins=h[5]
            start=9600+48*(h[4]+h[6]+h[7])+8*(bins+1);cells=2*bins
        else:
            h=struct.unpack('<8sIidIIQ',f.read(40));n=h[6];bins=h[5]
            start=40+32*h[4]+8*(bins+1);cells=3*bins
        f.seek(start);channels=np.frombuffer(f.read(16*cells),CHANNEL).copy()
    assert path.stat().st_size==start+16*cells+n*(24 if compact else 32)
    return channels,np.memmap(path,COMPACT if compact else NATIVE,mode='r',offset=start+16*cells,shape=(n,))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data/archive/electron_collision_native_20261005'))
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--merged',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=channels_checked=0
    for kind in ['charged','photon']:
        old=(a.source/kind/'index.txt').read_text();assert old==(a.candidate/kind/'index.txt').read_text()
        for line in old.splitlines()[1:]:
            _,_,name=line.split();oc,src=layout(a.source/kind/name,kind,False);nc,dst=layout(a.candidate/kind/name,kind,True)
            reference=convert(src)
            if not a.merged or kind!='photon':
                assert np.array_equal(oc,nc)
                assert np.array_equal(reference,dst),'FP32 runtime fields differ'
            else:
                assert len(oc)==len(nc)
                for oldch,newch in zip(oc,nc):
                    r=reference[int(oldch['offset']):int(oldch['offset']+oldch['count'])]
                    b=dst[int(newch['offset']):int(newch['offset']+newch['count'])]
                    rw=np.diff(np.r_[np.uint64(0),r['cdf']]);bw=np.diff(np.r_[np.uint64(0),b['cdf']])
                    rn=r['pdg']!=0;bn=b['pdg']!=0
                    assert np.array_equal(rw[rn],bw[bn]),'Nonlocal probability changed'
                    for key in ['fraction','cosine','pdg','pad']:assert np.array_equal(r[key][rn],b[key][bn])
                    assert rw[~rn].sum(dtype=np.uint64)==bw[~bn].sum(dtype=np.uint64),'Local probability changed'
                    assert not len(b) or b['cdf'][-1]==MAX
                    assert np.count_nonzero(~bn)<=1
            records+=len(reference);channels_checked+=len(oc)
            del src,dst,reference
    result=dict(passed=True,original_records=records,channels=channels_checked,all_runtime_fields_exact=not a.merged,
                nonlocal_integer_probabilities_exact=True,local_probability_exact=True,merged=a.merged)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
