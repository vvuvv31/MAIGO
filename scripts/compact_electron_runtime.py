#!/usr/bin/env python3
"""Store current FP32 collision fields directly; optionally merge photon local outcomes.

Scalar/rate/energy grids remain native. Probability intervals retain current
64-bit integer thresholds. No density node, nonlocal branch or tail is thinned.
This builds explicit candidates, never replaces default data.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np

NATIVE=np.dtype([('cdf','<f8'),('fraction','<f8'),('cosine','<f8'),('pdg','<i4'),('pad','<i4')])
COMPACT=np.dtype([('cdf','<u8'),('fraction','<f4'),('cosine','<f4'),('pdg','<i4'),('pad','<i4')])
CHANNEL=np.dtype([('offset','<u8'),('count','<u8')])
MAX=np.iinfo(np.uint64).max

def convert(a):
    out=np.zeros(len(a),COMPACT)
    middle=(a['cdf']>0)&(a['cdf']<1)
    out['cdf'][middle]=np.ldexp(a['cdf'][middle],64).astype(np.uint64)
    out['cdf'][a['cdf']>=1]=MAX
    for field in ['fraction','cosine','pdg','pad']:out[field]=a[field]
    return out

def merge(a,channels):
    chunks=[];new=np.zeros(len(channels),CHANNEL);offset=0;removed=0
    for i,ch in enumerate(channels):
        block=a[int(ch['offset']):int(ch['offset']+ch['count'])]
        if not len(block):new[i]=(offset,0);continue
        assert block['cdf'][-1]==MAX and np.all(block['cdf'][1:]>=block['cdf'][:-1])
        widths=np.diff(np.r_[np.uint64(0),block['cdf']])
        local=block['pdg']==0
        if not local.any():result=block.copy()
        else:
            mass=widths[local].sum(dtype=np.uint64)
            result=np.zeros(int((~local).sum())+int(mass>0),COMPACT)
            result[:(~local).sum()]=block[~local]
            cumulative=np.cumsum(widths[~local],dtype=np.uint64)
            result['cdf'][:len(cumulative)]=cumulative
            if mass:
                result[-1]=(MAX,1.,0.,0,0)
            assert np.array_equal(np.diff(np.r_[np.uint64(0),result['cdf']])[:len(cumulative)],widths[~local])
            assert int(result['cdf'][-1])==int(MAX)
            assert int(MAX)-int(cumulative[-1] if len(cumulative) else 0)==int(mass)
        removed+=len(block)-len(result);chunks.append(result);new[i]=(offset,len(result));offset+=len(result)
    return np.concatenate(chunks) if chunks else np.empty(0,COMPACT),new,removed

def one(source,dest,kind,merge_local=False):
    with source.open('rb') as f:
        if kind=='charged':
            h=list(struct.unpack('<8sIid4IQ4d',f.read(80)))
            assert (h[0],h[1])==(b'MAIECH02',2)
            count=h[8];bins=h[5];start=9600+48*(h[4]+h[6]+h[7])+8*(bins+1)
            cells=2*bins
        else:
            h=list(struct.unpack('<8sIidIIQ',f.read(40)))
            assert (h[0],h[1])==(b'MAIPHN01',1)
            count=h[6];bins=h[5];start=40+32*h[4]+8*(bins+1);cells=3*bins
        f.seek(0);prefix=bytearray(f.read(start));channels=np.frombuffer(f.read(16*cells),CHANNEL).copy()
        raw=np.frombuffer(f.read(32*count),NATIVE)
        assert len(raw)==count and not f.read(1)
    assert np.isfinite(raw['cdf']).all() and np.isfinite(raw['fraction']).all() and np.isfinite(raw['cosine']).all()
    assert np.all((raw['cdf']>=0)&(raw['cdf']<=1)) and np.all((raw['fraction']>0)&(raw['fraction']<=1))
    assert np.all(abs(raw['cosine'])<=1) and np.isin(raw['pdg'],[0,11,22]).all()
    compact=convert(raw);removed=0
    offset=0
    for ch in channels:
        assert ch['offset']==offset and int(ch['count'])<=len(raw)-offset
        b=raw[offset:offset+int(ch['count'])]
        assert not len(b) or (b['cdf'][-1]==1 and np.all(np.diff(b['cdf'])>=0))
        offset+=int(ch['count'])
    assert offset==len(raw)
    if merge_local:compact,channels,removed=merge(compact,channels)
    prefix[:8]=b'MAIECF01' if kind=='charged' else b'MAIPHF01'
    struct.pack_into('<I',prefix,8,1)
    struct.pack_into('<Q',prefix,40 if kind=='charged' else 32,len(compact))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(prefix);f.write(channels.tobytes());f.write(compact.tobytes())
    return dict(source=str(source),file=str(dest),source_bytes=source.stat().st_size,bytes=dest.stat().st_size,
                original_records=int(count),records=len(compact),merged_local_records=removed,
                compact_schema=prefix[:8].decode(),nonlocal_integer_probability_intervals_preserved=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/archive/electron_collision_native_20261005'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--merge-photon-local',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);records=[]
    for kind in ['charged','photon']:
        lines=(args.source/kind/'index.txt').read_text().splitlines()
        target=args.output/kind;target.mkdir()
        for line in lines[1:]:
            section,density,name=line.split()
            records.append(one(args.source/kind/name,target/name,kind,args.merge_photon_local and kind=='photon'))
        (target/'index.txt').write_text('\n'.join(lines)+'\n')
    report=dict(scope='Collision outcomes only; scalar tables/birth/joint unchanged',merged_photon_local=args.merge_photon_local,
                source_bytes=sum(x['source_bytes'] for x in records),bytes=sum(x['bytes'] for x in records),nodes=records)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='nodes'}))
if __name__=='__main__':main()
