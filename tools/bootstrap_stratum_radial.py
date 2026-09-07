"""Per-energy bulk endpoint statistics from pinned ordered paths.

NOT an interface uncertainty interval or evidence of model equivalence.
Uses exact sampled endpoints, not radial cell midpoints. Compression uncertainty
is not included; test that with independently sampled paths from the same raw.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

RHO={-1000:0.01131606474518776,100:1.0787997245788574}

def source_moments(path,hu,expected_sha):
    if hu not in RHO or sha(path)!=expected_sha:raise ValueError('Source identity/SHA mismatch')
    with np.load(path,allow_pickle=False) as a:
        v=a['vectors'];offsets=a['offsets'];bins=a['bins'];w=a['weights']
    if (v.ndim!=2 or v.shape[1]!=3 or offsets.ndim!=1 or bins.ndim!=1 or
        w.shape!=bins.shape or len(offsets)!=len(w)+1 or
        not np.issubdtype(offsets.dtype,np.integer) or not np.issubdtype(bins.dtype,np.integer)):
        raise ValueError('Invalid path schema')
    if (not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<0) or
        offsets[0]!=0 or offsets[-1]!=len(v) or np.any(np.diff(offsets)<=0) or
        np.any(bins<0) or np.any(bins>=37)):
        raise ValueError('Invalid path payload')
    # Each source is loaded once. reduceat avoids subtracting large cumulative sums.
    net=np.add.reduceat(v,offsets[:-1],axis=0) if len(w) else np.empty((0,3))
    radial2=(net[:,:2]**2).sum(axis=1)*(10/RHO[hu])**2
    moments=np.stack((np.bincount(bins,weights=w,minlength=37),
                      np.bincount(bins,weights=w*radial2,minlength=37)),axis=1)
    if not np.isfinite(moments).all():raise ValueError('Nonfinite moments')
    return moments

def summarize(items,replicates,seed):
    a=np.asarray(items,dtype=float)
    if (replicates<2 or seed<0 or a.ndim!=3 or a.shape[1:]!=(37,2) or len(a)==0
        or not np.isfinite(a).all() or np.any(a<0)):
        raise ValueError('Invalid bootstrap inputs')
    pooled=a.sum(axis=0);rng=np.random.default_rng(seed)
    sampled=a[rng.integers(0,len(a),size=(replicates,len(a)))].sum(axis=1)
    result=[]
    for b,(weight,moment) in enumerate(pooled):
        valid=sampled[:,b,0]>0
        # Missing-energy replicates are reported, NEVER discarded/renormalized.
        ci=None
        if valid.all():ci=np.quantile(np.sqrt(sampled[:,b,1]/sampled[:,b,0]),[.025,.975]).tolist()
        result.append(dict(energy_bin=b,energy_MeV=float(weight),
            rms_mm=float(np.sqrt(moment/weight)) if weight>0 else None,
            missing_exposure_replicates=int((~valid).sum()),bootstrap_percentile_95=ci))
    return result

def audit(root,r3_path,index_path,replicates=2000,seed=1906177):
    manifest_path=root/'manifest.json';m=json.loads(manifest_path.read_text())
    r3=json.loads(r3_path.read_text());index=json.loads(index_path.read_text())
    if m['source_metadata_sha256']!=sha(r3_path):raise ValueError('Metadata SHA mismatch')
    raw={s['path']:s for s in r3['sources']}
    if len(raw)!=len(r3['sources']):raise ValueError('Duplicate raw source')
    seen=set();cached={};pins={}
    for name,info in m['sources'].items():
        if Path(name).name!=name or info['raw_path'] in seen:raise ValueError('Duplicate/invalid source')
        seen.add(info['raw_path']);ref=raw.get(info['raw_path'])
        if ref is None or ref['hu']!=info['hu'] or ref['sha256']!=info['raw_sha256']:
            raise ValueError('Raw identity mismatch')
        cached[name]=source_moments(root/name,info['hu'],info['payload_sha256'])
        pins[name]=info['payload_sha256']
    if seen!=set(raw):raise ValueError('Incomplete source pool')
    result={};partition=[];index_pins={}
    for tag,spec in sorted(index.items()):
        names=[n for n,s in m['sources'].items() if s['hu']==spec['hu'] and
               Path(s['raw_path']).parents[2].name==spec['campaign']]
        if not names or len(names)!=spec['n_varied']:raise ValueError('Stratum drift')
        expected=set(names)|{n for n,s in m['sources'].items() if s['hu']!=spec['hu']}
        group_path=Path(spec['dir'])/'manifest.json';group=json.loads(group_path.read_text())
        if (group['sources']!={n:m['sources'][n] for n in expected} or
            len(expected)!=spec['n_total'] or group['source_metadata_sha256']!=sha(r3_path)):
            raise ValueError('Stratum manifest mismatch')
        index_pins[str(group_path)]=sha(group_path);partition.extend(names)
        result[tag]=dict(hu=spec['hu'],campaign=spec['campaign'],sources=names,
                        bins=summarize([cached[n] for n in names],replicates,seed))
    if len(partition)!=len(set(partition)) or set(partition)!=set(cached):raise ValueError('Incomplete/duplicate strata')
    return dict(status='BULK_ENDPOINT_STATISTICS_ONLY',interface_gate_passed=False,
        limitations=['not interface-window RMS','no model-equivalence test','fixed compressed paths',
                     'small source counts limit percentile reliability'],
        replicates=replicates,seed=seed,strata=result,npz_sha256=pins,
        provenance={str(p):sha(p) for p in (manifest_path,r3_path,index_path,Path(__file__))},
        stratum_manifest_sha256=index_pins)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ordered-dir',type=Path,required=True)
    p.add_argument('--r3-metadata',type=Path,required=True);p.add_argument('--strata-index',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=2000);p.add_argument('--seed',type=int,default=1906177)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.ordered_dir,a.r3_metadata,a.strata_index,a.replicates,a.seed)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
