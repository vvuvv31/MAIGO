"""Source-file block bootstrap of bulk energy fractions, not interface accuracy.

Stratifies by material and original campaign; preserves unequal parent exposure.
Fixed compressed paths are reused: this does NOT estimate path-compression error.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def audit(root,replicates=2000,seed=1906177):
    manifest=json.loads((root/'manifest.json').read_text());groups={};pins={}
    for name,info in manifest['sources'].items():
        path=root/name
        if sha(path)!=info['payload_sha256']:raise ValueError('Source SHA mismatch')
        with np.load(path) as a:
            loss=a['loss'].copy()
            deposit=np.bincount(a['bins'],weights=a['weights'],minlength=len(loss))
        campaign=str(Path(info['raw_path']).parents[2])
        groups.setdefault(info['hu'],{}).setdefault(campaign,[]).append((loss,deposit))
        pins[name]=info['payload_sha256']
    rng=np.random.default_rng(seed);result={}
    for hu,strata in groups.items():
        arrays=[np.asarray(items) for items in strata.values()]
        summed=sum(a.sum(axis=0) for a in arrays)
        samples=np.zeros((replicates,2,summed.shape[1]))
        for a in arrays:
            picks=rng.integers(0,len(a),size=(replicates,len(a)))
            samples+=a[picks].sum(axis=1)
        bins=[]
        for b in (33,34,35):
            valid=samples[:,0,b]>0
            if summed[0,b]<=0 or not valid.all():raise ValueError('Missing bootstrap exposure')
            fractions=samples[:,1,b]/samples[:,0,b]
            bins.append(dict(energy_bin=b,parent_loss_MeV=float(summed[0,b]),
                pooled_nonlocal_fraction=float(summed[1,b]/summed[0,b]),
                block_bootstrap_percentile_95=np.quantile(fractions,[.025,.975]).tolist()))
        result[str(hu)]=dict(source_counts={k:len(v) for k,v in strata.items()},bins=bins)
    return dict(status='BULK_FRACTION_UNCERTAINTY_ONLY',seed=seed,replicates=replicates,
        source_manifest_sha256=sha(root/'manifest.json'),source_payload_pins=pins,cases=result,
        limitations=['not an interface confidence interval','no path-compression resampling',
                     'no material-conversion model uncertainty','small source counts limit bootstrap reliability'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
