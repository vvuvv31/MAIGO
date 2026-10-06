#!/usr/bin/env python3
"""Check ion-EM-derived birth yield against completed independent native histories."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from extract_ion_electron_birth import DTYPE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,nargs='+',required=True)
    p.add_argument('--ion-em',type=Path,default=Path('data/em/unified_em_v1.bin'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);cases=[]
    for root in args.native:
        for d in sorted(root.glob('node*')):
            if not (d/'manifest.json').is_file():continue
            meta=json.loads((d/'manifest.json').read_text())
            raw=np.fromfile(d/'birth.phsp',DTYPE);summary=raw[raw['kind']==0]
            n=meta['histories'];bins=meta['channels'];rows=[]
            for b in range(bins):
                block=summary[summary['bin']==b];loss=block['loss'].sum();transfer=block['children'].sum()
                if loss<=0:raise ValueError('native empty slowing bin')
                fraction=transfer/loss
                residual=np.bincount(block['event'],weights=block['children']-fraction*block['loss'],minlength=n)
                se=np.sqrt(n/(n-1)*np.sum(residual**2))/loss
                rows.append((fraction,se))
            cases.append((meta,rows,str(d)))
    fixture=args.output/'yields.bin'
    with fixture.open('xb') as f:
        f.write(struct.pack('<I',len(cases)))
        for meta,rows,label in cases:
            f.write(struct.pack('<ifII',meta['section'],meta['density_g_cm3'],meta['projectile_a'],len(rows)))
            f.write(np.asarray(rows,dtype='<f8').tobytes())
    root=Path(__file__).resolve().parents[1];binary=args.output/'check'
    subprocess.run(['g++','-O3','-std=c++20','-I'+str(root/'include'),
        str(root/'scripts/check_electron_birth_fraction_native26.cpp'),str(root/'src/unified_em_package.cpp'),'-o',str(binary)],check=True)
    run=subprocess.run([str(binary),str(args.ion_em),str(fixture)],capture_output=True,text=True,check=True)
    (args.output/'check.txt').write_text(run.stdout);result=[]
    for line in run.stdout.splitlines():
        c,s,r,a,b,predicted,native,se=map(float,line.split())
        result.append(dict(case=int(c),section=int(s),density=r,a=int(a),bin=int(b),prediction=predicted,native=native,se=se,
                           absolute_difference=abs(predicted-native),relative_difference=predicted/native-1 if native else None,
                           z=(predicted-native)/se if se else None))
    summary={}
    for a in [1,12]:
        group=[r for r in result if r['a']==a]
        if group:summary[str(a)]=dict(cases=len(group),maximum_absolute=max(r['absolute_difference'] for r in group),
            median_absolute=float(np.median([r['absolute_difference'] for r in group])),
            maximum_z=max(abs(r['z']) for r in group if r['z'] is not None))
    report=dict(status='NATIVE_YIELD_COMPONENT_COMPARISON_NOT_DOSE_ACCEPTANCE',summaries=summary,rows=result,
                sources=[label for meta,rows,label in cases],
                quadrature='uniform energy loss through5MeV parent bin; reference has native finite pre-step binning')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
