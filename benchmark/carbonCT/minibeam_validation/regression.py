#!/usr/bin/env python3
"""Existing proton CT and carbon water smoke regressions for the C12 handoff."""
from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'out/carbon_ct_minibeam_20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','')
    binaries={'old':OUT/'baseline_carbon_mc','new':ROOT/'build/carbon_mc'}
    assert sha(binaries['old'])!=sha(binaries['new'])
    results=[]
    for preset in ['proton_ct_minibeam','carbon_water_minibeam']:
        rows=[]
        for mode,binary in binaries.items():
            d=OUT/f'regression_{preset}_{mode}';d.mkdir(exist_ok=False)
            cmd=[str(binary),'--config',str(ROOT/'config'/f'{preset}.yaml'),
                 '--histories-per-batch','1000','--batch-count','1','--batch-output',str(d/'batches')]
            with (d/'gpu.log').open('w') as log:
                code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            qpaths=list(d.rglob('quality_report.json'));assert len(qpaths)==1
            q=json.loads(qpaths[0].read_text());text=(d/'gpu.log').read_text()
            assert code==0 and q['accepted'] and q['queue_overflow_count']==0
            histories=re.findall(r'^Histories: (\d+)$',text,re.M);assert histories==['1000']
            dose=np.fromfile(qpaths[0].parent/'dose.raw',np.float32);assert np.isfinite(dose).all() and (dose>=0).all()
            rows.append((dose,q,text))
        a,b=rows[0][0].astype(float),rows[1][0].astype(float)
        l1=float(abs(a-b).sum()/max(a.sum(),1e-30));peak=float(abs(a-b).max()/max(a.max(),1e-30))
        em=[re.findall(r'^\[unified-em-audit\] (.*)$',x[2],re.M) for x in rows]
        assert em[0]==em[1] and em[0] and l1<1e-6 and peak<1e-6
        results.append(dict(preset=preset,histories=1000,dose_relative_l1=l1,peak_normalized_max_difference=peak,
            unified_em_audits_exact=True,quality=[x[1] for x in rows]))
        print(preset,l1,peak,flush=True)
    (OUT/'regression.json').write_text(json.dumps(dict(binary_hashes={k:sha(v) for k,v in binaries.items()},results=results),indent=2)+'\n')
if __name__=='__main__':main()
