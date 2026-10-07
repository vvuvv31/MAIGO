#!/usr/bin/env python3
"""Extract 26 composition oscillator and density-independent atomic laws."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--brems-elements',action='store_true');a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    source=a.reference
    jobs=[]
    for s in range(26):
        d=a.out/f'composition{s:02d}';d.mkdir()
        (d/'topas.txt').write_text((source/d.name/'topas.txt').read_text())
        if a.brems_elements:(d/'brems_elements.enabled').write_text('2BS conditional element mixture\n')
        (d/'requests.txt').write_text('0 0 0 .04 .00099 .00099 0\n')
        energies=1e-4*np.exp(np.arange(34)*np.log(1e9)/99)
        (d/'penelope_queries.txt').write_text(''.join(f'{i} {e:.17g} .00099\n' for i,e in enumerate(energies)))
        be=np.geomspace(.00099,.65,96);bx=np.geomspace(.00099/.65,1.,256)
        (d/'brems_queries.txt').write_text(''.join(f'{i*len(bx)+j} {e:.17g} {x:.17g}\n' for i,e in enumerate(be) for j,x in enumerate(bx)))
        jobs.append(d)
    env=os.environ.copy()
    binary=str(a.binary.resolve())
    def run(d):
        start=time.monotonic()
        with (d/'topas.log').open('x') as f:r=subprocess.run([binary,'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode or 'ELECTRON_COMPOSITION_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        for name in ['penelope_oscillators.csv','penelope_weights.csv','atomic_shells.csv','atomic_radiative.csv','atomic_auger.csv','atomic_settings.txt']:
            if not (d/name).exists():raise RuntimeError(f'Missing {d/name}')
        rec=dict(section=int(d.name[-2:]),directory=str(d),seconds=time.monotonic()-start,status='EXTRACTED_NOT_GPU_QUALIFIED')
        print(json.dumps(rec),flush=True);return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(run,jobs))
    (a.out/'report.json').write_text(json.dumps(dict(compositions=26,density_nodes=0,results=records),indent=2)+'\n')
if __name__=='__main__':main()
