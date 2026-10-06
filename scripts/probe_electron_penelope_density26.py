#!/usr/bin/env python3
"""Independent continuous-density oscillator inputs for26 compositions.

Explicit native runs are offline fitting/validation inputs, not runtime density
states. No source83 density identities or terminal CDFs are consumed here.
"""
import argparse,concurrent.futures,json,os,re,struct,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--points',type=int,default=9);p.add_argument('--split-conductor',action='store_true');a=p.parse_args()
    if a.points<3 or a.points>65:p.error('points must be3..65')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);domains=[]
    with (ROOT/'out/electron_cut_conditioned26_20261005/package/birth26.bin').open('rb') as f:
        magic,version,n=struct.unpack('<8sII',f.read(16))
        if (magic,version,n)!=(b'MAIBCA26',1,26):raise ValueError('Composition domains')
        for s in range(26):
            section,lo,hi,count=struct.unpack('<IffI',f.read(16))
            if section!=s:raise ValueError('Composition directory')
            domains.append((lo,hi));f.seek(16*count,1)
    jobs=[]
    for s,(lo,hi) in enumerate(domains):
        card=(ROOT/f'out/electron_complete26_20261006/composition_laws_v2/composition{s:02d}/topas.txt').read_text()
        # Chebyshev-Lobatto training coordinates and separate midpoint checks.
        coordinates=np.unique(np.r_[(1-np.cos(np.linspace(0,np.pi,a.points)))/2,
                                    (np.arange(a.points-1)+.5)/(a.points-1)]) if lo!=hi else np.array([0.])
        if a.split_conductor and s==1:
            ref=ROOT/'out/electron_complete26_20261006/composition_laws_v2/composition01'
            oscillators=np.atleast_1d(np.genfromtxt(ref/'penelope_oscillators.csv',delimiter=',',names=True))
            audit=np.atleast_1d(np.genfromtxt(ref/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
            conduction=oscillators[oscillators['ionization_MeV']==0]
            if len(conduction)!=1:raise ValueError('Expected native conductor')
            transition=float(audit[0]['density_g_cm3'])*(1e-6/float(conduction[0]['resonance_MeV']))**2
            split=(np.log(transition)-np.log(lo))/np.log(hi/lo)
            if not 0<split<1:raise ValueError('Conductor threshold outside domain')
            # Keep two independent native regimes; never smooth the onset.
            gap=1e-5
            coordinates=np.unique(np.r_[coordinates*(split-gap),split+gap+coordinates*(1-split-gap)])
        for j,x in enumerate(coordinates):
            rho=float(np.exp(np.log(lo)+x*np.log(hi/lo)))
            d=a.out/f'composition{s:02d}_sample{j:02d}';d.mkdir()
            if s==25 and lo==hi:body=card
            else:
                body,n=re.subn(r'(d:Ma/[^/]+/Density = )[\d.eE+-]+( g/cm3)',lambda m:m[1]+format(rho,'.17g')+m[2],card)
                if n!=1:raise ValueError('Expected one custom material density')
            (d/'topas.txt').write_text(body);(d/'requests.txt').write_text('0 0 0 .04 .00099 .00099 0\n')
            energies=1e-4*np.exp(np.arange(34)*np.log(1e9)/99)
            (d/'penelope_queries.txt').write_text(''.join(f'{i} {e:.17g} .00099\n' for i,e in enumerate(energies)))
            jobs.append(dict(section=s,density=rho,coordinate=float(x),directory=str(d)))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib');binary=str(a.binary.resolve())
    def run(job):
        d=Path(job['directory']);start=time.monotonic()
        with (d/'topas.log').open('x') as f:r=subprocess.run([binary,'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode or 'ELECTRON_COMPOSITION_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        osc=np.atleast_1d(np.genfromtxt(d/'penelope_oscillators.csv',delimiter=',',names=True))
        result=dict(job,oscillators=len(osc),seconds=time.monotonic()-start)
        (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    result=dict(status='NATIVE_DENSITY_COMPONENT_INPUT_NOT_GPU_PACKAGE',compositions=26,records=records)
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_runs=len(records),compositions=26,counts={s:sorted({r['oscillators'] for r in records if r['section']==s}) for s in range(26)})),flush=True)
if __name__=='__main__':main()
