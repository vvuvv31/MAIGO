#!/usr/bin/env python3
"""Independent native vacancy-cascade comparison, FP32 candidate arithmetic."""
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--laws',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=100000);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    atoms={}
    for file in a.laws.glob('composition*/atomic_shells.csv'):
        for r in csv.DictReader(file.open()):
            if int(r['index'])<=1:atoms[(int(r['z']),int(r['index']))]=r
    cases=[]
    for key,r in sorted(atoms.items()):
        for cut in [1e-6,.00099]:
            for auger in [0,1]:cases.append(dict(z=key[0],index=key[1],id=int(r['shell_id']),binding=float(r['binding_MeV']),ecut=cut,pcut=cut,auger=auger,cascade=auger))
    (a.out/'topas.txt').write_text((a.laws/'composition25/topas.txt').read_text())
    (a.out/'requests.txt').write_text('0 0 0 .04 .00099 .00099 0\n')
    (a.out/'atomic_queries.txt').write_text(''.join(f"{c['z']} {c['index']} {c['ecut']} {c['pcut']} {a.samples} {c['auger']} {c['cascade']}\n" for c in cases))
    (a.out/'queries.txt').write_text(''.join(f"{c['z']} {c['id']} {c['binding']:.17g} {c['ecut']} {c['pcut']} {a.samples} {c['auger']} {c['cascade']}\n" for c in cases))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    with (a.out/'native.log').open('x') as log:subprocess.run([str(a.binary.resolve()),'topas.txt'],cwd=a.out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_atomic26.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(a.package.resolve()),str(a.out/'queries.txt'),str(a.out/'candidate.bin')],check=True)
    dtype=np.dtype([(k,'<u4') for k in ['z','index','event','photons','electrons','auger','cascade','pad']]+[(k,'<f8') for k in ['photon','electron','local','residual']])
    native=np.memmap(a.out/'atomic_events.bin',dtype,mode='r',shape=(len(cases),a.samples));candidate=np.memmap(a.out/'candidate.bin','<f4',mode='r',shape=(len(cases),a.samples,5));rows=[]
    for i,c in enumerate(cases):
        row=dict(c,moments={});passed=True
        for k,name in enumerate(['photon','electron','local','photons','electrons']):
            x=native[i][name];y=candidate[i,:,k].astype(float);se=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/a.samples)
            tolerance=c['binding']*3e-7 if k<3 else 0
            difference=float(y.mean()-x.mean());z=max(0,abs(difference)-tolerance)/se if se else (0 if abs(difference)<=tolerance else float('inf'))
            row['moments'][name]=dict(native=float(x.mean()),candidate=float(y.mean()),difference=difference,z=float(z));passed&=z<=6
        row['passed']=bool(passed);rows.append(row)
    report=dict(passed=all(r['passed'] for r in rows),scope='NATIVE_ATOMIC_LAWS_ONLY_NOT_GPU_DOSE',cases=len(rows),samples_per_case=a.samples,max_z=max(x['z'] for r in rows for x in r['moments'].values()),rows=rows)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
