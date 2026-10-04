#!/usr/bin/env python3
"""Compare the electron Urban law with the initialized native fluctuation model."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
import numpy as np
from extract_photon_conditional import sha,save,ROOT
BASE=ROOT/'out/electron_standalone_20261003'
def one(task):
    out,node,samples,binary=task;d=out/f'node{node:02d}';d.mkdir()
    text=(BASE/'photon_conditional_bank512'/f'node{node:02d}/topas.txt').read_text().replace('PhotonConditionalDump','ElectronFluctuationDump')
    (d/'topas.txt').write_text(text);(d/'fluctuation_samples.txt').write_text(str(samples)+'\n')
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    with (d/'topas.log').open('x') as log:r=subprocess.run([str(binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(d/'execution.json',dict(returncode=r.returncode,binary_sha256=sha(binary)))
    assert r.returncode==0 and 'ELECTRON_FLUCTUATION_DUMP=PASS' in (d/'topas.log').read_text(),d
    a=np.genfromtxt(d/'urban_checks.csv',delimiter=',',names=True);assert len(a)==40
    constants=np.fromfile(d/'urban_material.bin','<f8');assert constants.size==7 and np.isfinite(constants).all()
    checks=[]
    for row in a:
        z=[]
        for k in [1,2]:
            nm,pm=row[f'native_m{k}'],row[f'portable_m{k}'];nv=max(0,row[f'native_m{2*k}']-nm*nm);pv=max(0,row[f'portable_m{2*k}']-pm*pm)
            z.append(float(abs(nm-pm)/np.sqrt((nv+pv)/samples+1e-24)))
        # Simultaneous conservative two-sample DKW bound for all82x40 cells.
        bound=float(np.sqrt(np.log(4*82*40/1e-6)/samples))
        passed=max(z)<7 and row['ks']<=bound
        checks.append(dict(cell=int(row['cell']),energy_MeV=float(row['energy']),mean_MeV=float(row['mean']),cut_MeV=float(row['cut']),moment_z=z,ks=float(row['ks']),ks_bound=bound,passed=bool(passed)))
    result=dict(node=node,passed=all(x['passed'] for x in checks),samples_per_cell=samples,cells=checks,
        binary_sha256=sha(binary),table_sha256=sha(d/'urban_material.bin'),checks_sha256=sha(d/'urban_checks.csv'),
        maximum_random_draws=int(a['max_draws'].max()),native_losses_exceeding_kinetic_energy=int(a['over_energy_native'].sum()),portable_losses_exceeding_kinetic_energy=int(a['over_energy_portable'].sum()),
        scope='Raw fluctuation law only; native AlongStep stopping/energy closure, hard branches and boundary transport are separate requirements. Over-energy raw draws are retained, not clamped here.')
    save(d/'report.json',result);print(node,result['passed'],max(max(x['moment_z']) for x in checks),max(x['ks'] for x in checks),flush=True);return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',nargs='+',type=int,default=[57,12,32]);p.add_argument('--samples',type=int,default=50000);p.add_argument('--workers',type=int,default=3);p.add_argument('--binary',type=Path,default=BASE/'electron_fluctuation_native/topas');a=p.parse_args()
    out=a.out.resolve();out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:r=list(pool.map(one,[(out,n,a.samples,a.binary.resolve()) for n in a.nodes]))
    index=out/'index.txt';index.write_text(f'MAIURB01 {len(r)}\n'+''.join(f'node{x["node"]:02d}/urban_material.bin {x["table_sha256"]} node{x["node"]:02d}/urban_checks.csv {x["checks_sha256"]}\n' for x in r))
    save(out/'report.json',dict(status='ELECTRON_URBAN_FLUCTUATION_LAW_NOT_FULL_TRANSPORT',passed=all(x['passed'] for x in r),index_sha256=sha(index),results=r));assert all(x['passed'] for x in r)
if __name__=='__main__':main()
