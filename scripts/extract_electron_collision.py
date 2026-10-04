#!/usr/bin/env python3
"""Native electron hard-collision final states; all offspring retained."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
import numpy as np
from extract_photon_conditional import analyze, save, ROOT
BASE=ROOT/'out/electron_standalone_20261003'
def one(task):
    out, node, samples, energy, binary, supplement = task
    d = out / f'node{node:02d}'
    d.mkdir()
    rates = np.loadtxt(BASE / 'electron_boundary_parameters82_v4' / f'node{node:02d}/boundary_parameters.csv', delimiter=',', skiprows=1)
    energies = rates[:, 0] if energy is None else [energy]
    requests = []
    for cell, e in enumerate(energies):
        for process in range(2):
            if np.interp(e, rates[:, 0], rates[:, 10 + process]) > 0:
                requests.append((cell, process, e, samples))
    if supplement:
        from electron_hard_rates import witnesses
        edges = np.r_[rates[0, 0], np.sqrt(rates[:-1, 0] * rates[1:, 0]), rates[-1, 0]]
        existing = {(i, p) for i, p, e, n in requests}
        cubics = np.loadtxt(BASE / 'electron_boundary_parameters82_v4' / f'node{node:02d}/hard_rate_cubics.csv', delimiter=',', skiprows=1)
        support = witnesses(cubics, edges)
        requests = [(i, p, e, samples) for (i, p), e in sorted(support.items()) if (i, p) not in existing]
    if supplement and (not requests):
        raw = d / 'conditional.phsp'
        raw.write_bytes(b'')
        identity = json.loads((BASE / 'electron_boundary_parameters82_v4' / f'node{node:02d}/report.json').read_text())['material_identity']
        result = dict(node=node, samples=samples, threshold_supplement=True, native_invoked=False, audit=dict(material_identity=identity, collision_samples=0, child_births=0, requests=0), scope='Native cubic support audit found no additional cells; no conditional extraction needed.')
        save(d / 'report.json', result)
        return result
    assert requests
    (d / 'requests.txt').write_text(''.join((f'{i} {p} {e:.17g} {n}\n' for i, p, e, n in requests)))
    text = (BASE / 'photon_conditional_bank512' / f'node{node:02d}/topas.txt').read_text().replace('PhotonConditionalDump', 'ElectronCollisionDump')
    (d / 'topas.txt').write_text(text)
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data', LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    with (d / 'topas.log').open('x') as log:
        r = subprocess.run([str(binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(d / 'execution.json', dict(returncode=r.returncode))
    assert r.returncode == 0 and 'ELECTRON_COLLISION_DUMP=PASS' in (d / 'topas.log').read_text(), d
    audit = analyze(d, dict(node=node), expected_pdg=11)
    result = dict(node=node, samples=samples, mono_energy=energy, threshold_supplement=supplement, audit=audit)
    save(d / 'report.json', result)
    print(node, audit, flush=True)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',nargs='+',type=int,default=[57,12,32]);p.add_argument('--samples',type=int,default=10000);p.add_argument('--energy',type=float,default=.4);p.add_argument('--grid',action='store_true');p.add_argument('--threshold-supplement',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--binary',type=Path,default=BASE/'electron_collision_native/topas');a=p.parse_args()
    assert len(set(a.nodes))==len(a.nodes) and all(0<=n<82 for n in a.nodes) and a.samples>0 and 1e-5<=a.energy<=.65
    out=a.out.resolve();out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:r=list(pool.map(one,[(out,n,a.samples,None if a.grid or a.threshold_supplement else a.energy,a.binary.resolve(),a.threshold_supplement) for n in a.nodes]))
    save(out/'report.json',dict(status='NATIVE_ELECTRON_CONDITIONAL_NOT_TRANSPORT_ACCEPTANCE',results=r))
if __name__=='__main__':main()
