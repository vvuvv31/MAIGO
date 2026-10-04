#!/usr/bin/env python3
"""Actual native AlongStep vs independent charged loss, including terminal rules."""
import argparse, concurrent.futures, json, os, subprocess
from pathlib import Path
import numpy as np
from extract_photon_conditional import save, ROOT
BASE = ROOT / 'out/electron_standalone_20261003'

def one(task):
    out, node, samples, binary = task
    d = out / f'node{node:02d}'
    d.mkdir()
    source = BASE / 'photon_conditional_bank512' / f'node{node:02d}'
    text = (source / 'topas.txt').read_text().replace('PhotonConditionalDump', 'ElectronAlongStepDump')
    (d / 'topas.txt').write_text(text)
    operator = BASE / 'charged_complete512_v2' / f'node{node:02d}/operator.bin'
    (d / 'charged_operator.txt').write_text(f'{operator} {samples}\n')
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data', LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    with (d / 'topas.log').open('x') as log:
        r = subprocess.run([str(binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(d / 'execution.json', dict(returncode=r.returncode))
    assert r.returncode == 0 and 'ELECTRON_ALONGSTEP_DUMP=PASS' in (d / 'topas.log').read_text(), d
    a = np.genfromtxt(d / 'alongstep_checks.csv', delimiter=',', names=True)
    assert len(a) == 66
    checks = []
    for row in a:
        nm, pm = (row['native_m1'], row['portable_m1'])
        variance = max(0, row['native_m2'] - nm * nm) + max(0, row['portable_m2'] - pm * pm)
        se = float(np.sqrt(variance / samples))
        passed = abs(nm - pm) <= 7 * se + 0.0002
        checks.append(dict(cell=int(row['cell']), energy_MeV=float(row['energy']), length_mm=float(row['length']), native_deposited_fraction=float(nm), portable_deposited_fraction=float(pm), delta_incident_fraction=float(pm - nm), difference_standard_error=se, native_stops=int(row['native_stops']), portable_stops=int(row['portable_stops']), maximum_energy_residual_MeV=float(row['max_closure']), passed=bool(passed)))
    result = dict(node=node, passed=all((x['passed'] for x in checks)), samples_per_cell=samples, cells=checks, scope='Actual native AlongStepDoIt including native cut/model and terminal rules, against approximate stopping/range plus validated Urban law; per-cell mean check permits7SE+0.0002 incident fraction. Not a distribution-shape, interface or CT acceptance.')
    save(d / 'report.json', result)
    print(node, result['passed'], max((abs(x['delta_incident_fraction']) for x in checks)), flush=True)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nodes', nargs='+', type=int, default=[57, 12, 32])
    p.add_argument('--samples', type=int, default=50000)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--binary', type=Path, default=BASE / 'electron_alongstep_native_v4/topas')
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        r = list(pool.map(one, [(out, n, a.samples, a.binary.resolve()) for n in a.nodes]))
    save(out / 'report.json', dict(status='ACTUAL_NATIVE_ALONGSTEP_COMPONENT_SCREEN', passed=all((x['passed'] for x in r)), results=r))
    assert all((x['passed'] for x in r))
if __name__ == '__main__':
    main()
