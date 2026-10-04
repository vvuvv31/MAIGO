#!/usr/bin/env python3
"""Extract all supported Water_75eV electron/photon collision cells and compile them.

Requires native primitives from extract_water_electron_primitives.py. This
creates a new material operator, and does not mark water dose as accepted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from electron_package_common import audit_collision_records, save
from electron_hard_rates import witnesses
from extract_water_electron_primitives import CARD
import compile_electron_response
import compile_photon_response

IDENTITY = dict(section_id=25, material_name='Water_75eV', runtime_density_g_cm3=1.0,
                mean_excitation_energy_eV=75, production_cut_mm=.05)


def extract(args, kind):
    d = args.out / kind / 'node00'
    d.mkdir(parents=True)
    requests = []
    if kind == 'charged':
        rates = np.loadtxt(args.primitives/'parameters/boundary_parameters.csv', delimiter=',', skiprows=1)
        cubics = np.loadtxt(args.primitives/'parameters/hard_rate_cubics.csv', delimiter=',', skiprows=1)
        energies = rates[:, 0]
        edges = np.r_[energies[0], np.sqrt(energies[:-1]*energies[1:]), energies[-1]]
        support = witnesses(cubics, edges)
        for (cell, process), witness in sorted(support.items()):
            energy = energies[cell] if rates[cell, 10+process] > 0 else witness
            requests.append((cell, process, energy, args.samples))
        binary = args.native_root/'electron_collision_native/topas'
        quantity, marker, pdg = 'ElectronCollisionDump', 'ELECTRON_COLLISION_DUMP=PASS', 11
    else:
        rates = np.loadtxt(args.primitives/'photon_rates/photon_rates.csv', delimiter=',', skiprows=1)
        energies = np.geomspace(1e-5, args.maximum_energy, 257)
        anchors = energies.astype(np.float32).astype(float)
        edges = np.r_[anchors[0], np.sqrt(anchors[:-1]*anchors[1:]), args.maximum_energy]
        for cell, energy in enumerate(energies):
            for process in range(3):
                chosen = energy
                if np.interp(energy, rates[:, 0], rates[:, process+1]) <= 0:
                    support = rates[(rates[:, 0] >= edges[cell]) & (rates[:, 0] < edges[cell+1]) & (rates[:, process+1] > 0), 0]
                    if not len(support):
                        continue
                    chosen = support[len(support)//2]
                requests.append((cell, process, chosen, args.samples))
        binary = args.native_root/'photon_conditional_native/topas'
        quantity, marker, pdg = 'PhotonConditionalDump', 'PHOTON_CONDITIONAL_DUMP=PASS', 22
    (d/'requests.txt').write_text(''.join(f'{i} {p} {e:.17g} {n}\n' for i, p, e, n in requests))
    (d/'topas.txt').write_text(CARD.format(quantity=quantity))
    env = dict(os.environ)
    env['TOPAS_G4_DATA_DIR'] = '/software/geant4-11.3.2/share/Geant4/data'
    env['LD_LIBRARY_PATH'] = '/software/topas/lib:/software/geant4-11.3.2/lib:'+env.get('LD_LIBRARY_PATH', '')
    env['MAIGO_ELECTRON_MAX_MEV'] = str(args.maximum_energy)
    env.pop('MAIGO_PHOTON_XS', None)
    start = time.monotonic()
    with (d/'topas.log').open('w') as log:
        result = subprocess.run([str(binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(d/'execution.json', dict(binary=str(binary), returncode=result.returncode, seconds=time.monotonic()-start))
    if result.returncode or marker not in (d/'topas.log').read_text():
        raise RuntimeError(f'{kind} native extraction failed: {d}')
    audit = audit_collision_records(d, IDENTITY, pdg)
    save(d/'report.json', dict(node=0, audit=audit))
    print(kind, 'native audited', audit['collision_samples'], flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--primitives', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=512)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    for field in ['out', 'primitives', 'native_root']:
        setattr(args, field, getattr(args, field).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.primitives/'manifest.json').read_text())
    if manifest['material'] != 'Water_75eV' or manifest['excitation_eV'] != 75:
        raise ValueError('Wrong native water primitives')
    args.maximum_energy = manifest['maximum_energy_MeV']
    for kind in ['parameters', 'elastic', 'urban', 'photon_rates']:
        shutil.copytree(args.primitives/kind, args.out/kind/'node00')
    par = args.out/'parameters/node00'
    options = dict(line.split('=', 1) for line in (par/'boundary_options.txt').read_text().splitlines())
    save(par/'report.json', dict(material_identity=IDENTITY, options=options))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract, args, kind) for kind in ['charged', 'photon']]
        for future in futures:
            future.result()
    charged_out = args.out/'charged_compiled'
    charged_out.mkdir()
    charged = compile_electron_response.one((0, args.out/'charged', None, args.out/'parameters',
                                             args.out/'elastic', args.out/'urban', charged_out))
    photon = compile_photon_response.one((args.out/'photon/node00', args.out/'photon_compiled/node00',
                                          257, args.out/'photon_rates/node00', args.maximum_energy))
    if not charged['coverage_complete'] or not photon['coverage_complete']:
        raise RuntimeError('Water operator has unsupported positive-rate cells; no deployment index emitted')
    for kind, magic in [('charged', 'MAIECB02'), ('photon', 'MAIPHB02')]:
        (args.out/f'{kind}_compiled/index.txt').write_text(f'{magic} 1\n25 1 node00/operator.bin\n')
    save(args.out/'manifest.json', dict(material=IDENTITY, charged=charged, photon=photon,
                                      status='NATIVE_OPERATOR_COVERAGE_ONLY_NOT_DOSE_ACCEPTANCE'))


if __name__ == '__main__':
    main()
