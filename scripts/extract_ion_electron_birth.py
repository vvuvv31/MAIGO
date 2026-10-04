#!/usr/bin/env python3
"""Record projectile-specific material births with the shared native ion scorer."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np
from electron_package_common import save
from extract_water_electron_primitives import CARD

DTYPE = np.dtype([(k, '<i4') for k in ['run', 'event', 'kind', 'bin', 'step']]
                 + [(k, '<f8') for k in ['parent', 'birth', 'mu', 'loss', 'local', 'children', 'residual', 'length', 'density']])
assert DTYPE.itemsize == 92


def extract(args, material):
    species = args.projectile
    z, mass = (1, 1) if species == 'proton' else (6, 12)
    section, density = int(material['section']), float(material['density_g_cm3'])
    name = material['material']
    energy = args.energy_mevu or (250 if species == 'proton' else (300 if section == 25 else 255))
    if not 0 <= section <= 25 or not np.isfinite(density) or density <= 0:
        raise ValueError('Invalid electron material node')
    if energy <= 0 or energy % 5 or energy > 450:
        raise ValueError('Birth energy ceiling must be a multiple of 5 MeV/u in (0,450]')
    if section == 25 and (name != 'Water_75eV' or density != 1):
        raise ValueError('Water must be unit-density Water_75eV')
    d = args.out/f"node{material['node']:02d}"
    d.mkdir()
    text = CARD.format(quantity='IonElectronBirthNtuple')
    if section != 25:
        definition = Path(material['definition_file']).read_text()
        lines = [line for line in definition.splitlines() if ':Ma/' in line]
        if not lines:
            raise ValueError('Schneider node requires explicit TOPAS material definitions')
        text = text.replace('s:Ge/ROI/Material = "Water_75eV"', f's:Ge/ROI/Material = "{name}"')
        text += '\n' + '\n'.join(lines) + '\n'
    text = text.replace('sv:Ph/Default/Modules = 1 "g4em-standard_opt4"',
                        'sv:Ph/Default/Modules = 3 "g4em-standard_opt4" "AllChargedUrbanPhysics" "PrimaryIonizationResponsePhysics"')
    particle = 'proton' if z == 1 else 'GenericIon(6,12)'
    text = text.replace('s:So/Beam/BeamParticle = "gamma"', f's:So/Beam/BeamParticle = "{particle}"')
    text = text.replace('d:So/Beam/BeamEnergy = 0.00001 MeV', f'd:So/Beam/BeamEnergy = {energy*mass} MeV')
    text = text.replace('i:So/Beam/NumberOfHistoriesInRun = 1', f'i:So/Beam/NumberOfHistoriesInRun = {args.histories}')
    text = text.replace('s:Sc/Steps/OutputFile = "photons"', 's:Sc/Steps/OutputFile = "birth"')
    (d/'topas.txt').write_text(text)
    env = dict(os.environ)
    env['TOPAS_G4_DATA_DIR'] = '/software/geant4-11.3.2/share/Geant4/data'
    env['LD_LIBRARY_PATH'] = '/software/topas/lib:/software/geant4-11.3.2/lib:'+env.get('LD_LIBRARY_PATH', '')
    env.pop('MAIGO_PHOTON_XS', None)
    start = time.monotonic()
    with (d/'topas.log').open('w') as log:
        result = subprocess.run([str(args.binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(d/'execution.json', dict(binary=str(args.binary), returncode=result.returncode, seconds=time.monotonic()-start))
    if result.returncode:
        raise RuntimeError(f'Birth extraction failed: {d}')
    log = (d/'topas.log').read_text()
    counts = re.findall(r'Particle source Beam: Total number of histories: (\d+)', log)
    assert counts and int(counts[-1]) == args.histories
    assert sum(map(float, re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV', log))) == 0
    raw = np.fromfile(d/'birth.phsp', DTYPE)
    assert raw.nbytes == (d/'birth.phsp').stat().st_size
    assert np.isin(raw['kind'], [0, 1]).all() and np.allclose(raw['density'], density, rtol=1e-12, atol=0)
    assert np.array_equal(np.unique(raw['event']), np.arange(args.histories))
    for key in DTYPE.names[5:]:
        assert np.isfinite(raw[key]).all(), key
    summary, births = raw[raw['kind'] == 0], raw[raw['kind'] == 1]
    assert np.all(summary['length'] <= .250001) and np.all(summary['residual'] < 1e-8)
    assert np.max(abs(np.bincount(summary['event'], weights=summary['loss'])-energy*mass)) < 1e-7
    assert np.all(births['birth'] > 0) and np.all(abs(births['mu']) <= 1)
    bins = int(energy//5)
    assert np.all(np.minimum((births['parent']/5).astype(int), bins-1) == births['bin'])
    channels, samples, offset = [], [], 0
    for b in range(bins):
        rows, emitted = summary[summary['bin'] == b], births[births['bin'] == b]
        loss, transfer, local = rows['loss'].sum(), rows['children'].sum(), rows['local'].sum()
        assert loss > 0 and abs(local+transfer-loss) < 1e-7*loss
        assert abs(emitted['birth'].sum()-transfer) < 1e-10*loss
        cdf = np.cumsum(emitted['birth'])/transfer if len(emitted) else np.empty(0)
        if len(cdf):
            cdf[-1] = 1
        samples.append(np.column_stack([cdf, emitted['birth'], emitted['mu']]).astype('<f8'))
        channels.append((5*b, 5*(b+1), transfer/loss, offset, len(emitted)))
        offset += len(emitted)
    payload = d/'birth_bank.bin'
    with payload.open('xb') as f:
        f.write(struct.pack('<8sIIII', b'MAIBIR02', 2, 1, z, mass))
        f.write(struct.pack('<idQI', section, density, offset, bins))
        for row in channels:
            f.write(struct.pack('<dddQQ', *row))
        for values in samples:
            f.write(values.tobytes())
    report = dict(material=name, section=section, density_g_cm3=density,
                  projectile_z=z, projectile_a=mass, histories=args.histories,
                  parent_energy_ceiling_MeVu=energy, maximum_step_mm=.25, production_cut_mm=.05,
                  samples=offset, channels=bins, bytes=payload.stat().st_size,
                  maximum_birth_energy_MeV=float(births['birth'].max()),
                  status='NATIVE_BIRTH_ENERGY_AUDITED_NOT_DOSE_ACCEPTANCE')
    save(d/'manifest.json', report)
    print(species, report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--projectile', choices=['proton', 'carbon'], required=True)
    parser.add_argument('--materials', type=Path, help='JSON list of node, section, density_g_cm3, material, definition_file; omitted means Water_75eV')
    parser.add_argument('--energy-mevu', type=float, help='Default: proton250, carbon Schneider255/water300')
    parser.add_argument('--histories', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.histories < 1 or args.workers < 1:
        parser.error('histories/workers must be positive')
    args.binary, args.out = args.binary.resolve(), args.out.resolve()
    materials = json.loads(args.materials.read_text()) if args.materials else [dict(node=0, section=25, density_g_cm3=1., material='Water_75eV')]
    if not materials or len({int(m['node']) for m in materials}) != len(materials):
        parser.error('Material list must be nonempty with unique nodes')
    if len({(int(m['section']), float(m['density_g_cm3'])) for m in materials}) != len(materials):
        parser.error('Duplicate material/density node')
    if args.materials:
        for m in materials:
            if 'definition_file' in m:
                m['definition_file'] = str((args.materials.resolve().parent/Path(m['definition_file'])).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(lambda m: extract(args, m), materials))
    save(args.out/'manifest.json', dict(projectile=args.projectile, nodes=reports,
         status='NATIVE_BIRTH_ENERGY_AUDITED_NOT_DOSE_ACCEPTANCE'))


if __name__ == '__main__':
    main()
