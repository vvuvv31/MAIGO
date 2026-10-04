#!/usr/bin/env python3
"""Export native Water_75eV electron/photon primitives for the shared operator.

This produces material data, not a water or CT dose-accuracy acceptance.
Existing source data and deployed banks are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

CARD = '''i:Ts/NumberOfThreads = 1
i:Ts/Seed = 114900013
b:Ts/PauseBeforeQuit = "False"
sv:Ph/Default/Modules = 1 "g4em-standard_opt4"
d:Ph/Default/CutForAllParticles = 0.05 mm
d:Ph/Default/EMRangeMax = 10 GeV
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 20000000 mm
d:Ge/World/HLY = 20000000 mm
d:Ge/World/HLZ = 20000000 mm
s:Ge/ROI/Parent = "World"
s:Ge/ROI/Type = "TsBox"
s:Ge/ROI/Material = "Water_75eV"
d:Ge/ROI/HLX = 10000000 mm
d:Ge/ROI/HLY = 10000000 mm
d:Ge/ROI/HLZ = 10000000 mm
d:Ge/ROI/MaxStepSize = 0.25 mm
s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0 mm
d:Ge/BeamPosition/TransY = 0 mm
d:Ge/BeamPosition/TransZ = 0 mm
d:Ge/BeamPosition/RotX = 0 deg
d:Ge/BeamPosition/RotY = 0 deg
d:Ge/BeamPosition/RotZ = 0 deg
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "gamma"
d:So/Beam/BeamEnergy = 0.00001 MeV
u:So/Beam/BeamEnergySpread = 0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 1
s:Sc/Steps/Quantity = "{quantity}"
s:Sc/Steps/Component = "ROI"
s:Sc/Steps/OutputType = "binary"
s:Sc/Steps/OutputFile = "photons"
s:Sc/Steps/IfOutputFileAlreadyExists = "Exit"
'''
STAGES = {
    'parameters': ('electron_boundary_native_v5', 'ElectronBoundaryProbe', 'ELECTRON_BOUNDARY_PROBE=PASS'),
    'elastic': ('electron_elastic_native_v2', 'ElectronElasticDump', 'ELECTRON_ELASTIC_DUMP=PASS'),
    'urban': ('electron_fluctuation_native', 'ElectronFluctuationDump', 'ELECTRON_FLUCTUATION_DUMP=PASS'),
    'photon_rates': ('photon_native', 'PhotonResponseNtuple', None),
}


def extract(root, native, stage, samples, maximum_energy):
    binary_dir, quantity, marker = STAGES[stage]
    d = root / stage
    d.mkdir()
    (d / 'topas.txt').write_text(CARD.format(quantity=quantity))
    (d / 'elastic_samples.txt').write_text(f'{samples}\n')
    (d / 'fluctuation_samples.txt').write_text(f'{samples}\n')
    binary = native / binary_dir / 'topas'
    env = dict(os.environ)
    env['TOPAS_G4_DATA_DIR'] = '/software/geant4-11.3.2/share/Geant4/data'
    env['LD_LIBRARY_PATH'] = '/software/topas/lib:/software/geant4-11.3.2/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['MAIGO_ELECTRON_MAX_MEV'] = str(maximum_energy)
    env.pop('MAIGO_PHOTON_XS', None)
    if stage == 'photon_rates':
        env['MAIGO_PHOTON_XS'] = '1'
    start = time.monotonic()
    with (d / 'topas.log').open('w') as log:
        result = subprocess.run([str(binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    execution = dict(binary=str(binary), returncode=result.returncode, seconds=time.monotonic()-start)
    (d / 'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    if result.returncode or (marker and marker not in (d / 'topas.log').read_text()):
        raise RuntimeError(f'Native {stage} export failed: {d / "topas.log"}')
    if stage == 'parameters':
        data = np.loadtxt(d / 'boundary_parameters.csv', delimiter=',', skiprows=1)
        if data.shape != (257, 12) or not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError('Invalid native parameter grid')
        if abs(data[-1, 0]-maximum_energy) > 1e-12:
            raise ValueError('Native parameter binary did not export the requested energy domain')
        options = dict(x.split('=', 1) for x in (d / 'boundary_options.txt').read_text().splitlines())
        if float(options['density_g_cm3']) != 1.0:
            raise ValueError('Water density differs from 1 g/cm3')
    elif stage == 'elastic':
        payload = (d / 'elastic_kernel.bin').read_bytes()
        if len(payload) != 9464 or struct.unpack_from('<8sII', payload) != (b'MAIELS01', 1, 31):
            raise ValueError('Invalid native elastic operator')
        audit = json.loads((d / 'elastic_audit.json').read_text())
        if audit['samples'] != samples or not audit['paired_random_streams_equal']:
            raise ValueError('Elastic random-stream audit failed')
    elif stage == 'urban':
        data = np.fromfile(d / 'urban_material.bin', dtype='<f8')
        if len(data) != 7 or not np.isfinite(data).all() or abs(data[1]-75e-6) > 1e-14:
            raise ValueError('Expected explicit Water_75eV excitation energy')
    else:
        data = np.loadtxt(d / 'photon_rates.csv', delimiter=',', skiprows=1)
        if data.shape[1] != 5 or not np.isfinite(data).all() or not np.all(data >= 0) or not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError('Invalid native photon rates')
        if abs(data[-1, 0]-maximum_energy) > 1e-12:
            raise ValueError('Native photon binary did not export the requested energy domain')
    execution['files'] = {p.name:p.stat().st_size for p in d.iterdir() if p.is_file()}
    print(stage, 'PASS', flush=True)
    return execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--maximum-energy-MeV', type=float, default=.65)
    args = parser.parse_args()
    if not .65 <= args.maximum_energy_MeV <= 1.0:
        parser.error('maximum energy must be in [0.65, 1.0] MeV')
    if not 1 <= args.samples <= 1000000:
        parser.error('samples must be in [1, 1000000]')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = args.native_root.resolve()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {stage:pool.submit(extract, out, native, stage, args.samples, args.maximum_energy_MeV) for stage in STAGES}
        results = {stage:f.result() for stage, f in futures.items()}
    (out / 'manifest.json').write_text(json.dumps(dict(material='Water_75eV', density_g_cm3=1.0,
        excitation_eV=75, production_cut_mm=.05, maximum_energy_MeV=args.maximum_energy_MeV, stages=results,
        status='NATIVE_PRIMITIVES_ONLY_NOT_DOSE_ACCEPTANCE'), indent=2)+'\n')


if __name__ == '__main__':
    main()
