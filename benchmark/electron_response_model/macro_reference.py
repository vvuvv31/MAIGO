#!/usr/bin/env python3
"""Native Geant4 sphere responses for a bounded electron macro-step experiment.

Vacuum outside the sphere makes the first material exit terminal for this
local problem. Geometry, rather than clipping an infinite-medium trajectory,
limits the electron step. This is a response generator, not CT acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_reference import ROOT, BINARY, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', default='s1_mid9')
    p.add_argument('--energy', type=float, default=.65)
    p.add_argument('--radius', type=float, default=1.)
    p.add_argument('--histories', type=int, default=100000)
    p.add_argument('--seed', type=int, default=1002230001)
    p.add_argument('--max-step', type=float, default=0., help='Optional native electron step-size diagnostic, mm')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=BINARY, help='Explicit native generator; identity is recorded')
    args = p.parse_args()
    args.binary=args.binary.resolve()
    if min(args.energy, args.radius, args.histories) <= 0:
        p.error('Energy, radius and histories must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    original = (ROOT/'data/ct_electron/carbon'/args.table/'topas.txt').read_text()
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
               LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH', ''))
    cases = []
    for name, seed in [('train', args.seed), ('test', args.seed+1)]:
        d = args.out/name; d.mkdir()
        source = re.sub(r'i:Ts/Seed = \d+', f'i:Ts/Seed = {seed}', original)
        source = re.sub(r'sv:Ph/Default/Modules = .*', 'sv:Ph/Default/Modules = 1 "g4em-standard_opt4"', source)
        source = source.replace('"GenericIon(6,12)"', '"e-"')
        source = re.sub(r'd:So/Beam/BeamEnergy = .*', f'd:So/Beam/BeamEnergy = {args.energy:.17g} MeV', source)
        source = re.sub(r'i:So/Beam/NumberOfHistoriesInRun = \d+', f'i:So/Beam/NumberOfHistoriesInRun = {args.histories}', source)
        source = source.replace('s:Ge/ROI/Type = "TsBox"', 's:Ge/ROI/Type = "TsSphere"')
        source = re.sub(r'd:Ge/ROI/HL[XYZ] = .*\n', '', source)
        source = re.sub(r'd:Ge/World/HL([XYZ]) = .*',
                        lambda m: f'd:Ge/World/HL{m[1]} = {max(100., 10*args.radius):.17g} mm', source)
        # Record the first vacuum step too. Geant4 may navigate across a face
        # within surface tolerance without a final fGeomBoundary ROI record;
        # the next pre-state is the authoritative exit energy/direction.
        source = source.replace('s:Sc/Steps/Component = "ROI"', 's:Sc/Steps/Component = "Envelope"')
        source = source.replace('s:Ge/ROI/Parent = "World"', 's:Ge/ROI/Parent = "Envelope"')
        source += ('\ns:Ge/Envelope/Parent = "World"\ns:Ge/Envelope/Type = "TsBox"\n'
                   's:Ge/Envelope/Material = "Vacuum"\n'+
                   ''.join(f'd:Ge/Envelope/HL{axis} = {max(50., 5*args.radius):.17g} mm\n' for axis in 'XYZ'))
        source += '\nb:Sc/Steps/PropagateToChildren = "True"\n'
        if args.max_step > 0:
            source += f'd:Ge/ROI/MaxStepSize = {args.max_step:.17g} mm\n'
        source += (f'\nd:Ge/ROI/RMin = 0 mm\nd:Ge/ROI/RMax = {args.radius:.17g} mm\n'
                   'd:Ge/ROI/SPhi = 0 deg\nd:Ge/ROI/DPhi = 360 deg\n'
                   'd:Ge/ROI/STheta = 0 deg\nd:Ge/ROI/DTheta = 180 deg\n')
        (d/'topas.txt').write_text(source)
        start = time.monotonic()
        with (d/'topas.log').open('x') as log:
            code = subprocess.run([str(args.binary), 'topas.txt'], cwd=d, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic()-start
        log = (d/'topas.log').read_text()
        counts = re.findall(r'Particle source Beam: Total number of histories: (\d+)', log)
        path = (d/'steps.phsp').resolve()
        unscored = sum(map(float, re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV', log)))
        accepted = (code == 0 and counts and int(counts[-1]) == args.histories and
                    path.exists() and unscored < 1e-6*args.energy*args.histories)
        status = dict(name=name, exit_code=code, accepted=bool(accepted), histories=args.histories,
                      seed=seed, seconds=elapsed, energy_MeV=args.energy, particle='e-',
                      geometry='sphere_in_vacuum', radius_mm=args.radius, table=args.table,
                      records_include_vacuum=True,
                      requested_max_step_mm=args.max_step,
                      density_g_cm3=float(re.search(r'd:Ma/[^\n]+/Density = (\S+) g/cm3', source)[1]),
                      production_cut_mm=.05, input_sha256=sha(d/'topas.txt'),
                      binary_sha256=sha(args.binary), path=str(path),
                      sha256=sha(path) if path.exists() else None, unscored_MeV=unscored,
                      ct_acceptance=False)
        (d/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        cases.append(status)
        (args.out/'manifest.json').write_text(json.dumps(cases, indent=2)+'\n')
        print(json.dumps(status), flush=True)
        if not accepted:
            raise RuntimeError(f'Incomplete sphere reference: {d}')


if __name__ == '__main__':
    main()
