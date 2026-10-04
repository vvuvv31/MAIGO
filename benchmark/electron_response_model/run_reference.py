#!/usr/bin/env python3
"""Independent monoenergetic electron training/test sources; preserve full raw records."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'out/electron_mono_reference_20261002'
BINARY = Path('/mnt/sda/wuwei/real_ct_electron_diagnostic_20260928/build/topas')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    original = (ROOT/'data/ct_electron/carbon/s1_mid9/topas.txt').read_text()
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
               LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH', ''))
    cases = []
    for name, seed in [('train', 1002210001), ('test', 1002210002)]:
        d = OUT/name;d.mkdir()
        text = re.sub(r'i:Ts/Seed = \d+', f'i:Ts/Seed = {seed}', original)
        # These two custom modules only modify baryons/hadron MSC; native
        # standard_opt4 already gives the same electron physics without them.
        text = re.sub(r'sv:Ph/Default/Modules = .*', 'sv:Ph/Default/Modules = 1 "g4em-standard_opt4"', text)
        text = text.replace('"GenericIon(6,12)"', '"e-"')
        text = re.sub(r'd:So/Beam/BeamEnergy = .*', 'd:So/Beam/BeamEnergy = 0.65 MeV', text)
        text = re.sub(r'i:So/Beam/NumberOfHistoriesInRun = \d+', 'i:So/Beam/NumberOfHistoriesInRun = 100000', text)
        (d/'topas.txt').write_text(text)
        start = time.monotonic()
        with (d/'topas.log').open('x') as log:
            code = subprocess.run([str(BINARY), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log = (d/'topas.log').read_text()
        counts = re.findall(r'Particle source Beam: Total number of histories: (\d+)', log)
        path = d/'steps.phsp'
        unscored = sum(map(float, re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV', log)))
        accepted = code == 0 and counts and int(counts[-1]) == 100000 and path.exists() and unscored/(.65*100000) < 1e-6
        status = dict(name=name, exit_code=code, accepted=bool(accepted), histories=100000,
                      seconds=time.monotonic()-start, energy_MeV=.65, particle='e-',
                      density_g_cm3=.2993595004081726, section=1, production_cut_mm=.05,
                      input_sha256=sha(d/'topas.txt'), binary_sha256=sha(BINARY),
                      path=str(path), sha256=sha(path) if path.exists() else None,
                      unscored_MeV=unscored)
        # Read exact material density from the source configuration.
        status['density_g_cm3'] = float(re.search(r'd:Ma/[^\n]+/Density = (\S+) g/cm3', text)[1])
        (d/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        if not accepted:
            raise RuntimeError(f'Incomplete electron reference: {d}')
        cases.append(status)
        (OUT/'manifest.json').write_text(json.dumps(cases, indent=2)+'\n')
        print(name, status['seconds'], path.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
