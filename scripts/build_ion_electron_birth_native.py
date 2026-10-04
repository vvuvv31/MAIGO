#!/usr/bin/env python3
"""Build a shared proton/C12 birth recorder without changing native physics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from electron_package_common import save

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Existing native birth build with its compiler/link recipe')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    spec = json.loads((base/'build_manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    for name in ['IonElectronBirthNtuple.cc', 'IonElectronBirthNtuple.hh']:
        shutil.copy2(ROOT/'benchmark/electron_response_model/native'/name, out/name)
    manager = (base/'TsExtensionManager.cc').read_text()
    manager = manager.replace('ProtonElectronBirthNtuple', 'IonElectronBirthNtuple').replace('protonelectronbirthntuple', 'ionelectronbirthntuple')
    (out/'TsExtensionManager.cc').write_text(manager)
    commands = []
    for original in spec['commands']:
        command = [arg.replace(str(base), str(out)).replace('ProtonElectronBirthNtuple', 'IonElectronBirthNtuple') for arg in original]
        commands.append(command)
        subprocess.run(command, cwd=spec['base_build'], check=True)
    save(out/'build_manifest.json', dict(commands=commands, base_build=spec['base_build'],
         source_files=['IonElectronBirthNtuple.cc', 'IonElectronBirthNtuple.hh', 'TsExtensionManager.cc'],
         physics_changes='none; recorder consumes no RNG, never kills a track or changes a process',
         scope='proton/C12 ionization energy and electron birth states, 0..450 MeV/u, primary step <=0.25 mm'))


if __name__ == '__main__':
    main()
