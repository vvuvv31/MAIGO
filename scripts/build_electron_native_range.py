#!/usr/bin/env python3
"""Rebuild native material probes with an explicit extraction energy domain."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from electron_package_common import save

ROOT = Path(__file__).resolve().parents[1]
PROBES = {'electron_boundary_native_v5':'ElectronBoundaryProbe', 'photon_native':'PhotonResponseNtuple',
          'electron_collision_native':'ElectronCollisionDump', 'photon_conditional_native':'PhotonConditionalDump'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.base_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder, name in PROBES.items():
        old, new = base/folder, out/folder
        spec = json.loads((old/'build_manifest.json').read_text())
        new.mkdir()
        for file in [name+'.cc', name+'.hh', 'ElectronExtractionRange.hh']:
            shutil.copy2(ROOT/'benchmark/electron_response_model/native'/file, new/file)
        shutil.copy2(old/'TsExtensionManager.cc', new/'TsExtensionManager.cc')
        commands = []
        for original in spec['commands']:
            command = [arg.replace(str(old), str(new)) for arg in original]
            commands.append(command)
            subprocess.run(command, cwd=spec['base_build'], check=True)
        save(new/'build_manifest.json', dict(commands=commands, base_build=spec['base_build'],
             source_files=[name+'.cc', name+'.hh', 'ElectronExtractionRange.hh', 'TsExtensionManager.cc'],
             physics_changes='none; configurable exported energy range only'))
    # These probes export energy-independent material constants and native
    # formula corrections; their binaries do not need a domain modification.
    for folder in ['electron_elastic_native_v2', 'electron_fluctuation_native']:
        (out/folder).symlink_to(base/folder, target_is_directory=True)


if __name__ == '__main__':
    main()
