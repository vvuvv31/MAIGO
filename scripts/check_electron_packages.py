#!/usr/bin/env python3
"""Check the three electron packages' sizes, schemas, dependencies and coverage.

GPU loaders additionally check numeric fields, conditional support and rate bounds.
"""
import argparse
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    registry = json.loads((root/'docs/electron_packages_manifest.json').read_text())
    require(registry['schema'] == 2, 'Registry schema')
    packages = registry['packages']
    require(set(packages) == {'proton', 'carbon', 'shared'}, 'Expected three packages')
    report = {}
    for name, entry in packages.items():
        directory = root/entry['root']
        spec = json.loads((directory/'manifest.json').read_text())
        require(spec == entry['manifest'], f'{name}: registry mismatch')
        total = 0
        for filename, item in spec['files'].items():
            relative = Path(filename)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Nonrelative member')
            path = directory/relative
            require(path.resolve().is_relative_to(directory.resolve()), 'Member outside package')
            require(path.is_file() and path.stat().st_size == item['bytes'], f'{name}: size mismatch: {filename}')
            total += item['bytes']
        require(total == spec['total_bytes'], f'{name}: total size')
        for dep in spec.get('requires', []):
            require(dep['package_id'] == packages['shared']['manifest']['package_id'], 'Shared dependency')
        if name != 'shared':
            with (directory/'birth_bank.bin').open('rb') as stream:
                magic, version, nodes, z, a = struct.unpack('<8s4I', stream.read(24))
            require((magic, version) == (b'MAIBIR02', 2), 'Birth schema')
            require([z, a] == spec['projectile'] and nodes == 83, 'Birth projectile/material count')
        else:
            with (directory/'joint.bin').open('rb') as stream:
                magic, count = struct.unpack('<8sI', stream.read(12))
            require((magic, count) == (b'MAIJNB02', 25), 'Joint schema/Schneider coverage')
            for component, magic in [('charged', 'MAIECB02'), ('photon', 'MAIPHB02')]:
                lines = (directory/component/'index.txt').read_text().splitlines()
                require(lines[0].split() == [magic, '83'] and len(lines) == 84, 'Operator index/count')
                keys, sections = set(), set()
                for row in lines[1:]:
                    section, density, filename = row.split()
                    section, density = int(section), float(density)
                    require(0 <= section <= 25 and density > 0, 'Material/density domain')
                    require((section, density) not in keys, 'Duplicate density node')
                    keys.add((section, density)); sections.add(section)
                    if section == 25:
                        require(density == 1, 'Water_75eV density')
                    require(str(Path(component)/filename) in spec['files'], 'Unregistered operator')
                require(sections == set(range(26)), 'Missing material')
        report[name] = dict(files=len(spec['files']), bytes=total, verified=True)
    require(sum(x['bytes'] for x in report.values()) == registry['total_unique_bytes'], 'Combined size')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(verify(args.root.resolve()), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
