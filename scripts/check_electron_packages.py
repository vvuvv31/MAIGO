#!/usr/bin/env python3
"""Verify the three independent electron packages without loading GPU data."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    registry = json.loads((root / 'docs/electron_packages_manifest.json').read_text())
    packages = registry['packages']
    require(set(packages) == {'proton', 'carbon', 'shared'}, 'Expected three packages')
    report = {}
    for name, entry in packages.items():
        directory = root / entry['root']
        manifest = directory / 'manifest.json'
        require(sha(manifest) == entry['manifest_sha256'], f'{name}: manifest SHA')
        spec = json.loads(manifest.read_text())
        require(spec == entry['manifest'], f'{name}: registry mismatch')
        total = 0
        for filename, item in spec['files'].items():
            relative = Path(filename)
            require(not relative.is_absolute() and '..' not in relative.parts,
                    f'{name}: non-relative package member')
            path = directory / relative
            require(path.resolve().is_relative_to(directory.resolve()),
                    f'{name}: payload outside package: {filename}')
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                    f'{name}: payload mismatch: {filename}')
            total += item['bytes']
        require(total == spec['total_bytes'], f'{name}: total size')
        for dep in spec.get('requires', []):
            require(dep['package_id'] == packages['shared']['manifest']['package_id']
                    and dep['manifest_sha256'] == packages['shared']['manifest_sha256'],
                    f'{name}: shared dependency mismatch')
        if name != 'shared':
            with (directory / 'birth_bank.bin').open('rb') as stream:
                magic, version, nodes = struct.unpack('<8sII', stream.read(16))
                if magic == b'MAIBIR01' and version == 1:
                    projectile = [1, 1]
                else:
                    require(magic == b'MAIBIR02' and version == 2, 'Birth header')
                    projectile = list(struct.unpack('<II', stream.read(8)))
            require(projectile == spec['projectile'] and nodes > 0,
                    f'{name}: projectile identity')
        else:
            for component, magic in [('charged', 'MAIECB01'), ('photon', 'MAIPHB01')]:
                index = directory / component / 'index.txt'
                lines = index.read_text().splitlines()
                require(lines[0].split() == [magic, '82'] and len(lines) == 83,
                        f'{component}: index header/count')
                for row in lines[1:]:
                    section, density, filename, pin = row.split()
                    key = str(Path(component) / filename)
                    require(key in spec['files'] and spec['files'][key]['sha256'] == pin,
                            f'{component}: unpinned member {filename}')
        report[name] = dict(files=len(spec['files']), bytes=total, verified=True)
    require(sum(p['bytes'] for p in report.values()) == registry['total_unique_bytes'],
            'Combined size mismatch')
    for old, new in registry['compatibility_aliases'].items():
        require((root / old).resolve() == (root / new).resolve(), 'Compatibility alias')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
