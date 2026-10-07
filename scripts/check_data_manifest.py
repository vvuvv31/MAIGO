#!/usr/bin/env python3
"""Generate or verify the installed release assets and their SHA256 digests."""
import argparse
import hashlib
import json
from pathlib import Path

from list_preset_dependencies import inventory

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def installed(root):
    entries = {}
    for folder in ('data', 'inputs'):
        for path in sorted((root/folder).rglob('*')):
            relative = str(path.relative_to(root))
            if not path.is_file() or relative == 'data/manifest.json':
                continue
            if path.is_symlink():
                raise ValueError(f'Installed assets must be local files: {relative}')
            entries[relative] = dict(bytes=path.stat().st_size, sha256=digest(path))
            if path.name.endswith('.metadata.json'):
                metadata = json.loads(path.read_text())
                filename = metadata.get('data_filename')
                if filename is not None and not (path.parent/filename).is_file():
                    raise ValueError(f'Missing metadata payload: {relative}: {filename}')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true', help='Regenerate after an intentional asset change')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    files = installed(root)
    dependencies = inventory(root)
    presets = {name: entry['files'] for name, entry in dependencies['presets'].items()}
    for name, members in presets.items():
        if not set(members).issubset(files):
            raise ValueError(f'{name}: unregistered dependency')
    manifest = dict(schema=1, scope='Installed release assets; paths relative to repository root. '
                    'Includes auxiliary generation assets; historical provenance is under docs/data_provenance.',
                    files=files, presets=presets)
    path = root/'data/manifest.json'
    if args.write:
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    elif json.loads(path.read_text()) != manifest:
        raise ValueError('Installed assets or dependencies differ from data/manifest.json')
    report = dict(passed=True, assets=len(files), presets=len(presets),
                  bytes=sum(item['bytes'] for item in files.values()))
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
