#!/usr/bin/env python3
"""Split an assembled strict26 bank into the current shared/proton/carbon packages."""
import argparse
import json
import os
from pathlib import Path
import shutil

KIT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Flat bank from assemble_package26.py')
    parser.add_argument('--output-root', type=Path, required=True, help='Staged root containing data/ and docs/')
    args = parser.parse_args()
    registry = json.loads((KIT/'reference/electron_packages.json').read_text())
    entries = registry['packages']
    for name, entry in entries.items():
        destination = args.output_root/entry['root']
        destination.mkdir(parents=True, exist_ok=False)
        spec = entry['manifest']
        for filename in spec['files']:
            source = args.source/(f'{name}26.bin' if name != 'shared' else filename)
            target = destination/filename
            shutil.copyfile(source, target)
            spec['files'][filename]['bytes'] = target.stat().st_size
        spec['total_bytes'] = sum(item['bytes'] for item in spec['files'].values())
        (destination/'manifest.json').write_text(json.dumps(spec, indent=2)+'\n')
    shared = args.output_root/entries['shared']['root']
    for alias, relative in entries['shared']['manifest']['runtime_aliases'].items():
        os.link(shared/relative, shared/alias)
    registry['total_unique_bytes'] = sum(entry['manifest']['total_bytes'] for entry in entries.values())
    docs = args.output_root/'docs'; docs.mkdir(parents=True, exist_ok=True)
    for destination in (docs/'electron_packages_manifest.json', args.output_root/'data/em/electrons/manifest.json'):
        destination.write_text(json.dumps(registry, indent=2)+'\n')
    print('Staged strict26 packages:', registry['total_unique_bytes'], 'bytes')


if __name__ == '__main__':
    main()
