#!/usr/bin/env python3
"""Verify extraction-kit sources, reference coverage and local includes/imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

KIT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    provenance = json.loads((KIT/'source_provenance.json').read_text())
    for record in provenance['files']:
        path = KIT/record['file']
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing/nonlocal extraction source: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['installed_sha256']:
            raise ValueError(f'Extraction source digest changed: {path}')
    for path in (KIT/'tools').rglob('*.py'):
        ast.parse(path.read_text())
    headers = {p.name for p in (KIT/'topas').rglob('*.hh')}
    headers.update(p.name for p in (KIT/'topas').rglob('*.hpp'))
    for path in (KIT/'topas').rglob('*.cc'):
        for name in re.findall(r'^#include "([^"]+)"', path.read_text(), re.M):
            if name.startswith(('G4', 'Ts', 'Randomize', 'CLHEP')):
                continue
            if name.startswith('carbon/'):
                if not (KIT.parent/'include'/name).is_file():
                    raise ValueError(f'Missing runtime math header: {name}')
            elif name not in headers:
                raise ValueError(f'Missing exporter header: {path}: {name}')
    coverage = json.loads((KIT/'coverage.json').read_text())
    installed = json.loads((KIT.parent/'data/manifest.json').read_text())
    if set(coverage['assets']) != set(installed['files']):
        raise ValueError('Extraction coverage differs from installed asset manifest')
    if (KIT/'tools/neutron').exists():
        raise ValueError('Neutron package extraction is outside this runtime')
    result = dict(passed=True, sources=len(provenance['files']), assets=len(coverage['assets']))
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
