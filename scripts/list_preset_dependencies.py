#!/usr/bin/env python3
"""Inventory the eight presets and their explicit local input/package dependencies."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {f'{ion}_{medium}_{beam}' for ion in ('proton', 'carbon')
           for medium in ('ct', 'water') for beam in ('broadbeam', 'minibeam')}


def inventory(root):
    paths = sorted((root/'examples').glob('*.yaml'))
    if {p.stem for p in paths} != PRESETS:
        raise ValueError('examples must contain exactly the eight current presets')
    entries, files = {}, {}
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        dependencies = {}
        for key, value in cfg.items():
            if isinstance(value, str) and value.startswith(('data/', 'inputs/')):
                dependencies[key] = value
        expanded = set(dependencies.values())
        if cfg.get('enable_ct_grid') or cfg.get('unified_water_material_file'):
            expanded.add(cfg.get('ct_schneider_file', 'data/materials/schneider/hu_to_material.txt'))
        for filename in sorted(expanded):
            p = root/filename
            if not p.is_file():
                raise ValueError(f'{path.name}: missing input {filename}')
            files[filename] = {'bytes': p.stat().st_size}
        entries[path.stem] = dict(config=str(path.relative_to(root)),
             primary=path.stem.split('_')[0], medium=path.stem.split('_')[1],
             beam=path.stem.split('_')[2], transport_pipeline=cfg.get('transport_pipeline','generic'), ion_scattering=cfg['multiple_scattering_model'],
             inputs=dependencies)
        entries[path.stem]['files'] = sorted(expanded)
    return dict(schema=4, scope='Eight preset inputs and runtime physics packages. No verification JSON sidecars.',
                presets=entries, files=files,
                total_unique_bytes=sum(v['bytes'] for v in files.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inventory(args.root.resolve())
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
