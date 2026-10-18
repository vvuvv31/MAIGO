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
    paths = sorted((root/'config').glob('*.yaml'))
    if {p.stem for p in paths} != PRESETS:
        raise ValueError('config must contain exactly the eight current presets')
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
        # Bundle members and companion metadata are part of the installed package.
        def bundle_files(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('file', 'channels_file'):
                        if not isinstance(item, str) or not item.startswith('data/'):
                            raise ValueError(f'{path.name}: invalid bundle member {item}')
                        expanded.add(item)
                    else:
                        bundle_files(item)
            elif isinstance(value, list):
                for item in value:
                    bundle_files(item)
        bundle = cfg.get('ct_schneider_physics_bundle_file')
        if bundle:
            bundle_files(json.loads((root/bundle).read_text()))
        for filename in list(expanded):
            for suffix in ('.metadata.json', '.channels.json'):
                companion = Path(filename).with_suffix(suffix)
                if (root/companion).is_file():
                    expanded.add(str(companion))
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
    return dict(schema=4, scope='Eight preset inputs, fixed runtime defaults, bundle members and companion metadata. Generation provenance is not a runtime dependency.',
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
