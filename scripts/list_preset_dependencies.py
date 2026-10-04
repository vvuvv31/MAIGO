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
    electron = json.loads((root/'docs/electron_packages_manifest.json').read_text())
    entries, files = {}, {}
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        dependencies = {}
        for key, value in cfg.items():
            if isinstance(value, str) and value.startswith(('data/', 'inputs/')):
                dependencies[key] = value
        # The unified loss model resolves this default even if absent in YAML.
        if cfg.get('em_model') == 'g4_material_joint_v1':
            dependencies.setdefault('em_delta_moments_file', 'data/em/unified_em_delta_moments_v2.bin')
        packages = []
        if cfg.get('material_electron_birth_file'):
            packages = [path.stem.split('_')[0], 'shared']
        expanded = set(dependencies.values())
        for name in packages:
            package = electron['packages'][name]
            expanded.update(str(Path(package['root'])/p) for p in package['manifest']['files'])
        for filename in sorted(expanded):
            p = root/filename
            if not p.is_file():
                raise ValueError(f'{path.name}: missing input {filename}')
            files[filename] = {'bytes': p.stat().st_size}
        entries[path.stem] = dict(config=str(path.relative_to(root)),
             primary=cfg.get('primary_particle', 'carbon'), medium=path.stem.split('_')[1],
             beam=path.stem.split('_')[2], ion_scattering=cfg['multiple_scattering_model'],
             electron_packages=packages, inputs=dependencies)
    return dict(schema=2, scope='Eight preset inputs plus expanded electron payloads. Auxiliary inputs referenced inside other package metadata are validated by their loaders.',
                presets=entries, files=files,
                total_unique_bytes=sum(v['bytes'] for v in files.values()),
                electron_registry='docs/electron_packages_manifest.json')


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
