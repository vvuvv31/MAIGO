#!/usr/bin/env python3
"""Extract independent dense/held-out Urban steps for the current species registry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

KIT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--medium', choices=['ct', 'copper', 'air'], required=True)
    parser.add_argument('--species', type=Path, default=KIT/'reference/urban_species.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--dense-nodes', type=int, default=32769)
    parser.add_argument('--validation-nodes', type=int, default=4097)
    args = parser.parse_args()
    if args.workers < 1 or min(args.dense_nodes, args.validation_nodes) < 3:
        parser.error('Positive workers and at least three nodes required')
    ions = json.loads(args.species.read_text())
    if args.medium == 'air':
        ions = [[1, 1], [6, 12]]
    materials = ['Water_75eV']+[f'M{i}' for i in range(25)] if args.medium == 'ct' else ['G4_Cu' if args.medium == 'copper' else 'G4_AIR']
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()

    def run(task):
        phase, z, a, nodes, offset = task
        case = out/phase/f'z{z}_a{a}'
        case.parent.mkdir(parents=True, exist_ok=True)
        log = case.with_suffix('.log')
        command = [str(binary), str(case), str(z), str(a), str(nodes), str(offset)]
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        accepted = result.returncode == 0 and 'ACTIVE_URBAN_ION_CONTEXT=PASS' in log.read_text()
        files = {}
        if accepted:
            for material in materials:
                for suffix in ('', '_vectors', '_mfp', '_mfp_edges'):
                    path = case/f'{material}{suffix}.csv'
                    if not path.is_file():
                        raise ValueError(f'Missing native export: {path}')
                    files[material+suffix] = dict(path=str(path), bytes=path.stat().st_size)
        record = dict(phase=phase, z=z, a=a, accepted=accepted,
                      returncode=result.returncode, files=files, command=command, log=str(log))
        print(phase, z, a, accepted, flush=True)
        return record

    tasks = [(phase, z, a, nodes, offset) for phase, nodes, offset in
             [('dense', args.dense_nodes, 0), ('validation', args.validation_nodes, .37)] for z, a in ions]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(run, tasks))
    failed = [r for r in completed if not r['accepted']]
    state = dict(status='FAILED' if failed else 'COMPLETE', failed=failed, completed=completed)
    (out/'reference_status.json').write_text(json.dumps(state, indent=2)+'\n')
    if failed:
        raise RuntimeError('Native extraction failed; inspect logs')


if __name__ == '__main__':
    main()
