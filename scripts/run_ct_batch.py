#!/usr/bin/env python3
"""Run proton CT minibeam batches in one process, retaining validated physics data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', type=int, required=True, help='Incident histories per batch')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New isolated output directory')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/carbon_mc')
    args = parser.parse_args()
    if args.histories <= 0 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Use positive histories and distinct nonnegative seeds')
    executable = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('data', 'inputs'):
        (output / name).symlink_to(ROOT / name, target_is_directory=True)
    config = yaml.safe_load((ROOT / 'config/proton_ct_minibeam.yaml').read_text())
    names = []
    hashes = {}
    for i, seed in enumerate(args.seeds):
        case = dict(config, number_of_histories=args.histories, random_seed=seed,
                    material_electron_retain_bank=True)
        name = f'batch{i:03d}.yaml'
        path = output / name
        path.write_text(yaml.safe_dump(case, sort_keys=False))
        names.append(name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'manifest.txt').write_text('\n'.join(names) + '\n')
    environment = os.environ.copy()
    compiler = Path(environment.get('CXX', '/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx'))
    environment['LD_LIBRARY_PATH'] = str(compiler.parent.parent / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
    environment['CARBON_RUNTIME_BREAKDOWN'] = '1'
    command = [str(executable), '--plan-manifest', 'manifest.txt']
    result = {'command': command, 'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'config_sha256': hashes, 'histories_per_batch': args.histories, 'seeds': args.seeds,
              'environment': {k:v for k,v in environment.items() if k.startswith('CARBON_')}}
    summary = output / 'execution.json'
    summary.write_text(json.dumps(result, indent=2) + '\n')
    start = time.perf_counter()
    with (output / 'gpu.log').open('w') as log:
        process = subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT)
    result.update(exit_code=process.returncode, wall_seconds=time.perf_counter() - start)
    text = (output / 'gpu.log').read_text()
    result['actual_histories'] = [int(n) for n in re.findall(r'^Histories: (\d+)$', text, re.M)]
    result['batch_seconds'] = [float(n) for n in re.findall(r'\[batch-wall\].*?seconds=([\d.eE+-]+)', text)]
    result['electron_bank_cache_hits'] = text.count('[electron-bank-cache] hit')
    result['quality'] = []
    for name in names:
        path = output / 'out' / Path(name).stem / 'quality_report.json'
        result['quality'].append(json.loads(path.read_text()) if path.exists() else {})
    result['accepted'] = (process.returncode == 0 and result['actual_histories'] == [args.histories]*len(names)
                          and all(q.get('accepted') and q.get('queue_overflow_count') == 0 for q in result['quality']))
    result['histories_per_second'] = (args.histories*len(names)/result['wall_seconds']) if result['accepted'] else None
    summary.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('accepted', 'wall_seconds', 'histories_per_second', 'electron_bank_cache_hits')}))
    if not result['accepted']:
        raise SystemExit(f'CT batch rejected; inspect {summary}')


if __name__ == '__main__':
    main()
