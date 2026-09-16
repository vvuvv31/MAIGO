#!/usr/bin/env python3
"""Serial fixed-shard measurements, telemetry and audit for a single GPU.

Use identical source/config/input hashes on both hosts. Profiler runs are separate.
An optional candidate is alternated with the baseline; warmups are never scored.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', type=Path, help='Portable runtime/lib with its ELF loader')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--screen', action='store_true',
                   help='Exploratory rejection only; fewer repeats cannot establish acceptance')
    args = p.parse_args()
    if args.repeats < 1 or (args.repeats < 5 and not args.screen):
        p.error('At least five measured runs per binary are required')
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for verifier in ('verify_schneider_v2_1_data.py', 'verify_unified_em_data.py'):
        with (out / (verifier + '.log')).open('w') as log:
            subprocess.run(['python3', str(root / 'tools' / verifier)], cwd=root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    config = args.config.read_text()
    # Keep physical parameters unchanged, including production/research mode.
    inputs = {}
    for line in config.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        path = Path(value.strip().strip('"\''))
        if path.is_absolute() and path.is_file():
            inputs[key] = dict(path=str(path), sha256=sha(path))
    binaries = {'base': args.baseline.resolve()}
    if args.candidate:
        binaries['cand'] = args.candidate.resolve()
    env = dict(os.environ, ONEAPI_DEVICE_SELECTOR='cuda:*',
               CARBON_RUNTIME_BREAKDOWN='1', CARBON_SECONDARY_TAIL_DIAG='1')
    prefix = []
    if args.runtime:
        runtime = args.runtime.resolve()
        env['UR_ADAPTERS_FORCE_LOAD'] = str(runtime / 'libur_adapter_cuda.so.0')
        prefix = [str(runtime / 'ld-linux-x86-64.so.2'), '--library-path',
                  str(runtime) + ':/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu']
    manifest = dict(screening_only=args.screen, config_sha256=sha(args.config), inputs=inputs,
                    binaries={k: dict(path=str(v), sha256=sha(v)) for k, v in binaries.items()},
                    prefix=prefix, environment={k: v for k, v in env.items()
                    if k.startswith(('CARBON_', 'ONEAPI_', 'UR_', 'LD_LIBRARY_PATH'))})
    if args.runtime:
        manifest['runtime_hashes'] = {f.name: sha(f) for f in runtime.iterdir()
                                      if f.is_file() and not f.is_symlink()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (out / 'gpu.txt').open('w') as f:
        subprocess.run(['nvidia-smi', '-q'], stdout=f, check=True)
    order = [(True, i, k) for i in range(1 if args.screen else 2) for k in binaries]
    for i in range(args.repeats):
        order.extend((False, i, k) for k in (list(binaries) if i % 2 == 0 else list(binaries)[::-1]))
    rows, reference, reference_audit = [], None, None
    fields = 'timestamp,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used'
    for warmup, index, kind in order:
        dest = out / f'{"warm" if warmup else "run"}_{index:02d}_{kind}'
        dest.mkdir()
        (dest / 'data').symlink_to(root / 'data', target_is_directory=True)
        (dest / 'config.yaml').write_text(config)
        with (dest / 'processes.txt').open('w') as f:
            subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv'], stdout=f, check=True)
        with (dest / 'telemetry.csv').open('w') as telemetry, (dest / 'run.log').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=' + fields,
                                        '--format=csv,nounits', '-lms', '200'], stdout=telemetry)
            try:
                start = time.perf_counter()
                proc = subprocess.run(prefix + [str(binaries[kind]), '--config', 'config.yaml',
                                      '--device', 'cuda', '--voxel-dose-mhd', 'dose.mhd'],
                                      cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT)
                wall = time.perf_counter() - start
            finally:
                monitor.terminate()
                monitor.wait()
        if proc.returncode:
            raise RuntimeError(f'Run failed: {dest}')
        q = json.loads((dest / 'out/config/quality_report.json').read_text())
        if not q['accepted'] or q['failures'] or q['queue_overflow_count']:
            raise RuntimeError(f'Quality failure: {dest}; overflow requires a new, smaller matched shard campaign')
        text = (dest / 'run.log').read_text()
        def metric(pattern):
            m = re.search(pattern, text)
            if not m:
                raise RuntimeError(f'Missing metric {pattern}: {dest}')
            return float(m[1])
        audit = re.findall(r'^\[unified-em-audit\].*$', text, re.M)
        if not audit:
            raise RuntimeError('Missing EM audit')
        dose = np.fromfile(dest / 'dose.raw', dtype='<f4').astype(np.float64)
        if not np.isfinite(dose).all() or not dose.size or dose.max() <= 0:
            raise RuntimeError('Invalid dose')
        if reference is None and not warmup:
            reference, reference_audit = dose, audit
        stages = {}
        for key, value in re.findall(r'^\[runtime-stage\] (\S+) seconds=([\d.eE+-]+)', text, re.M):
            stages[key] = stages.get(key, 0.) + float(value)
        row = dict(kind=kind, index=index, warmup=warmup, wall_s=wall,
                   elapsed_s=metric(r'Elapsed: ([\d.]+)'),
                   primary_s=metric(r'primary=([\d.]+) s'), secondary_s=metric(r'secondary=([\d.]+) s'),
                   histories=metric(r'Histories: (\d+)'), energy_error=metric(r'Energy balance error: ([\d.eE+-]+)'),
                   compaction_s=sum(map(float, re.findall(r'compaction_s=([\d.eE+-]+)', text))),
                   stages=stages, audit=audit, quality=q, dose_sha256=sha(dest / 'dose.raw'))
        if reference is not None:
            if dose.shape != reference.shape:
                raise RuntimeError('Dose grid mismatch')
            row.update(audit_identical=audit == reference_audit,
                       max_diff_pct_peak=float(abs(dose-reference).max()/reference.max()*100))
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k not in ('quality', 'stages', 'audit')}), flush=True)


if __name__ == '__main__':
    main()
