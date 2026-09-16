#!/usr/bin/env python3
"""Separate NCU runs selected by deterministic generation/round/active identity.

Run after timing (never concurrently). Counters need administrator privileges on
hosts with RmProfilingAdminOnly=1. Does not change driver/clock configuration.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def stages(log):
    rounds = []
    for begin, rnd, active, tail in re.findall(
            r'\[secondary-round\] begin=(\d+) round=(\d+) active=(\d+) blocks256=\d+ (segment|finish_tail)', log):
        rounds.append(dict(begin=int(begin), round=int(rnd), active=int(active), tail=tail))
    if not rounds:
        raise ValueError('Missing secondary round diagnostics')
    generations = list(dict.fromkeys(r['begin'] for r in rounds))
    first = [i for i, r in enumerate(rounds) if r['begin'] == generations[0] and r['tail'] == 'segment']
    if len(first) < 3 or len(generations) < 2:
        raise ValueError('Need early/middle/late and second generation')
    indices = {'secondary_early': first[0], 'secondary_middle': first[len(first)//2],
               'secondary_late': first[-1],
               'secondary_generation2': next(i for i, r in enumerate(rounds) if r['begin'] == generations[1])}
    return {k: dict(launch_skip=i, **rounds[i]) for k, i in indices.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--timing-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path,
                   help='Directory linked as ./data for runtime default files')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--ncu', default='/usr/local/cuda-12.6/bin/ncu')
    p.add_argument('--secondary-entry', choices=('rounded-wrapper', 'named'),
                   default='rounded-wrapper',
                   help='Actual secondary launch entry produced by the selected build')
    p.add_argument('--replay-mode', choices=('kernel', 'application'), default='kernel')
    p.add_argument('--cache-control', choices=('all', 'none'), default='all',
                   help='Keep identical between compared binaries; none preserves application cache state')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = {'primary_steady': dict(launch_skip=10, history_chunk=10)} | stages(a.timing_log.read_text())
    with a.binary.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    (out/'selection.json').write_text(json.dumps(dict(binary_sha256=digest, stages=selected,
        timing_log=str(a.timing_log.resolve()),
        secondary_entry=a.secondary_entry, replay_mode=a.replay_mode,
        cache_control=a.cache_control,
        protocol=(f'{a.repeats} independent application runs; {a.replay_mode} replay; '
                  f'cache-control={a.cache_control}; clocks unchanged')), indent=2))
    env = dict(os.environ, ONEAPI_DEVICE_SELECTOR='cuda:*', CARBON_SECONDARY_TAIL_DIAG='1',
               LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    rows = []
    for repeat in range(a.repeats):
        for name, identity in selected.items():
            dest = out/f'{repeat:02d}_{name}'
            dest.mkdir()
            config = a.config.resolve()
            # All campaign inputs are frozen absolute paths. data supplies defaults.
            data_dir = (a.data_dir.resolve() if a.data_dir else config.parent/'source/data')
            (dest/'data').symlink_to(data_dir, target_is_directory=True)
            if name.startswith('primary'):
                kernel_name_base = 'demangled'
                pattern = 'regex:.*transport_sycl_impl.*nd_item.*'
            elif a.secondary_entry == 'rounded-wrapper':
                # The NVIDIA backend exposes the actual SYCL wrapper under this
                # mangled typeinfo name. Matching the named device function can
                # silently profile a different entry or no entry at all.
                kernel_name_base = 'mangled'
                pattern = ('regex:_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIN6carbon'
                           '30CarbonSecondaryTransportKernel.*')
            else:
                kernel_name_base = 'mangled'
                pattern = 'regex:_ZTSN6carbon30CarbonSecondaryTransportKernel.*'
            cmd = [a.ncu, '--kernel-name-base', kernel_name_base, '--kernel-name', pattern,
                   '--launch-skip', str(identity['launch_skip']), '--launch-count', '1', '--kill', 'yes',
                   '--clock-control', 'none', '--replay-mode', a.replay_mode,
                   '--cache-control', a.cache_control, '--export', 'report']
            for section in ('SpeedOfLight','Occupancy','SchedulerStats','WarpStateStats','LaunchStats',
                            'MemoryWorkloadAnalysis','ComputeWorkloadAnalysis','SourceCounters'):
                cmd += ['--section', section]
            cmd += [str(a.binary.resolve()), '--config', str(config), '--device', 'cuda']
            (dest/'command.json').write_text(json.dumps(cmd, indent=2))
            with (dest/'run.log').open('w') as f:
                subprocess.run(cmd, cwd=dest, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            for page, extra, filename in [('raw',['--csv'],'metrics.csv'),('source',['--print-source','sass'],'sass.txt')]:
                with (dest/filename).open('w') as f:
                    subprocess.run([a.ncu,'--import',str(dest/'report.ncu-rep'),'--page',page]+extra,
                                   stdout=f,stderr=subprocess.STDOUT,check=True)
            raw=list(csv.DictReader((dest/'metrics.csv').open()))
            if len(raw)!=2:
                raise RuntimeError('Expected units row and exactly one matched kernel')
            rows.append(dict(stage=name, repeat=repeat, identity=identity, metrics=raw[1]))
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(name,repeat,flush=True)


if __name__=='__main__':
    main()
