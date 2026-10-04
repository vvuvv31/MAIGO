#!/usr/bin/env python3
"""Serialized independent sphere sources and models for composition testing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prototype import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--table', default='s1_mid9')
    p.add_argument('--radius', type=float, default=1.)
    p.add_argument('--max-step', type=float, default=0.)
    p.add_argument('--histories', type=int, default=100000)
    p.add_argument('--seed', type=int, default=1002240001)
    p.add_argument('--reuse-references', type=Path)
    p.add_argument('--reuse-models', type=Path)
    p.add_argument('--energies', type=float, nargs='+', default=[.00001,.0001,.001,.005,.01,.02,.04,.06,.08,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65])
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    status = dict(status='running', pid=os.getpid(), ct_acceptance=False, cases=[],
                  script_sha256=sha(Path(__file__)))
    save(a.out/'status.json', status)
    for i, energy in enumerate(a.energies):
        ref = a.out/f'e{i:02d}_reference'; model = a.out/f'e{i:02d}_model'
        reuse = False
        if a.reuse_references:
            old = a.reuse_references/f'e{i:02d}_reference'
            if (old/'manifest.json').exists():
                previous = json.loads((old/'manifest.json').read_text())
                reuse = len(previous) == 2 and all(c['accepted'] and c['energy_MeV'] == energy and
                    c['radius_mm'] == a.radius and c['table'] == a.table and c['records_include_vacuum'] and
                    c.get('requested_max_step_mm', 0) == a.max_step for c in previous)
                if reuse:
                    ref = old
        reuse_model = False
        if reuse and a.reuse_models:
            old_model = a.reuse_models/f'e{i:02d}_model'
            if (old_model/'report.json').exists():
                report = json.loads((old_model/'report.json').read_text())
                reuse_model = (report['selected'] is not None and
                    report['source_manifest_sha256'] == sha(ref/'manifest.json') and
                    report['model_sha256'] == sha(old_model/'model.npz'))
                if reuse_model:
                    model = old_model
        commands = [
            [sys.executable, str(here/'macro_reference.py'), '--energy', str(energy),
             '--radius', str(a.radius), '--table', a.table, '--histories', str(min(a.histories, 1000) if energy <= .001 else a.histories),
             '--max-step', str(a.max_step),
             '--seed', str(a.seed+2*i), '--out', str(ref)],
            [sys.executable, str(here/'macro_model.py'), '--input', str(ref), '--out', str(model)]]
        case = dict(energy_MeV=energy, reference=str(ref.resolve()), model=str(model.resolve()))
        status['current'] = case; save(a.out/'status.json', status)
        for j, command in enumerate(commands):
            if j == 0 and reuse:
                continue
            if j == 1 and reuse_model:
                continue
            with (a.out/f'e{i:02d}_stage{j}.log').open('x') as f:
                code = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if code:
                status.update(status='failed', exit_code=code, stage=j)
                save(a.out/'status.json', status)
                raise RuntimeError(f'Case {i}, stage {j} failed')
        report = json.loads((model/'report.json').read_text())
        if report['selected'] is None:
            status.update(status='representation_rejected', case=case)
            save(a.out/'status.json', status)
            raise RuntimeError('No representation passed')
        case.update(actual_energy_MeV=report['train_audit']['actual_energy_MeV'],
                    model_sha256=sha(model/'model.npz'), payload_bytes=report['selected']['payload_bytes'])
        status['cases'].append(case)
        save(a.out/'status.json', status)
        print(json.dumps(case), flush=True)
    status.update(status='complete', payload_bytes=sum(x['payload_bytes'] for x in status['cases']))
    save(a.out/'status.json', status)


if __name__ == '__main__':
    main()
