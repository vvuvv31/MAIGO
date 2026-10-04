#!/usr/bin/env python3
"""Serialize cache validation and the pending C12 200k comparison; never bypass a failed gate."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / 'out/carbon_ct_minibeam_20261002'
OUT = ROOT / 'out/material_prepared_validation'
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_minibeam import integer_fields


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def stage(name, **fields):
    save(OUT / 'status.json', dict(stage=name, **fields))
    print(name, flush=True)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(a, b, log_a, log_b):
    qa, qb = [json.loads((d / 'quality_report.json').read_text()) for d in (a, b)]
    assert all(q['accepted'] and q['queue_overflow_count'] == 0 for q in (qa, qb))
    x, y = [np.memmap(d / 'dose.raw', dtype=np.float32, mode='r') for d in (a, b)]
    assert x.shape == y.shape
    total = delta = peak = maximum = 0.
    for i in range(0, x.size, 1000000):
        u, v = x[i:i+1000000].astype(float), y[i:i+1000000].astype(float)
        assert np.isfinite(u).all() and np.isfinite(v).all() and (u >= 0).all() and (v >= 0).all()
        total += u.sum(); delta += abs(u-v).sum()
        peak = max(peak, float(u.max(initial=0))); maximum = max(maximum, float(abs(u-v).max(initial=0)))
    l1, mx = float(delta / total), maximum / peak
    assert l1 < 1e-6 and mx < 1e-6, (l1, mx)
    ledgers = [integer_fields(json.loads((d / 'energy_ledger.json').read_text())) for d in (a, b)]
    assert ledgers[0] == ledgers[1], [(k, ledgers[0].get(k), ledgers[1].get(k)) for k in ledgers[0].keys() | ledgers[1].keys() if ledgers[0].get(k) != ledgers[1].get(k)]
    em = [re.findall(r'^\[unified-em-audit\] (.*)$', p.read_text(), re.M) for p in (log_a, log_b)]
    assert em[0] and em[0] == em[1]
    return dict(dose_relative_l1=l1, peak_normalized_max_difference=mx, discrete_ledger_exact=True, em_audit_exact=True, quality=qb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-compact-pid', type=int, required=True)
    p.add_argument('--old-parent-pid', type=int, required=True)
    args = p.parse_args()
    OUT.mkdir(exist_ok=False)
    binary = OUT / 'carbon_mc'
    shutil.copy2(ROOT / 'build/carbon_mc', binary)
    cache = (ROOT / 'build/CMakeCache.txt').read_text()
    assert 'CARBON_DOSE_FP32:BOOL=ON' in cache and 'CARBON_DOSE_FP64:BOOL=OFF' in cache
    save(OUT / 'binary.json', dict(sha256=sha(binary), cache_sha256=sha(ROOT / 'build/CMakeCache.txt')))
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/wuwei/sycl_workspace/llvm/build/install/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['CARBON_RUNTIME_BREAKDOWN'] = '1'
    env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS'] = '1'
    stage('waiting_raw_compact_baseline')
    deadline = time.monotonic() + 3*3600
    def wait_until(fn):
        while not fn():
            if time.monotonic() > deadline:
                raise RuntimeError('Validation prerequisite timeout')
            time.sleep(10)
    def stopped():
        p = Path(f'/proc/{args.old_compact_pid}/status')
        return not p.exists() or '\nState:\tZ' in p.read_text()
    wait_until(stopped)
    # The old orchestrator was held before a third costly raw startup. Its
    # completed outputs remain immutable; this driver takes over after checks.
    try:
        os.kill(args.old_parent_pid, signal.SIGTERM)
        os.kill(args.old_parent_pid, signal.SIGCONT)
    except ProcessLookupError:
        pass
    baseline = compare(CAMPAIGN/'full_response_combined/batch000000', CAMPAIGN/'full_response_compact/batch000000',
                       CAMPAIGN/'full_response_combined.log', CAMPAIGN/'full_response_compact.log')
    save(CAMPAIGN/'full_response_handoff.json', dict(histories=1000, **baseline))
    stage('waiting_prepared_carbon_bank')
    carbon_cache = ROOT / 'out/material_prepared_cache/carbon'
    wait_until(lambda: len(list(carbon_cache.glob('*/manifest.sha256'))) == 63)

    def run(name, config, cache_dir, histories=1000, batches=1, output=None):
        stage(name)
        e = dict(env)
        if cache_dir:
            e['CARBON_MATERIAL_PREPARED_DIR'] = str(cache_dir)
        else:
            e.pop('CARBON_MATERIAL_PREPARED_DIR', None)
        d = output or OUT/name
        log = OUT/(name+'.log')
        command = [str(binary), '--config', str(config), '--histories-per-batch', str(histories),
                   '--batch-count', str(batches), '--batch-output', str(d)]
        start = time.monotonic()
        with log.open('x') as stream:
            r = subprocess.run(command, cwd=ROOT, env=e, stdout=stream, stderr=subprocess.STDOUT)
        wall = time.monotonic()-start
        text = log.read_text()
        result = dict(exit_code=r.returncode, wall_seconds=wall, histories=histories, batches=batches,
                      batch_wall_seconds=[float(v) for v in re.findall(r'\[batch-wall\].*seconds=([\d.eE+-]+)', text)],
                      prepared_load_seconds=sum(float(v) for v in re.findall(r'\[material-response-prepared\].*seconds=([\d.eE+-]+)', text)),
                      bank_hits=text.count('[electron-bank-cache] hit'))
        save(OUT/(name+'.json'), result)
        assert r.returncode == 0, (name, r.returncode)
        for i in range(batches):
            q = json.loads((d/f'batch{i:06d}'/'quality_report.json').read_text())
            assert q['accepted'] and q['queue_overflow_count'] == 0
        assert re.findall(r'^Histories: (\d+)$', text, re.M) == [str(histories)]*batches
        return d, log, result

    checks = {}
    d, log, timing = run('carbon_cached_smoke', ROOT/'config/carbon_ct_minibeam.yaml', carbon_cache)
    checks['carbon'] = compare(CAMPAIGN/'full_response_compact/batch000000', d/'batch000000',
                               CAMPAIGN/'full_response_compact.log', log)
    save(OUT/'carbon_cache_check.json', dict(**checks['carbon'], timing=timing))

    stage('prepare_proton_bank')
    proton_cache = ROOT/'out/material_prepared_cache/proton'
    with (OUT/'prepare_proton.log').open('x') as stream:
        subprocess.run([sys.executable, str(ROOT/'scripts/prepare_material_electron.py'), '--index',
                        str(ROOT/'data/ct_electron/proton/index.json'), '--cache', str(proton_cache)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for name, cache_dir in [('proton_ct_minibeam', proton_cache), ('carbon_water_minibeam', None)]:
        d, log, timing = run(name+'_smoke', ROOT/'config'/f'{name}.yaml', cache_dir)
        old = CAMPAIGN/f'regression_{name}_new'
        checks[name] = dict(**compare(old/'batches/batch000000', d/'batch000000', old/'gpu.log', log), timing=timing)
    save(OUT/'checks.json', checks)
    save(CAMPAIGN/'campaign_status.json', dict(stage='gpu_prepared_response_2x100k', accepted=False))
    d, log, timing = run('carbon_2x100k', ROOT/'config/carbon_ct_minibeam.yaml', carbon_cache,
                         histories=100000, batches=2, output=CAMPAIGN/'gpu_batches')
    assert timing['bank_hits'] == 1
    shutil.copy2(log, CAMPAIGN/'gpu_batches.log')
    e = dict(env, PYTHONPATH='/mnt/sda/wuwei/maigo_gamma_python_deps')
    stage('initial_topas_comparison')
    with (CAMPAIGN/'comparison.log').open('x') as stream:
        subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py'))], cwd=ROOT, env=e,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    save(CAMPAIGN/'campaign_status.json', dict(stage='initial_comparison_complete', accepted=False,
         note='Prepared cache validated against original replay; initial 200k dose comparison is not final acceptance',
         binary_sha256=sha(binary), comparison=str(CAMPAIGN/'comparison.json')))
    stage('complete', cache_validation_passed=True, final_dose_acceptance=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if OUT.exists():
            stage('failed', error=str(e))
        raise
