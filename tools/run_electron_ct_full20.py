"""Full-history electron-response experiment versus a frozen historical full20.

Reuses exact old shard sources. Only the explicit smoke response differs.
No production acceptance, fitted scale, or reuse of overflow-contaminated dose.
"""
import argparse
import csv
import json
import re
import io
from pathlib import Path

import numpy as np
import yaml

from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import config_write, sha
from evaluate_topas10x_gpu_gamma import pass_mask


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def half_sources(m, out, chunk, maximum_step=None):
    """Join omitted spots by identity; require an exact half for every spot."""
    rows, counts, configs = {}, {}, []
    for t in m['shards']:
        cfg = yaml.safe_load((Path(t['directory'])/'config.yaml').read_text())
        configs.append(cfg)
        subtotal = 0
        with Path(cfg['tps_spots_file']).open() as f:
            for row in csv.DictReader(f):
                key = row['spot_id']
                identity = {k:v for k,v in row.items() if k != 'weight'}
                n = int(row['weight'])
                if n < 0 or (key in rows and rows[key] != identity):
                    raise ValueError('Invalid/inconsistent spot identity')
                rows[key] = identity
                counts[key] = counts.get(key, 0)+n
                subtotal += n
        if subtotal != t['histories']:
            raise ValueError('Historical spot counts mismatch')
    if sum(counts.values()) != m['histories'] or any(n % 2 for n in counts.values()):
        raise ValueError('Cannot halve every spot exactly')
    varying = {'tps_spots_file', 'number_of_histories', 'random_seed', 'dose_to_medium_name'}
    canonical = {k:v for k,v in configs[0].items() if k not in varying}
    if any({k:v for k,v in c.items() if k not in varying} != canonical for c in configs):
        raise ValueError('Historical shards differ beyond source allocation')
    sources = []
    directory = out/'half_sources'
    directory.mkdir(exist_ok=True)
    for i in range(10):
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=[*next(iter(rows.values())), 'weight'])
        writer.writeheader()
        total = 0
        for key, identity in rows.items():
            n = counts[key]//2//10 + int(i < (counts[key]//2)%10)
            if n: writer.writerow(dict(identity, weight=n))
            total += n
        spots = directory/f'spots_{i+1:02d}.csv'
        expected = text.getvalue().encode()
        if spots.exists() and spots.read_bytes() != expected:
            raise ValueError('Half source changed during resume')
        if not spots.exists(): spots.write_bytes(expected)
        cfg = dict(configs[i], number_of_histories=total, tps_spots_file=str(spots), history_chunk_size=chunk)
        if maximum_step is not None:
            if not 0 < maximum_step <= float(configs[i]['maximum_step_mm']):
                raise ValueError('Only positive step refinement is supported')
            cfg['maximum_step_mm']=maximum_step
        source = directory/f'config_{i+1:02d}.yaml'
        if source.exists() and yaml.safe_load(source.read_text()) != cfg:
            raise ValueError('Half config changed during resume')
        if not source.exists(): config_write(source, cfg)
        sources.append(source)
    return sources


def full(old, out, binary, joint, half=False, chunk=131072, maximum_step=None):
    m = json.loads((old/'manifest.json').read_text())
    execution = json.loads((old/'execution.json').read_text())
    if execution['status'] != 'complete' or len(m['shards']) != 20:
        raise ValueError('Require a completed historical full20')
    if sum(t['histories'] for t in m['shards']) != m['histories'] or sum(t['histories'] for t in execution['completed']) != m['histories']:
        raise ValueError('Historical counts do not close')
    if sha(old/'gpu_sum.raw') != execution['aggregate_sha256']:
        raise ValueError('Historical aggregate SHA mismatch')
    if sha(old/'topas_sum.raw') != m['reference_sum_sha256']:
        raise ValueError('Reference SHA mismatch')
    pins = {str(p): sha(p) for p in [old/'manifest.json', old/'execution.json',
            old/'gpu_sum.raw', old/'gpu_sum.mhd', old/'topas_sum.raw', binary, joint,
            joint.with_suffix('.metadata.json')]}
    for t in m['shards']:
        cfg = Path(t['directory'])/'config.yaml'
        pins[str(cfg)] = sha(cfg)
        for value in yaml.safe_load(cfg.read_text()).values():
            if isinstance(value, str) and Path(value).is_file():
                pins[value] = sha(value)
    out.mkdir(parents=True, exist_ok=True)
    if maximum_step is not None and not half:raise ValueError('Step refinement requires explicit half mode')
    sources = half_sources(m, out, chunk, maximum_step) if half else [Path(t['directory'])/'config.yaml' for t in m['shards']]
    requested_histories = m['histories']//2 if half else m['histories']
    new_label = 'new_half10' if half else 'new_full20'
    dose_scale = m['histories']/requested_histories
    for source in sources:
        pins[str(source)] = sha(source)
        spots = Path(yaml.safe_load(source.read_text())['tps_spots_file'])
        pins[str(spots)] = sha(spots)
    freeze = out/'inputs.json'
    if freeze.exists():
        if json.loads(freeze.read_text()) != pins: raise ValueError('Resume inputs changed')
    else: save(freeze, pins)
    if (out/'gamma.json').exists(): raise ValueError('Completed result exists; refuse overwrite')
    state = dict(status='running_experiment', production_accepted=False,
                 requested_histories=requested_histories, normalization_scale=dose_scale,
                 completed=[], overflow_excluded=[])
    total = np.zeros(m['gpu_shape_zyx'], dtype=np.float64)
    geometry = None

    def task(source, dest, depth=0):
        nonlocal geometry
        report = dest/'run_report.json'
        if not report.exists() and not (dest/'quality_report.json').exists():
            try:
                run(source, dest, binary, joint)
            except RuntimeError as error:
                if not str(error).startswith('OVERFLOW:'): raise
        q = json.loads((dest/'quality_report.json').read_text())
        if q['queue_overflow_count'] or q['queue_overflow_energy_MeV']:
            state['overflow_excluded'].append(str(dest))
            save(out/'execution.json', state)
            if depth >= 5: raise ValueError('Overflow after five subdivisions')
            cfg = yaml.safe_load(source.read_text())
            with Path(cfg['tps_spots_file']).open() as f: rows = list(csv.DictReader(f))
            for half in (0, 1):
                counts = [int(r['weight'])//2 + (int(r['weight'])%2 if half else 0) for r in rows]
                if not sum(counts): continue
                sub = dest/f'split_{half}'
                sub.mkdir(exist_ok=True)
                if not (sub/'source.yaml').exists():
                    with (sub/'spots.csv').open('x') as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                        for row, n in zip(rows, counts):
                            if n: w.writerow(dict(row, weight=n))
                    config_write(sub/'source.yaml', dict(cfg, number_of_histories=sum(counts),
                        tps_spots_file=str(sub/'spots.csv'), random_seed=cfg['random_seed']+1000000007+half))
                task(sub/'source.yaml', sub/'run', depth+1)
            return
        r = json.loads(report.read_text())
        if r['binary_sha256'] != pins[str(binary)] or sha(dest/'dose.raw') != r['dose_sha256']:
            raise ValueError('Shard binary/dose pin mismatch')
        for path, pin in r['inputs'].items():
            if sha(path) != pin: raise ValueError('Shard input changed: '+path)
        cfg = yaml.safe_load(source.read_text())
        if r['histories'] != cfg['number_of_histories']:
            raise ValueError('Shard count mismatch')
        if q['accepted'] or [f['code'] for f in q['failures']] != ['unvalidated_electron_joint_response']:
            raise ValueError('Unexpected quality status')
        ledger = json.loads((dest/'energy_ledger.json').read_text())
        d = ledger['electron_joint_response']
        if not d['patient_experiment'] or d['ordered_path_replays'] <= 0 or d['domain_misses'] or d['invalid_marches']:
            raise ValueError('Response not exercised cleanly')
        text = re.sub(r'ElementDataFile\s*=.*', 'ElementDataFile = gpu_sum.raw', (dest/'dose.mhd').read_text())
        old_geometry = re.sub(r'ElementDataFile\s*=.*', 'ElementDataFile = gpu_sum.raw', (old/'gpu_sum.mhd').read_text())
        if text != old_geometry: raise ValueError('New geometry differs from historical GPU geometry')
        if geometry is not None and geometry != text: raise ValueError('Shard geometry differs')
        geometry = text
        a = np.fromfile(dest/'dose.raw', '<f4').reshape(total.shape)
        if not np.isfinite(a).all() or np.any(a < 0): raise ValueError('Invalid dose')
        total[:] += a
        state['completed'].append(dict(directory=str(dest), histories=r['histories'],
            dose_sha256=r['dose_sha256'], seconds=r['seconds']))
        save(out/'execution.json', state)
        print('Completed', len(state['completed']), 'leaves;', sum(x['histories'] for x in state['completed']), '/', requested_histories, flush=True)

    try:
        for index, source in enumerate(sources, 1):
            task(source, out/f'shard_{index:02d}')
        if sum(x['histories'] for x in state['completed']) != requested_histories:
            raise ValueError('Incomplete full-history aggregate')
        for path, pin in pins.items():
            if sha(path) != pin: raise ValueError('Frozen input changed: '+path)
        total.astype('<f4').tofile(out/'gpu_sum.raw')
        (out/'gpu_sum.mhd').write_text(geometry)
        state.update(status='complete_experiment', aggregate_sha256=sha(out/'gpu_sum.raw'))
        save(out/'execution.json', state)
        ref = np.fromfile(old/'topas_sum.raw', '<f4').reshape(m['topas_shape_zyx'])
        pts = np.argwhere(ref >= .1*ref.max())
        results = {}
        for label, path in [('old_full20', old/'gpu_sum.raw'), (new_label, out/'gpu_sum.raw')]:
            a = np.fromfile(path, '<f4').reshape(m['gpu_shape_zyx'])
            if m['mapping'] == 'packed_xneg': a = np.flip(a.transpose(1,2,0), axis=2)
            elif m['mapping'] != 'native': raise ValueError('Unknown mapping')
            if a.shape != ref.shape: raise ValueError('Reference geometry shape mismatch')
            if label == new_label: a = a.astype(np.float64)*dose_scale
            results[label] = {}
            for dd, dta in [(1,1), (3,0), (3,3), (2,2)]:
                for local in (False, True):
                    key = f'{"local" if local else "global"}_{dd}{dta}'
                    value = 100*float(pass_mask(a, ref, pts, np.array(m['spacing_zyx']), dd, dta, local).mean())
                    results[label][key] = value
                    print(label, key, value, flush=True)
        save(out/'gamma.json', dict(case=m['case'], histories=requested_histories, status='EXPERIMENT_NOT_PRODUCTION',
            results=results, delta_percentage_points={k:results[new_label][k]-results['old_full20'][k] for k in results['old_full20']},
            reference_sha256=m['reference_sum_sha256'], mask_voxels=len(pts), scale=dose_scale,
            method='Full reference >=10% mask; 0.5mm lattice + trilinear; no fitting; 30 same-voxel',
            limitation='Historical full20 comparison, not a same-binary isolated A/B; response remains unvalidated'))
    except BaseException as error:
        state.update(status='failed_experiment', error=str(error)); save(out/'execution.json', state); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['old', 'out', 'binary', 'joint']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--half', action='store_true')
    p.add_argument('--chunk', type=int, default=131072)
    p.add_argument('--maximum-step-mm',type=float)
    a = p.parse_args(); full(a.old.resolve(), a.out.resolve(), a.binary.resolve(), a.joint.resolve(), a.half, a.chunk,a.maximum_step_mm)
