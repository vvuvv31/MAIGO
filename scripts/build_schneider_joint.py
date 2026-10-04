#!/usr/bin/env python3
"""Generate one native joint operator per Schneider composition (density-scaled pilot)."""
import argparse, concurrent.futures, json, os, re, subprocess, sys, time, struct
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
from electron_package_common import save
from electron_joint_compiler import compile_family, load_model, DTYPE, extract, basis, STEP, GRAPH

BINARY = ROOT / 'out/electron_joint_family_20261003/gs_displacement/topas'
INDEX = ROOT / 'benchmark/protonCT/RT07575_minibeam_GPU/response_extension/index.json'

def native(d, table, source, energy, histories, seed):
    if (d / 'status.json').exists():
        prior = json.loads((d / 'status.json').read_text())
        assert prior['accepted'] and prior['seed'] == seed and (prior['histories'] == histories) and (prior['energy_MeV'] == energy) and (prior['table'] == table)
        return prior
    d.mkdir(parents=True, exist_ok=False)
    text = (ROOT / 'out/electron_joint_family_20261003/native065_displacement/train/topas.txt').read_text()
    name = re.search('s:Ge/ROI/Material = "([^"]+)"', source)[1]
    text = '\n'.join((l for l in text.splitlines() if ':Ma/' not in l)) + '\n'
    text = text.replace('ResponseSection1DensityMid9', name)
    text += '\n'.join((l for l in source.splitlines() if ':Ma/' in l)) + '\n'
    text = re.sub('i:Ts/NumberOfThreads = \\d+', 'i:Ts/NumberOfThreads = 2', text)
    text = re.sub('i:Ts/Seed = \\d+', f'i:Ts/Seed = {seed}', text)
    text = re.sub('d:So/Beam/BeamEnergy = .*', f'd:So/Beam/BeamEnergy = {energy:.17g} MeV', text)
    text = re.sub('i:So/Beam/NumberOfHistoriesInRun = \\d+', f'i:So/Beam/NumberOfHistoriesInRun = {histories}', text)
    (d / 'topas.txt').write_text(text)
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data', LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    start = time.perf_counter()
    with (d / 'topas.log').open('x') as log:
        r = subprocess.run([str(BINARY), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (d / 'topas.log').read_text()
    counts = re.findall('Particle source Beam: Total number of histories: (\\d+)', log)
    unscored = sum(map(float, re.findall('Total amount of energy not scored for this reason: ([\\d.eE+-]+) MeV', log)))
    path = d / 'steps.phsp'
    accepted = r.returncode == 0 and counts and (int(counts[-1]) == histories) and (unscored == 0) and path.exists()
    result = dict(name=d.name, exit_code=r.returncode, accepted=bool(accepted), histories=histories, seed=seed, seconds=time.perf_counter() - start, energy_MeV=energy, particle='e-', geometry='sphere_in_vacuum', radius_mm=10000.0, table=table, records_include_vacuum=True, requested_max_step_mm=0.05, density_g_cm3=float(re.search('d:Ma/[^\\n]+/Density = (\\S+) g/cm3', text)[1]), production_cut_mm=0.05, path=str(path), unscored_MeV=unscored, ct_acceptance=False)
    save(d / 'status.json', result)
    assert accepted, result
    return result

def build_one(task):
    section, entry, output, histories = task
    d = output / f'section{section:02d}'
    if (d / 'complete.json').exists():
        done = json.loads((d / 'complete.json').read_text())
        assert done['native_train']['histories'] == histories and done['material_identity'] == entry['material_identity']
        return done
    d.mkdir(exist_ok=True)
    table = Path(entry['path']).parent.name
    path = Path(entry['path'])
    path = path if path.is_absolute() else INDEX.parent / path
    template = path.parent / 'topas.txt'
    if not template.exists():
        template = ROOT / 'data/ct_electron/carbon' / table / 'topas.txt'
    source = template.read_text()
    identity = entry['material_identity']
    rho = identity['runtime_density_g_cm3']
    assert float(re.search('d:Ma/[^\\n]+/Density = (\\S+) g/cm3', source)[1]) == rho
    assert abs(float(re.search('d:Ma/[^\\n]+/MeanExcitationEnergy = (\\S+) eV', source)[1]) - identity['composition']['mean_excitation_energy_eV']) < 1e-10
    seed = 110400001 + section * 1000
    train = native(d / 'train', table, source, 0.65, histories, seed)
    test = native(d / 'test', table, source, 0.4, 1000, seed + 1)
    save(d / 'test_audit.json', extract(test)['audit'])
    full = d / 'full'
    if not (full / 'report.json').exists():
        compile_family(train, full, seed + 2)
    compact = d / 'compact'
    if not (compact / 'report.json').exists():
        with (d / 'compact.log').open('x') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/compact_electron_joint.py'), '--model', str(full), '--out', str(compact)], stdout=log, stderr=subprocess.STDOUT, check=True)
    models, report = load_model(compact)
    raw = np.memmap(train['path'], DTYPE, mode='r')
    ids = np.flatnonzero(raw['parent'] != 0)
    allmodels, _ = load_model(full)
    g = allmodels['graph']
    photon = np.zeros(len(ids), bool)
    active = np.flatnonzero(g['pdg'] == 22)
    while len(active):
        photon[active] = True
        following = np.unique(np.r_[g['next'][active], g['child'][active].ravel()])
        following = following[following >= 0]
        active = following[~photon[following]]
    g['next'][~photon & g['inside'] & (g['pdg'] == 11) & (g['fraction'][:, 1] > 0)] = -2
    reached = np.zeros(len(ids), bool)
    active = np.unique(np.r_[allmodels['continuation']['child'].ravel(), allmodels['entry']['child'].ravel()])
    active = active[active >= 0]
    while len(active):
        reached[active] = True
        following = np.unique(np.r_[g['next'][active], g['child'][active].ravel()])
        following = following[following >= 0]
        active = following[~reached[following]]
    selected = np.flatnonzero(reached)
    remap = np.full(len(ids), -1, np.int32)
    remap[selected] = np.arange(len(selected))
    for k, v in g.items():
        c = v[selected].copy()
        if k in ['child', 'next']:
            c = np.where(c >= 0, remap[np.maximum(c, 0)], c)
        assert np.array_equal(c, models['graph'][k]), k
    pre = raw['direction'][ids[selected]].copy()
    del raw, allmodels, g
    for model in models.values():
        if not isinstance(model, dict) or 'child' not in model:
            continue
        count = model['child'].shape[1]
        assert count <= 2, ('too many physical children', section, count)
        if count < 2:
            for key in ['child', 'childpdg', 'childdir']:
                if key in model:
                    pads = [(0, 0)] * model[key].ndim
                    pads[1] = (0, 2 - count)
                    model[key] = np.pad(model[key], pads, constant_values=-1 if key == 'child' else 0)
            model['fraction'] = np.pad(model['fraction'], ((0, 0), (0, 2 - count)))
            if 'channel_cdf' in model:
                model['channel_cdf'] = np.pad(model['channel_cdf'], ((0, 0), (0, 2 - count)), constant_values=1)
    controls = []
    terminal = None
    endpoint_energies = [5e-05, 0.0001]
    if models['continuation']['edges'][0] <= 0.0001:
        endpoint_energies.append(1e-05)
    for j, energy in enumerate(endpoint_energies):
        for k in range(2):
            ctl = native(d / f'endpoint{j}_{k}', table, source, energy, 1000, seed + 10 + 2 * j + k)
            audit = extract(ctl)['audit']
            r = np.memmap(ctl['path'], DTYPE, mode='r')
            assert len(r) == 1000 and np.all(r['parent'] == 0) and np.all(r['pdg'] == 11) and np.all(r['step'] == 1)
            assert np.all(r['remaining'] == 0) and np.all(r['dep'] == r['energy'])
            extent = float(np.linalg.norm(r['post'] - r['pre'], axis=1).max())
            assert extent <= 0.05 and r['length'].max() <= 0.05
            controls.append(dict(source=ctl, audit=audit, max_chord_mm=extent))
            if j == len(endpoint_energies) - 1 and k == 0:
                terminal = np.array(r)
    r = terminal
    n = len(r)
    m = models['continuation']
    e = float(r['energy'][0])
    assert e < m['edges'][0]
    u, v = basis(r['direction'])
    frame = np.stack([u, v, r['direction']], axis=1)
    values = dict(cdf=np.arange(1, n + 1) / n, delta=np.einsum('nij,nj->ni', frame, r['post'] - r['pre']), postdir=np.einsum('nij,nj->ni', frame, r['post_direction']), length=r['length'], source_direction=r['direction'], source_energy=r['energy'], child=np.full((n, 2), -1, np.int32), childpdg=np.zeros((n, 2), np.int32), childdir=np.zeros((n, 2, 3)), fraction=np.column_stack([np.ones(n), np.zeros((n, 3))]), channel_cdf=np.ones((n, 4)))
    for k in values:
        m[k] = np.concatenate([values[k], m[k]])
    m['edges'] = np.r_[e, m['edges']]
    m['offsets'] = np.r_[0, m['offsets'] + n]
    m['nearest'] = np.r_[0, m['nearest'] + 1]
    steps = np.zeros(len(m['cdf']), STEP)
    for k in ['cdf', 'delta', 'length', 'fraction', 'child']:
        steps[k] = m[k]
    steps['post'] = m['postdir']
    steps['source'] = m['source_direction']
    steps['energy'] = m['source_energy']
    graph = models['graph']
    gr = np.zeros(len(graph['pdg']), GRAPH)
    for k in graph:
        if k != 'energy':
            gr[k] = graph[k]
    bins = len(m['nearest'])
    out = d / 'operator.bin'
    with out.open('xb') as f:
        f.write(struct.pack('<8sdIIII', b'JTFAM002', 0.4, len(steps), 0, len(gr), bins))
        for a in [m['edges'].astype('<f8'), m['offsets'].astype('<u4'), m['nearest'].astype('<u4'), steps, gr, pre.astype('<f8')]:
            f.write(a.tobytes())
    done = dict(section=section, table=table, density_g_cm3=rho, material_identity=identity, native_train=train, native_test=test, native_cutoff_controls=controls, bytes=out.stat().st_size, continuing=len(steps), graph=len(gr), maximum_native_step_mm=0.05, density_scaling=True, status='EXPERIMENTAL_DENSITY_SCALED_NOT_ACCEPTED')
    save(d / 'complete.json', done)
    print('COMPLETE', section, done['bytes'], flush=True)
    return done

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--histories', type=int, default=10000)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--sections', type=int, nargs='*')
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(exist_ok=True, parents=True)
    tables = json.loads(INDEX.read_text())['tables']
    anchors = []
    for section in range(25):
        anchors.append(min((t for t in tables if t['material_identity']['section_id'] == section), key=lambda t: t['material_identity']['runtime_density_g_cm3']))
    save(a.out / 'anchors.json', dict(anchors=anchors, histories_per_material=a.histories, anchor_choice='Minimum indexed density per material; density scaling never enlarges a step for covered response nodes'))
    sections = a.sections if a.sections is not None else list(range(25))
    tasks = [(i, anchors[i], a.out, a.histories) for i in sections]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(build_one, tasks):
            results.append(result)
    if sections != list(range(25)):
        return
    coverage = []
    for t in tables:
        identity = t['material_identity']
        section = identity['section_id']
        rho = identity['runtime_density_g_cm3']
        scale = results[section]['density_g_cm3'] / rho
        assert 0 < scale <= 1.0000001 and 0.05 * scale <= 0.25
        coverage.append(dict(section=section, density_g_cm3=rho, spatial_scale=scale, max_charged_step_mm=0.05 * scale))
    save(a.out / 'node_coverage.json', coverage)
    path = a.out / 'schneider_joint.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<8sI', b'MAIJNB02', 25))
        for i, r in enumerate(results):
            payload = a.out / f'section{i:02d}' / 'operator.bin'
            f.write(struct.pack('<IddQ', i, r['density_g_cm3'], 0.05, payload.stat().st_size))
            with payload.open('rb') as source:
                while (block := source.read(1 << 20)):
                    f.write(block)
    save(a.out / 'manifest.json', dict(status='EXPERIMENTAL_ALL_SCHNEIDER_DENSITY_SCALING', bytes=path.stat().st_size, materials=25, table_nodes=82, results=results))
    print('BANK', path, path.stat().st_size, flush=True)
if __name__ == '__main__':
    main()
