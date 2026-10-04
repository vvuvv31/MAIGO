#!/usr/bin/env python3
"""Prepare/run isolated TOPAS primary-source extraction (local Slurm worker)."""
import argparse, concurrent.futures, importlib.util, json, os, subprocess, uuid
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')

def template_module():
    spec = importlib.util.spec_from_file_location('templates', REPO / 'extensions/tools/schneider/generate_v2_manifest.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def prepare(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'campaign.json').exists():
        raise RuntimeError('Use a new campaign directory; existing manifest is immutable')
    mod = template_module()
    tasks = []
    coverage = json.loads(args.coverage.read_text()) if args.coverage else None
    for z, sym, mat, element, rho, hlz in mod.TARGETS:
        energies = args.energies if coverage is None else sorted(set((round(p['energy'] - 0.01, 5) for p in coverage['positive_rate_without_events'] if p['target_z'] == z)))
        for energy in energies:
            case = root / 'cases' / f'{sym}_E{energy:g}'
            case.mkdir(parents=True)
            text = mod.TEMPLATE.format(mat_name=mat, hlz=hlz, elem_name=element, density=rho, part_name='proton', beam_energy=energy, histories=args.histories, seed=26092600 + len(tasks), pz=1, pa=1)
            text += '\ni:Ts/NumberOfThreads = 1\ni:Ts/ShowHistoryCountAtInterval = 1000000\n'
            (case / 'run.txt').write_text(text)
            tasks.append(dict(case=str(case.relative_to(root)), target_z=z, energy_MeV=energy, histories=args.histories, input_text=text))
    dicom = Path('/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box')
    dump(root / 'campaign.json', dict(schema=1, uuid=str(uuid.uuid4()), projectile={'z': 1, 'a': 1}, requested_source_energy_MeV=[0.1, 250.0], guard_energy_MeV=max((t['energy_MeV'] for t in tasks)), tasks=tasks, final_state_only_xs_factor=args.final_state_xs_factor, physics_modules=['g4em-standard_opt4', 'g4h-phy_QGSP_BIC_HP', 'g4ion-inclxx', 'CarbonInelasticCapturePhysics', 'g4h-elastic_HP', 'g4stopping'], dicom_files=[str(p) for p in sorted(dicom.rglob('*')) if p.is_file()]))
    print(root / 'campaign.json')

def run(args):
    root = args.root.resolve()
    m = json.loads((root / 'campaign.json').read_text())
    topas = args.topas.resolve()

    def task(t):
        case = root / t['case']
        status = case / 'status.json'
        if status.exists():
            old = json.loads(status.read_text())
            if old['returncode'] == 0 and old['topas'] == str(topas) and (old['input_text'] == (case / 'run.txt').read_text()):
                return old
            raise RuntimeError(f'Existing failed/different run: {case}; create an explicit retry campaign')
        if (case / 'run.txt').read_text() != t['input_text']:
            raise RuntimeError('Input changed after prepare')
        env = os.environ.copy()
        env.update(TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data', LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:' + env.get('LD_LIBRARY_PATH', ''), CARBON_CINEL02_OUTPUT_DIR=str(case / 'raw'), CARBON_CINEL02_CAMPAIGN_UUID=m['uuid'], CARBON_CINEL02_RUN_TAG=case.name, CARBON_CINEL02_PRIMARY_ONLY='true', CARBON_CINEL02_OVERWRITE_CAMPAIGN='false')
        env.pop('MAIGO_CINEL_FINAL_STATE_XS_FACTOR', None)
        if m.get('final_state_only_xs_factor', 1) != 1:
            env['MAIGO_CINEL_FINAL_STATE_XS_FACTOR'] = str(m['final_state_only_xs_factor'])
        with open(case / 'topas.log', 'w') as log:
            r = subprocess.run([str(topas), 'run.txt'], cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = dict(t, returncode=r.returncode, topas=str(topas), raw_files=[str(p.relative_to(root)) for p in sorted(case.rglob('worker_*.cinel02'))])
        dump(status, result)
        print(case.name, r.returncode, flush=True)
        return result
    tasks = m['tasks']
    if args.targets:
        tasks = [t for t in tasks if t['target_z'] in args.targets]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(task, tasks))
    if any((r['returncode'] != 0 for r in results)):
        raise RuntimeError('TOPAS extraction failed; inspect case logs')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--energies', type=float, nargs='+', default=[8, 25, 100, 251])
    p.add_argument('--histories', type=int, default=20000)
    p.add_argument('--topas', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--targets', type=int, nargs='+')
    p.add_argument('--coverage', type=Path)
    p.add_argument('--final-state-xs-factor', type=float, default=1)
    a = p.parse_args()
    if a.histories <= 0 or a.workers <= 0 or any((not 0 < e <= 251 for e in a.energies)):
        p.error('Require positive histories/workers and energies within (0, 251] MeV')
    if a.action == 'run' and a.topas is None:
        p.error('--topas is required for run')
    prepare(a) if a.action == 'prepare' else run(a)
if __name__ == '__main__':
    main()
