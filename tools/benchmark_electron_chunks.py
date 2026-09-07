"""Local-only fixed-binary chunk sweep; experimental dose, never production."""
import csv
import json
import re
from pathlib import Path
import subprocess
import numpy as np
import yaml
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import config_write, sha


def main():
    root = Path('/mnt/sda/wuwei/electron_chunk_sweep_20260906/20022516')
    root.mkdir(parents=True, exist_ok=False)
    source = Path('/mnt/sda/wuwei/electron_bounds_probe_20260906/20022516/run.yaml')
    binary = Path('build/oneapi-nvidia-electron-bounds/carbon_mc').resolve()
    cfg = yaml.safe_load(source.read_text())
    original_spots = Path(cfg['tps_spots_file'])
    with original_spots.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row['weight'] = str(int(int(row['weight']) / 8 + .5))
    rows = [row for row in rows if int(row['weight']) > 0]
    with (root/'spots.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cfg['number_of_histories'] = sum(int(r['weight']) for r in rows)
    cfg['tps_spots_file'] = str(root/'spots.csv')
    joint = Path(cfg['ct_electron_joint_response_diagnostic_file'])
    pins = {str(p): sha(p) for p in (source, original_spots, binary, root/'spots.csv')}
    result = dict(status='RUNNING_EXPERIMENT', histories=cfg['number_of_histories'],
                  pins=pins, runs=[])
    baseline = None
    for i, chunk in enumerate((16384, 32768, 65536, 131072, 16384)):
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if gpu:
            raise RuntimeError('GPU is occupied: '+gpu)
        cfg['history_chunk_size'] = chunk
        config = root/f'input_{i}_{chunk}.yaml'
        config_write(config, cfg)
        out = root/f'run_{i}_{chunk}'
        run(config, out, binary, joint)
        log = (out/'run.log').read_text()
        ledger = json.loads((out/'energy_ledger.json').read_text())
        d = ledger['electron_joint_response']
        if d['domain_misses'] or d['invalid_marches']:
            raise RuntimeError('Invalid response coverage/geometry')
        dose = np.fromfile(out/'dose.raw', dtype='<f4').astype(np.float64)
        if not np.isfinite(dose).all() or (dose < 0).any():
            raise RuntimeError('Invalid dose')
        if baseline is None:
            baseline = dose.copy()
        diff = dose-baseline
        record = dict(chunk=chunk, directory=str(out),
            seconds=float(re.search(r'Elapsed: ([\d.e+-]+) s', log)[1]),
            throughput=float(re.search(r'Throughput: ([\d.e+-]+)', log)[1]),
            primary_seconds=float(re.search(r'Kernel time: primary=([\d.e+-]+)', log)[1]),
            secondary_seconds=float(re.search(r' s secondary=([\d.e+-]+)', log)[1]),
            dose_max_diff_over_peak=float(np.max(np.abs(diff))/baseline.max()),
            dose_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(baseline)),
            dose_sum_ratio=float(dose.sum()/baseline.sum()),
            energy_deposited_MeV=ledger['E_dep_MeV'], electron=d,
            telemetry=subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,memory.used', '--format=csv,noheader'], text=True).strip())
        result['runs'].append(record)
        (root/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(record), flush=True)
    for p, pin in pins.items():
        if sha(p) != pin:
            raise RuntimeError('Changed input: '+p)
    result['status'] = 'COMPLETE_EXPERIMENT_NOT_PRODUCTION'
    (root/'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
