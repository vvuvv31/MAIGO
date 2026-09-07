"""Independent local Slurm EM reference: 100k noise probe, not a new Gamma baseline."""
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from run_topas10x_gpu_benchmark import sha


def main():
    base = Path('/mnt/sda/wuwei/ct_clinical_em_control_20260907')
    repeat = Path('/mnt/sda/wuwei/ct_clinical_em_seed9296301_20260907')
    root = Path('/mnt/sda/wuwei/ct_clinical_em_reference_noise_20260907')
    if subprocess.check_output(['squeue', '-u', 'wuwei', '-h', '-o', '%i'], text=True).strip():
        raise RuntimeError('Review existing allocation first')
    old = json.loads((base / 'comparison.json').read_text())
    if sha(base / 'dose_topas.bin') != old['reference_sha256']:
        raise ValueError('Reference changed')
    source = (base / 'topas.txt').read_text()
    root.mkdir(exist_ok=False)
    edits = {'i:Ts/Seed': '9396301', 'i:Ts/NumberOfThreads': '96',
             'u:So/CarbonPBS/HistoriesScale': '0.3333333333333333',
             's:Sc/OSMK_Dtotal/OutputFile': f'"{root}/dose_topas"'}
    for key, value in edits.items():
        source, n = re.subn(r'^' + re.escape(key) + r'\s*=.*$', key + ' = ' + value, source, flags=re.M)
        if n != 1:
            raise ValueError('Replacement count: ' + key)
    (root / 'topas.txt').write_text(source)
    exe = Path('/home/wuwei/topas/topas-build/topas')
    script = f'''#!/bin/bash
#SBATCH --job-name=ct_em_noise
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output={root}/job_%j.log
#SBATCH --error={root}/job_%j.err
set -euo pipefail
cd {root}
{exe} {root}/topas.txt
'''
    (root / 'run.slurm').write_text(script)
    pins = {str(p): sha(p) for p in (exe, base / 'topas.txt', root / 'topas.txt', root / 'run.slurm')}
    # Inherit and verify geometry/source pins from the completed control.
    for path, expected in json.loads((base / 'manifest.json').read_text())['pins'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Control input changed: ' + path)
        pins[path] = expected
    job = subprocess.check_output(['sbatch', '--parsable', str(root / 'run.slurm')], text=True).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Invalid job id')
    with (root / 'manifest.json').open('x') as stream:
        json.dump(dict(job=job, histories=100000, cpus=96, memory_GiB=64, pins=pins), stream, indent=2)
    print('Submitted local TOPAS', job, flush=True)
    deadline = time.monotonic() + 4000
    while True:
        lines = subprocess.check_output(['sacct', '-j', job, '--format=JobIDRaw,State,ExitCode', '-n', '-P'], text=True)
        rows = [line.split('|') for line in lines.splitlines() if line.startswith(job + '|')]
        if rows and rows[0][1:3] == ['COMPLETED', '0:0']:
            break
        if rows and rows[0][1].split()[0] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'):
            raise RuntimeError(str(rows))
        if time.monotonic() > deadline:
            raise TimeoutError('Inspect saved Slurm job')
        time.sleep(15)
    log = (root / f'job_{job}.log').read_text()
    if 'Finalization:' not in log:
        raise ValueError('Incomplete reference')
    if '1 spots, 100000 histories (skipped 0 zero-weight rows)' not in log:
        raise ValueError('Unexpected reference history allocation')
    header = (root / 'dose_topas.binheader').read_text()
    if [int(re.search(rf'# {axis} in (\d+) bins', header)[1]) for axis in 'XYZ'] != [960, 607, 42]:
        raise ValueError('Unexpected reference grid')
    if sha(base / 'gpu/dose.raw') != old['gpu_sha256']:
        raise ValueError('Baseline GPU dose changed')
    repeated = json.loads((repeat / 'comparison.json').read_text())
    if sha(repeat / 'gpu/dose.raw') != repeated['dose_sha256']:
        raise ValueError('Repeated GPU dose changed')
    for path, expected in pins.items():
        if sha(Path(path)) != expected:
            raise ValueError('Input changed: ' + path)
    shape = (42, 607, 960)
    ref = np.fromfile(base / 'dose_topas.bin', '<f8').reshape(shape)
    low = np.fromfile(root / 'dose_topas.bin', '<f8').reshape(shape)
    a = np.fromfile(base / 'gpu/dose.raw', '<f4').reshape(shape)
    b = np.fromfile(repeat / 'gpu/dose.raw', '<f4').reshape(shape)
    for arr in (ref, low, a, b):
        if not np.isfinite(arr).all() or np.any(arr < 0) or arr.max() <= 0:
            raise ValueError('Invalid dose')
    bands = {}
    for lo, hi in ((.1, .2), (.2, .5), (.5, 1.01)):
        m = (ref >= lo * ref.max()) & (ref < hi * ref.max())
        # Var(R300 - 3*R100) = 4 Var(R300), under independent histories.
        tn = (ref[m] - 3 * low[m]) / (2 * ref[m])
        gn = (a[m].astype(float) - b[m]) / (np.sqrt(2) * ref[m])
        error = (a[m] - ref[m]) / ref[m]
        bands[f'{lo}-{hi}'] = dict(voxels=int(m.sum()), error_rms_pct=float(np.sqrt(np.mean(error**2))*100),
            gpu_noise_pct=float(np.sqrt(np.mean(gn**2))*100), topas_noise_pct=float(np.sqrt(np.mean(tn**2))*100),
            combined_noise_pct=float(np.sqrt(np.mean(gn**2)+np.mean(tn**2))*100))
    output = dict(status='NOISE_ESTIMATE_NOT_NOISE_SUBTRACTED_GAMMA', bands=bands,
        reference_repeat_sha256=sha(root / 'dose_topas.bin'), normalized_sum_ratio=float(3*low.sum()/ref.sum()),
        limitation='Single independent pair per engine; masks and denominators use noisy reference. RMS estimates do not establish absence of systematic errors.')
    with (root / 'analysis.json').open('x') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
