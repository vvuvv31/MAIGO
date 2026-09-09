"""Sequential local GPU smoke comparison; not a Gamma acceptance test."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import argparse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmark/benchmark20260909/ct_runtime_64_chunk64/material_patient_20022516_64.yaml'
BIN = ROOT / 'build/oneapi-nvidia-electron-hot-metadata/carbon_mc'
DEST = ROOT / 'benchmark/benchmark20260909/short_range_64_comparison'

def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def main():
    global DEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, default=DEST)
    DEST = parser.parse_args().output_directory.resolve()
    subprocess.run(['python3', 'tools/verify_schneider_v2_1_data.py'], cwd=ROOT, check=True)
    DEST.mkdir(exist_ok=True)
    env = dict(os.environ, ONEAPI_DEVICE_SELECTOR='cuda:*')
    env['LD_LIBRARY_PATH'] = '/home/wuwei/sycl_workspace/llvm/build/install/lib:' + env.get('LD_LIBRARY_PATH', '')
    manifest = DEST / 'execution.json'
    results = json.loads(manifest.read_text()) if manifest.exists() else []
    for name, threshold in [('off', 0), ('mm010', 0.1), ('mm025', 0.25)]:
        directory = DEST / name
        previous = next((r for r in results if r['name'] == name), None)
        if previous:
            assert previous['binary_sha256'] == sha(BIN)
            assert previous['config_sha256'] == sha(directory / 'gpu.yaml')
            if (directory / 'out/gpu/quality_report.json').exists():
                check_quality(directory)
                continue
            # Preserve failed startup evidence before retrying missing outputs.
            (directory / 'startup_failure.json').write_text(json.dumps(previous, indent=2)+'\n')
            (directory / 'run.log').rename(directory / 'startup_failure.log')
            results.remove(previous)
        directory.mkdir(exist_ok=True)
        # Config defaults are repository-relative even when explicit inputs are absolute.
        data_link = directory / 'data'
        if not data_link.exists():
            data_link.symlink_to(ROOT / 'data', target_is_directory=True)
        lines = [line for line in BASE.read_text().splitlines() if not line.startswith(('validation_output_directory:', 'material_electron_short_range_mm:'))]
        lines += [f'validation_output_directory: {directory}', f'material_electron_short_range_mm: {threshold}']
        config = directory / 'gpu.yaml'
        config.write_text('\n'.join(lines) + '\n')
        record = dict(name=name, threshold_mm=threshold, binary_sha256=sha(BIN), config_sha256=sha(config), histories=64)
        print('START', name, flush=True)
        start = time.monotonic()
        with (directory / 'run.log').open('w') as log:
            process = subprocess.run([str(BIN), '--config', str(config), '--device', 'cuda'], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=process.returncode, wall_seconds=time.monotonic()-start)
        record['reports'] = [str(p) for p in directory.rglob('quality_report.json')]
        results.append(record)
        (DEST / 'execution.json').write_text(json.dumps(results, indent=2)+'\n')
        print('FINISH', record, flush=True)
        if process.returncode not in (0, 1):
            raise SystemExit(process.returncode)
        check_quality(directory)

def check_quality(directory):
    quality = json.loads((directory / 'out/gpu/quality_report.json').read_text())
    codes = [item['code'] for item in quality['failures']]
    if codes != ['unvalidated_material_electron_response'] or quality['queue_overflow_count'] != 0:
        raise RuntimeError(f'Unexpected quality failures: {codes}')
    if abs(quality['voxel_to_ingrid_ratio'] - 1) > 1e-3:
        raise RuntimeError('Voxel energy closure failed')

if __name__ == '__main__':
    main()
