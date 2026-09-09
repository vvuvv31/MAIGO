"""Smoke-only dose and runtime comparison, deliberately not patient Gamma."""
import json
import re
from pathlib import Path
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parent / 'short_range_64_comparison'

def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, default=ROOT)
    ROOT = parser.parse_args().output_directory.resolve()
    execution = {r['name']: r for r in json.loads((ROOT / 'execution.json').read_text())}
    results = []
    baseline = None
    for name in ('off', 'mm010', 'mm025'):
        folder = ROOT / name
        report = json.loads((folder / 'out/gpu/quality_report.json').read_text())
        dose = np.fromfile(folder / 'out/gpu/dose.raw', dtype='<f4').astype(np.float64)
        if baseline is None:
            baseline = dose
        assert dose.shape == baseline.shape and np.isfinite(dose).all()
        log = (folder / 'run.log').read_text()
        kernel = float(re.search(r'Kernel time: primary=([\d.eE+-]+)', log)[1])
        hits = re.search(r'shortcut_packets=(\d+)', log)
        delta = dose-baseline
        results.append(dict(name=name, primary_kernel_seconds=kernel,
            primary_kernel_histories_per_second=64/kernel,
            process_wall_histories_per_second=64/execution[name]['wall_seconds'],
            reported_histories_per_second=float(re.search(r'Throughput: ([\d.eE+-]+)', log)[1]),
            shortcut_packets=int(hits[1]) if hits else (0 if name == 'off' else None),
            failures=report['failures'], overflow=report['queue_overflow_count'],
            physical_relative_energy_residual=report['physical_relative_energy_residual'],
            voxel_to_ingrid_ratio=report['voxel_to_ingrid_ratio'],
            max_absolute_dose_difference_Gy=float(np.abs(delta).max()),
            max_difference_percent_baseline_peak=float(np.abs(delta).max()/baseline.max()*100),
            dose_array_sum_relative_difference=float(delta.sum()/baseline.sum()),
            primary_kernel_speedup=results[0]['primary_kernel_seconds']/kernel if results else 1))
    output = dict(scope='64 histories, single timing trial; NOT throughput or Gamma acceptance', results=results)
    (ROOT / 'comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
