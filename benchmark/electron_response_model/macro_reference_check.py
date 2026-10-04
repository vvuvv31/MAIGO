#!/usr/bin/env python3
"""Compare native sphere histories against the independent full-track pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import extract, metrics, DTYPE
from prototype import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('out/electron_mono_model_20261002/report.json'))
    a = p.parse_args()
    target = a.input/'full_reference_check.json'
    if target.exists():
        raise FileExistsError(target)
    ref = json.loads(a.reference.read_text())['independent_absolute_errors']
    results = []
    for source in json.loads((a.input/'manifest.json').read_text()):
        assert source['table'] == 's1_mid9' and abs(source['energy_MeV']-.65) < 1e-9
        data = extract(source)
        value, sem = metrics(data, data['audit']['actual_energy_MeV'])
        checks = {}
        for i, key in enumerate(['r2', 'z', 'z2', 'peak', 'valley']):
            baseline = ref[key]['reference']
            checks[key] = dict(reference=baseline, native=float(value[i]),
                               relative_difference=float((value[i]-baseline)/baseline),
                               pointwise_relative_95_halfwidth=float(2.093*np.hypot(sem[i], ref[key]['test_block_standard_error'])/baseline))
        raw = np.memmap(source['path'], DTYPE, mode='r')
        steps = np.memmap(source['path'], dtype=np.dtype(dict(names=['length'], formats=['<f8'], offsets=[280], itemsize=312)), mode='r')
        charged = (raw['pdg'] == 11) & (raw['pre_material'] != 0)
        maximum = float(steps['length'][charged].max(initial=0))
        limit = source.get('requested_max_step_mm', 0)
        if limit > 0:
            assert maximum <= limit*(1+1e-6), 'Requested step limiter not actually active'
        results.append(dict(name=source['name'], checks=checks, audit=data['audit'],
                            requested_max_step_mm=limit, measured_max_charged_step_mm=maximum))
        print(json.dumps(results[-1]), flush=True)
        save(target, dict(scope='Native sphere versus native large-box reference; no macro composition',
                          results=results, ct_acceptance=False, script_sha256=sha(Path(__file__)),
                          reference_sha256=sha(a.reference)))


if __name__ == '__main__':
    main()
