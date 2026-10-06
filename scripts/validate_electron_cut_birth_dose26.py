#!/usr/bin/env python3
"""Analyze completed, matched RT07575 head birth-only experiments.

This does not validate a replacement of shared charged/photon/joint transport.
FP32 GPU doses are summed on the host for analysis; no scorer is changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import validate_em_density_factorization as v


def identical_files(left, right):
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        while True:
            x, y = a.read(1048576), b.read(1048576)
            if x != y:
                raise ValueError(f'Package contents differ: {left}, {right}')
            if not x:
                return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-root', type=Path, required=True)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tags', nargs='+', default=['s1', 's2', 'h01'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    case = 'proton_head_minibeam'
    members = []
    for tag in a.tags:
        old = a.control_root / 'runs' / f'{case}_continuous_{tag}'
        new = a.candidate_root / 'runs' / f'{case}_birth26v3_{tag}'
        configs = [yaml.safe_load((d / 'case.yaml').read_text()) for d in [old, new]]
        ignored = {'em_package_file', 'em_delta_moments_file',
                   'spot_weights_file', 'material_electron_birth_file'}
        differences = {k: [c.get(k) for c in configs]
                       for k in configs[0].keys() | configs[1].keys()
                       if k not in ignored and configs[0].get(k) != configs[1].get(k)}
        if differences:
            raise ValueError(f'Control physics differs: {differences}')
        for key in ['em_package_file', 'em_delta_moments_file']:
            identical_files(configs[0][key], configs[1][key])
        if not np.array_equal(*[np.loadtxt(c['spot_weights_file']) for c in configs]):
            raise ValueError('Spot weights differ')
        with Path(configs[1]['material_electron_birth_file']).open('rb') as f:
            if f.read(8) != b'MAIBCA26':
                raise ValueError('Candidate is not the cut-conditioned birth model')
        records = [json.loads((d / 'execution.json').read_text()) for d in [old, new]]
        if not all(r['runtime_pass'] for r in records):
            raise ValueError('Incomplete or rejected campaign')
        for key in ['seed', 'n', 'h', 'histories', 'spot_allocation']:
            if records[0][key] != records[1][key]:
                raise ValueError(f'Matched campaign differs: {key}')
        members.append(dict(tag=tag, control=str(old), candidate=str(new),
                            control_execution=records[0], candidate_execution=records[1]))
    v.OUT = a.control_root
    old_report, old_dose, mask, spacing = v.analyze(case, 'continuous', a.tags, report_dir=a.out)
    v.OUT = a.candidate_root
    new_report, new_dose, new_mask, new_spacing = v.analyze(case, 'birth26v3', a.tags, report_dir=a.out)
    if not np.array_equal(mask, new_mask) or not np.array_equal(spacing, new_spacing):
        raise ValueError('Gamma domains differ')
    hit = v.evaluator()(new_dose, old_dose, np.argwhere(mask), spacing,
                        3, .2, True, step=.02)
    difference = new_dose[mask] - old_dose[mask]
    local = 100 * abs(difference) / old_dose[mask]
    report = dict(status='BIRTH_ONLY_DOSE_COMPARISON_SHARED83_TRANSPORT',
                  default_promoted=False, complete_electron26_qualified=False,
                  case_identity='RT07575 head', current_TOPAS=old_report,
                  birth26_TOPAS=new_report,
                  against_current_pass_percent=float(100 * hit.mean()),
                  relative_L1=float(abs(difference).sum() / old_dose[mask].sum()),
                  dose_sum_change_percent=float(100 * difference.sum() / old_dose[mask].sum()),
                  local_percent_quantiles={str(q): float(np.percentile(local, q))
                                           for q in [50, 95, 99, 99.9, 100]},
                  accuracy_accepted=bool(new_report['passed']), members=members)
    v.save(a.out / 'comparison.json', report)
    # Preserve the exact mapped absolute doses for independent spatial review.
    np.savez_compressed(a.out / 'dose_arrays.npz', old=old_dose.astype('<f4'),
                        new=new_dose.astype('<f4'), mask=mask, spacing=spacing)
    print(json.dumps({k: value for k, value in report.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    main()
