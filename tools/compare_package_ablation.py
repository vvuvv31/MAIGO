"""Run paired local GPU package-code ablation; no physics or Gamma fitting."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import sha


def check_ledger_preservation(before, after):
    keys = {k for k in before if k.startswith('cinel02_')}
    if keys != {k for k in after if k.startswith('cinel02_')}:
        raise RuntimeError('Legacy/shared ledger schema changed')
    for key in keys:
        if key == 'cinel02_species_transport_ledger_MeV':
            a, b = np.asarray(before[key]), np.asarray(after[key])
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise RuntimeError('Invalid shared species ledger')
            if not np.allclose(a, b, rtol=2.e-6, atol=1.e-3):
                raise RuntimeError('Shared species ledger changed')
        elif before[key] != after[key]:
            raise RuntimeError('Retained diagnostic/terminal field changed: '+key)
    for key in ('E_dep_in_grid_MeV', 'E_dep_outside_grid_MeV'):
        if key in before or key in after:
            if key not in before or key not in after or not np.isclose(before[key], after[key], rtol=2.e-6, atol=1.e-3):
                raise RuntimeError('Shared grid ledger changed: '+key)
    return sorted(keys)


def compare(before, after):
    a = np.fromfile(before / 'dose.raw', '<f4')
    b = np.fromfile(after / 'dose.raw', '<f4')
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Invalid paired dose')
    if a.max() <= 0 or np.any(a < 0) or np.any(b < 0):
        raise RuntimeError('Nonphysical paired dose')
    if not np.array_equal(a, b):
        raise RuntimeError('Ablation changed dose; maximum absolute difference=' + str(np.max(np.abs(a-b))))
    ledgers = [json.loads((d/'energy_ledger.json').read_text()) for d in (before, after)]
    if ledgers[0]['histories'] != ledgers[1]['histories']:
        raise RuntimeError('History mismatch')
    x, y = [l['schneider_diagnostics'] for l in ledgers]
    counts = [k for k in x if isinstance(x[k], int) and not k.startswith('E_') and 'MeV' not in k]
    if any(x[k] != y[k] for k in counts):
        raise RuntimeError('Nuclear counts changed')
    ledger_fields = check_ledger_preservation(*ledgers)
    for d in (before, after):
        q = json.loads((d/'quality_report.json').read_text())
        if q['queue_overflow_count'] or q['queue_overflow_energy_MeV']:
            raise RuntimeError('Overflow: split both runs')
    return dict(bitwise_dose_equal=True, nuclear_counts_equal=True,
                preserved_ledger_fields=ledger_fields,
                checked_integer_fields=counts, histories=ledgers[0]['histories'],
                dose_sha256=sha(before/'dose.raw'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--water', action='store_true')
    p.add_argument('--histories', type=int, default=10000)
    p.add_argument('--joint', type=Path)
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    pins = {str(f.resolve()): sha(f) for f in (args.before, args.after, args.config)}
    for role, binary in [('before', args.before), ('after', args.after)]:
        out = root / role
        if args.water:
            # Probe quality directories use out.name: keep both roles unique.
            out = root / (root.name + '_' + role)
            subprocess.run(['python3', str(repo/'tools/probe_unified_water.py'),
                            '--config', str(args.config.resolve()), '--out', str(out),
                            '--binary', str(binary.resolve()), '--histories', str(args.histories)],
                           cwd=repo, check=True)
        else:
            run(args.config.resolve(), out, binary.resolve(), args.joint)
    before = root / (root.name+'_before' if args.water else 'before')
    after = root / (root.name+'_after' if args.water else 'after')
    result = compare(before, after)
    for path, pin in pins.items():
        if sha(Path(path)) != pin:
            raise RuntimeError('Input/binary changed during pair: '+path)
    result.update(status='PACKAGE_CODE_ABLATION_PASS_NOT_NEW_GAMMA', pins=pins)
    with (root/'comparison.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
