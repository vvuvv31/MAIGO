#!/usr/bin/env python3
"""Exact per-condition check of raw Urban losses exceeding incident KE."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    checks = []
    threshold = 1e-06 / (82 * 40)
    for path in sorted(a.source.glob('node*/urban_checks.csv')):
        rows = np.genfromtxt(path, delimiter=',', names=True)
        for row in rows:
            n = int(row['samples'])
            x = int(row['over_energy_native'])
            y = int(row['over_energy_portable'])
            pv = float(fisher_exact([[x, n - x], [y, n - y]]).pvalue)
            checks.append(dict(node=path.parent.name, cell=int(row['cell']), native=x, portable=y, samples=n, pvalue=pv, passed=pv >= threshold))
    result = dict(passed=bool(checks) and all((x['passed'] for x in checks)), checks=checks, threshold=threshold, scope='Raw-loss law only; exact per-condition tests. Conditions reuse random streams and may be correlated across materials, so counts must not be pooled as independent trials. Actual AlongStep terminal-energy handling remains unimplemented.')
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['passed'], len(checks), min((x['pvalue'] for x in checks)))
    assert result['passed']
if __name__=='__main__':main()
