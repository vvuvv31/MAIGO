#!/usr/bin/env python3
"""Compare independent raw TOPAS elastic banks; distribution diagnostic, not dose acceptance."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp


def read_raw(path):
    with path.open('rb') as f:
        if f.read(8) != b'ELRAW001':
            raise ValueError(f'Wrong schema: {path}')
        z, a, ns, nt, ne, nq = struct.unpack('<6I', f.read(24))
        mass = struct.unpack('<d', f.read(8))[0]
        energies = np.fromfile(f, '<f8', ne)
        rates = np.fromfile(f, '<f8', ns * nt * ne).reshape(ns, nt, ne)
        samples = np.fromfile(f, [('fraction', '<f4'), ('mass', '<f4'), ('a', '<u4')], nt * ne * nq).reshape(nt, ne, nq)
        if f.read(1):
            raise ValueError(f'Trailing bytes: {path}')
    return (z, a, mass), energies, rates, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    channels, files = [], []
    for p in sorted(args.baseline.glob('z*a*.bin')):
        q = args.candidate / p.name
        za, e, r, s = read_raw(p)
        zb, eb, rb, sb = read_raw(q)
        if za != zb or not np.array_equal(e, eb) or not np.array_equal(r, rb):
            raise ValueError(f'Projectile, grid or rate changed: {p.name}')
        files.append({'baseline': str(p), 'candidate': str(q), 'baseline_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'candidate_sha256': hashlib.sha256(q.read_bytes()).hexdigest()})
        for t, i in np.argwhere(np.any(r > 0, axis=0)):
            x, y = s['fraction'][t, i], sb['fraction'][t, i]
            ks = ks_2samp(x, y, method='asymp')
            se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
            channels.append({'z': za[0], 'a': za[1], 'target_index': int(t), 'energy_MeVu': float(e[i]), 'ks_distance': float(ks.statistic), 'ks_p': float(ks.pvalue), 'baseline_mean_transfer_fraction': float(x.mean()), 'candidate_mean_transfer_fraction': float(y.mean()), 'mean_difference_standard_errors': float((y.mean()-x.mean())/se) if se > 0 else 0., 'baseline_q99': float(np.quantile(x, .99)), 'candidate_q99': float(np.quantile(y, .99))})
    if len(files) != 18:
        raise ValueError(f'Expected all 18 projectiles, got {len(files)}')
    threshold = .01 / len(channels)
    report = {'scope': 'Same energy nodes, independent seeds; raw t/tmax distributions. This does not establish dose or energy-grid convergence.', 'files': files, 'channel_count': len(channels), 'familywise_alpha': .01, 'bonferroni_p_threshold': threshold, 'significant_ks_channels': sum(c['ks_p'] < threshold for c in channels), 'largest_ks_channels': sorted(channels, key=lambda c:c['ks_distance'], reverse=True)[:100], 'channels': channels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','channels','largest_ks_channels')}, indent=2))


if __name__ == '__main__':
    main()
