"""Statistical native rate-grid refinement. Never certifies a transport package.

Run in Slurm with the Geant4 environment loaded. Prior dense rates only guide
node placement; all accepted node values are freshly queried. Float32 nodes
preserve representable one-sided model seams for the eventual GPU consumer.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
PROCESSES = (111, 121, 131, 141)

def read_rates(path):
    if not (path.parent / 'COMPLETE').exists():
        raise ValueError(f'incomplete native extraction: {path}')
    table = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            e, p = (float(row['energy_MeV']), int(row['subtype']))
            value, sem = (float(row['rate_per_mm']), float(row['rate_sem_per_mm']))
            count = int(row['rate_queries'])
            if not (np.isfinite([e, value, sem]).all() and e > 0 and (value >= 0) and (sem >= 0) and (count > 1)):
                raise ValueError(f'invalid rate estimate: {row}')
            if (e, p) in table:
                raise ValueError('duplicate native rate')
            table[e, p] = (value, sem, count)
    energies = sorted({e for e, _ in table})
    if len(table) != len(energies) * len(PROCESSES) or any(((e, p) not in table for e in energies for p in PROCESSES)):
        raise ValueError('incomplete native process grid')
    return table

def initial_grid(guide):
    nodes = set(np.geomspace(1e-11, 300.0, 1025))
    nodes.update([2.53e-08, 0.1, 1.0, 10.0, 170.0, 250.0])
    for seam in (19.5, 19.9, 20.0):
        point = np.float32(seam)
        nodes.update([float(np.nextafter(point, np.float32(0))), float(point), float(np.nextafter(point, np.float32(np.inf)))])
    if guide:
        data = np.genfromtxt(guide, delimiter=',', names=True, dtype=None, encoding='utf8')
        for p in PROCESSES:
            rows = data[data['subtype'] == p]
            e, y = (rows['energy_MeV'], rows['rate_per_mm'])
            extrema = ((y[1:-1] - y[:-2]) * (y[2:] - y[1:-1]) < 0) & (e[1:-1] > 1e-05)
            extrema &= abs(y[1:-1] - 0.5 * (y[:-2] + y[2:])) > 0.01 * np.maximum(y[1:-1], 1e-30)
            for i in np.flatnonzero(extrema) + 1:
                nodes.update(e[i - 1:i + 2])
    return sorted({float(np.float32(e)) for e in nodes})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extractor', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('material', choices=['Water_75eV', 'G4_Cu', 'G4_AIR'])
    parser.add_argument('--guide', type=Path)
    parser.add_argument('--resume', type=Path, help='completed refinement to extend without requerying resolved intervals')
    parser.add_argument('--refresh-statistics', action='store_true', help='resample noisy endpoints and re-audit all adjacent intervals; use a larger --queries')
    parser.add_argument('--refine-inconclusive', action='store_true', help='also subdivide statistically resolved curvature near the tolerance boundary')
    parser.add_argument('--queries', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--relative-tolerance', type=float, default=0.02)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.refresh_statistics and (not args.resume):
        parser.error('--refresh-statistics requires --resume')
    if args.queries < 2 or args.rounds < 1 or (not 0 < args.relative_tolerance < 1):
        parser.error('invalid refinement controls')
    previous = None
    if args.resume:
        previous = json.loads((args.resume / 'manifest.json').read_text())
        if previous['material'] != args.material:
            parser.error('resume material mismatch')
        if previous['relative_tolerance'] != args.relative_tolerance:
            parser.error('resume tolerance mismatch')
    if args.seed is None:
        args.seed = previous['seed'] + 1000003 if previous else 92727101
    args.output.mkdir(parents=True, exist_ok=False)
    native = args.output / 'native'
    native.mkdir()
    manifest = dict(schema='maigo.neutron.rate-refinement.v1', runtime_validated=False, material=args.material, extractor=str(args.extractor.resolve()), queries=args.queries, relative_tolerance=args.relative_tolerance, seed=args.seed, notes=['Rate means are native-query ensemble estimates, not validated transport rates.', 'Midpoint agreement cannot exclude features narrower than sampled intervals.', 'No conditional final-state interpolation is provided.'], rounds=[])
    query_index = 0

    def query(energies):
        nonlocal query_index
        grid = native / f'grid_{query_index:03d}.txt'
        np.savetxt(grid, energies, fmt='%.17g')
        out = native / f'query_{query_index:03d}'
        seed = args.seed + 1009 * query_index
        with (native / f'query_{query_index:03d}.log').open('w') as log:
            subprocess.run([str(args.extractor.resolve()), str(out.resolve()), args.material, str(grid.resolve()), '0', str(args.queries), str(seed)], stdout=log, stderr=subprocess.STDOUT, check=True)
        query_index += 1
        return read_rates(out / 'rates.csv')
    audit = []
    if args.resume:
        table = read_rates(args.resume / 'rates.csv')
        with (args.resume / 'interval_audit.csv').open() as stream:
            old = list(csv.DictReader(stream))
        active = sorted({(float(r['low_MeV']), float(r['high_MeV'])) for r in old if r['status'] == 'refine' or (args.refine_inconclusive and r['status'] == 'statistics_limited')})
        if args.refresh_statistics:
            refresh_nodes = sorted({float(r[k]) for r in old if r['status'] == 'statistics_limited' for k in ['low_MeV', 'high_MeV']})
            if refresh_nodes:
                if any((args.queries <= table[e, p][2] for e in refresh_nodes for p in PROCESSES)):
                    raise ValueError('statistical refresh must increase endpoint query count')
                table.update(query(refresh_nodes))
                touched = set(refresh_nodes)
                active = sorted(set(active) | {(float(r['low_MeV']), float(r['high_MeV'])) for r in old if float(r['low_MeV']) in touched or float(r['high_MeV']) in touched})
                manifest['refreshed_endpoint_count'] = len(refresh_nodes)
        active_set = set(active)
        for r in old:
            lo, hi = (float(r['low_MeV']), float(r['high_MeV']))
            if (lo, hi) not in active_set:
                audit.append((lo, hi, int(r['subtype']), float(r['midpoint_residual_per_mm']), float(r['combined_sem_per_mm']), float(r['midpoint_rate_per_mm']), r['status']))
    else:
        nodes = initial_grid(args.guide)
        table = query(nodes)
        active = list(zip(nodes[:-1], nodes[1:]))
    for iteration in range(args.rounds):
        intervals = []
        for lo, hi in active:
            mid = float(np.float32(np.sqrt(lo * hi)))
            if lo < mid < hi:
                intervals.append((lo, mid, hi))
            else:
                for p in PROCESSES:
                    audit.append((lo, hi, p, 0.0, 0.0, 0.0, 'adjacent_float32'))
        if not intervals:
            break
        mids = query(sorted({mid for _, mid, _ in intervals}))
        active = []
        split = 0
        for lo, mid, hi in intervals:
            fraction = np.log(mid / lo) / np.log(hi / lo)
            values = []
            needs_refinement = False
            for p in PROCESSES:
                y0, s0, _ = table[lo, p]
                y1, s1, _ = table[hi, p]
                ym, sm, _ = mids[mid, p]
                estimate = y0 + fraction * (y1 - y0)
                residual = abs(estimate - ym)
                uncertainty = np.sqrt(sm * sm + (1 - fraction) ** 2 * s0 * s0 + fraction * fraction * s1 * s1)
                threshold = args.relative_tolerance * ym
                significant = residual > threshold + 3 * uncertainty
                upper = residual + 3 * uncertainty
                status = 'refine' if significant else 'pointwise_bound_met' if upper <= threshold else 'statistics_limited'
                if ym == estimate == uncertainty == 0:
                    status = 'zero_process'
                needs_refinement |= significant or (args.refine_inconclusive and status == 'statistics_limited' and (residual > 3 * uncertainty))
                values.append((lo, hi, p, residual, uncertainty, ym, status))
            if needs_refinement and iteration + 1 < args.rounds:
                table.update({(mid, p): mids[mid, p] for p in PROCESSES})
                active.extend([(lo, mid), (mid, hi)])
                split += 1
            else:
                audit.extend(values)
        manifest['rounds'].append(dict(iteration=iteration, queried_intervals=len(intervals), refined_intervals=split, total_nodes=len(table) // 4))
        print(json.dumps(manifest['rounds'][-1]), flush=True)
        (args.output / 'progress.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if not active:
            break
    with (args.output / 'rates.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['energy_MeV', 'subtype', 'rate_per_mm', 'rate_sem_per_mm', 'rate_queries'])
        for (e, p), (y, s, n) in sorted(table.items()):
            writer.writerow([e, p, y, s, n])
    with (args.output / 'interval_audit.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['low_MeV', 'high_MeV', 'subtype', 'midpoint_residual_per_mm', 'combined_sem_per_mm', 'midpoint_rate_per_mm', 'status'])
        writer.writerows(audit)
    from collections import Counter
    manifest['interval_status_counts'] = dict(Counter((row[-1] for row in audit)))
    manifest['rate_nodes'] = len(table) // 4
    manifest['status'] = 'native_rate_refinement_only'
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'COMPLETE').write_text('native rate refinement only; runtime_validated=false\n')
    print(json.dumps(manifest['interval_status_counts']), flush=True)
if __name__ == '__main__':
    main()
