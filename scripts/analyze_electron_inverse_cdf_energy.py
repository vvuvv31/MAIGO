#!/usr/bin/env python3
"""Compare independent FP32 single-elastic sampler outputs with angular theory."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, binomtest
from numpy.polynomial.legendre import leggauss
import electron_elastic_moments as m
from check_electron_inverse_cdf_energy import interpolate


def transport_second_variance(node, spec, order):
    a = spec['screening']
    edges = m.edges(a)
    u = edges*(1+a)/(2*a+edges);u[-1] = 1.
    x, w = leggauss(order);half = np.diff(u)*.5
    sample = u[:-1, None]+half[:, None]*(1+x)
    q = 2*a*sample/(1+a-sample)
    low = spec['energy_low'];high_weight = spec['energy_high_weight']
    factors = np.array([1-high_weight, high_weight])/spec['acceptance_normalizers']
    density = sum(factor*m.acceptance(node['table'][row], q) for factor, row in
                  zip(factors, [low, min(low+1, 30)]))
    moment = np.sum(half[:, None]*w*density*(3*q-1.5*q*q)**2)
    return float(moment-spec['total_moments'][2]**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=131072)
    args = parser.parse_args()
    table = json.loads((args.tables/'report.json').read_text())
    offgrid = json.loads((args.root/'report.json').read_text())
    if table['index_sha256'] != offgrid['index_sha256']:raise ValueError('source identity')
    nodes = m.read_nodes(args.index, table['index_sha256'])
    cases = offgrid['results'];n = args.samples
    samples = {arm: np.memmap(args.root/f'samples.{arm}.bin', '<f4', mode='r').reshape(len(cases), n)
               for arm in ('reference', 'candidate')}
    quantiles = np.fromfile(args.tables/'quantiles.bin', '<f4').reshape(-1, table['knots'])
    probs = np.fromfile(args.tables/'probabilities.bin', '<f8')
    results = []
    for i, case in enumerate(cases):
        a, low, w = m.angular_parameters(nodes[case['node']], case['energy'])
        native_rows=[nodes[case['node']]['table'][low],nodes[case['node']]['table'][min(low+1,30)]]
        raw=m.integrate(a,native_rows,m.edges(a),32).sum(axis=1)
        total=(raw*(np.array([1-w,w])/raw[:,0])[:,None]).sum(axis=0)
        spec=dict(screening=a,energy_low=low,energy_high_weight=w,
                  total_moments=total,acceptance_normalizers=raw[:,0])
        expected = spec['total_moments'][1:]
        variance1 = (3*expected[0]-expected[1])/1.5-expected[0]**2
        variance2 = transport_second_variance(nodes[case['node']], spec, 32)
        coarse = transport_second_variance(nodes[case['node']], spec, 16)
        if variance1<=0 or variance2<=0 or abs(coarse/variance2-1)>1e-7:
            raise ValueError('theoretical variance quadrature')
        row = dict(node=case['node'], energy=case['energy'], arms={})
        for arm, data in samples.items():
            q = data[i].astype(float)
            if not np.isfinite(q).all() or q.min() < 0 or q.max() > 2:
                raise ValueError('invalid sample')
            metrics = []
            for actual, target, variance in zip((q, 3*q-1.5*q*q), expected, (variance1,variance2)):
                se = float(np.sqrt(variance/n));delta = float(actual.mean()-target)
                metrics.append(dict(mean=float(actual.mean()), theory=target, standard_error=se,
                    empirical_standard_error=float(actual.std(ddof=1)/np.sqrt(n)),
                    relative_error=delta/target, z=abs(delta)/se,
                    screen_pass=bool(abs(delta) <= 7*se+1e-4*target)))
            tails = []
            for survival in (.1, .01, .001, .0001):
                q0=np.array([np.interp(1-survival,probs,quantiles[case['low']])])
                q1=np.array([np.interp(1-survival,probs,quantiles[case['high']])])
                cutoff = float(interpolate(q0,q1,case['screening_low'],case['screening_high'],a,case['weight'],np.array([1-survival]))[0])
                count = int(np.sum(q > cutoff))
                tails.append(dict(expected_probability=survival, cutoff_q=cutoff, count=count,
                    exact_binomial_p=float(binomtest(count, n, survival).pvalue)))
            row['arms'][arm] = dict(moments=metrics, tails=tails)
        ks = ks_2samp(samples['reference'][i], samples['candidate'][i])
        row['ks'] = float(ks.statistic);row['ks_p'] = float(ks.pvalue)
        results.append(row)
    # Bonferroni screens are component diagnostics, not CT acceptance. Tail
    # thresholds carry the independently quantified table representation error.
    failures = [(r['node'], r['energy'], arm, j) for r in results
                for arm in samples for j, v in enumerate(r['arms'][arm]['moments']) if not v['screen_pass']]
    tail_tests = len(cases)*2*4
    tail_failures = [(r['node'], r['energy'], arm, j) for r in results for arm in samples
                     for j,v in enumerate(r['arms'][arm]['tails']) if v['exact_binomial_p'] < .01/tail_tests]
    logs = [json.loads(line) for line in (args.root/'gpu.log').read_text().splitlines() if line.startswith('{')]
    times = {str(arm): [r['kernel_seconds'] for r in logs if r['arm'] == arm and not r['warmup']]
             for arm in (0,1)}
    report = dict(scope='off-grid single-elastic component only; no transport/CT/performance promotion',
        moment_uncertainty='independent theoretical variances, 16/32-point quadrature checked; empirical SE diagnostic only',
        cases=len(cases), samples_per_case_per_arm=n, total_samples_per_arm=n*len(cases),
        maximum_ks=max(r['ks'] for r in results), moment_failures=failures,
        ks_family_failures=[(r['node'], r['energy']) for r in results if r['ks_p'] < .01/len(cases)],
        tail_family_failures=tail_failures, standalone_kernel_seconds=times, results=results)
    (args.root/'gpu_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    if failures or tail_failures or report['ks_family_failures']:
        raise SystemExit('component screen failed')


if __name__ == '__main__':
    main()
