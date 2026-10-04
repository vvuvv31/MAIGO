#!/usr/bin/env python3
"""Research inverse CDF of the full single-elastic Mott law; no transport integration.

Probability knots resolve both tails geometrically, including exact support
endpoints. Each energy row is normalized before mixing, matching native row
selection before rejection. Host quadrature uses double; stored samples FP32.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import electron_elastic_moments as m


def probability_knots(bins):
    half = np.unique(np.r_[np.linspace(0, 2.**-24, bins+1),
                         np.concatenate([np.linspace(2.**e, 2.**(e+1), bins+1)
                                         for e in range(-24, -1)])])
    return np.r_[half, 1-half[-2::-1]]


def make_table(node, energy, bins, order=16, grid_points=4097):
    a, low, weight = m.angular_parameters(node, energy)
    rows = [node['table'][low], node['table'][min(low+1, 30)]]
    edges = np.unique(np.r_[0., np.geomspace(min(a*1e-10, 1e-10), 2., grid_points),
                           2*(np.arange(32)/31)**2])
    pieces = m.integrate(a, rows, edges, order)
    norm = pieces[:, :, 0].sum(axis=1)
    mix = np.array([1-weight, weight])/norm
    cumul = np.r_[0., np.cumsum(np.einsum('rij,r->ij', pieces, mix)[:, 0])]
    cumul[-1] = 1.
    total = np.einsum('rij,r->j', pieces, mix)
    uedge = edges*(1+a)/(2*a+edges)
    uedge[-1] = 1.
    gx, gw = leggauss(order)

    def integrate_partial(start, end):
        half = (end-start)*.5
        u = start[:, None]+half[:, None]*(gx+1)
        q = 2*a*u/(1+a-u)
        acc = sum(w*m.acceptance(row, q) for w, row in zip(mix, rows))
        return half*np.sum(gw*acc, axis=1)

    def cdf(q):
        idx = np.clip(np.searchsorted(edges, q, side='right')-1, 0, len(edges)-2)
        u = q*(1+a)/(2*a+q)
        return cumul[idx]+integrate_partial(uedge[idx], u)

    p = probability_knots(bins)
    idx = np.clip(np.searchsorted(cumul, p, side='right')-1, 0, len(edges)-2)
    left, right = uedge[idx].copy(), uedge[idx+1].copy()
    for _ in range(32):
        mid = (left+right)*.5
        f = cumul[idx]+integrate_partial(uedge[idx], mid)
        left = np.where(f < p, mid, left)
        right = np.where(f >= p, mid, right)
    u = (left+right)*.5
    q = (2*a*u/(1+a-u)).astype(np.float32)
    q[0], q[-1] = 0, 2
    if not np.all(np.isfinite(q)) or np.any(np.diff(q) < 0):
        raise ValueError('nonmonotone/nonfinite quantiles')
    q0, q1 = q[:-1].astype(float), q[1:].astype(float)
    first = np.dot(np.diff(p), (q0+q1)/2)
    second = np.dot(np.diff(p), (q0*q0+q0*q1+q1*q1)/3)
    represented = np.array([1., first, 3*first-1.5*second])
    fraction = np.array([.125, .375, .625, .875])
    test_p = (p[:-1, None]+np.diff(p)[:, None]*fraction).ravel()
    test_q = (q0[:, None]+(q1-q0)[:, None]*fraction).ravel()
    residual = cdf(test_q)-test_p
    resolved = np.minimum(test_p, 1-test_p) >= 1e-6
    return q, dict(energy=energy, screening=a, low=low, high_weight=weight,
        moment_relative_error=(represented/total-1).tolist(),
        max_cdf_residual=float(abs(residual).max()),
        max_relative_tail_probability_error=float(np.max(abs(residual[resolved])/
            np.minimum(test_p[resolved], 1-test_p[resolved]))),
        support=[float(q[0]), float(q[-1])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--moments', type=Path, required=True)
    p.add_argument('--nodes', default='12,32,57')
    p.add_argument('--bins', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.bins not in (16, 32, 64):
        p.error('bins must be 16/32/64')
    raw = json.loads(args.moments.read_text())
    if raw['index_sha256'] != args.sha256:
        raise ValueError('moment source identity')
    nodes = m.read_nodes(args.index, args.sha256)
    selected = set(range(82)) if args.nodes == 'all' else set(map(int, args.nodes.split(',')))
    cases, tables = [], []
    for row in raw['results']:
        if row['node'] not in selected:
            continue
        q, report = make_table(nodes[row['node']], row['energy_MeV'], args.bins)
        cases.append(dict(node=row['node'], **report));tables.append(q)
    args.out.mkdir(parents=True, exist_ok=True)
    np.asarray(tables, dtype='<f4').tofile(args.out/'quantiles.bin')
    probability_knots(args.bins).astype('<f8').tofile(args.out/'probabilities.bin')
    worst_moment = max(max(abs(v) for v in r['moment_relative_error']) for r in cases)
    report = dict(scope='single-elastic angular representation, exact energy grid only; no CT or throughput acceptance',
        index_sha256=args.sha256, nodes=sorted(selected), bins=args.bins,
        knots=len(tables[0]), bytes=(args.out/'quantiles.bin').stat().st_size,
        maximum_relative_moment_error=worst_moment,
        maximum_cdf_residual=max(r['max_cdf_residual'] for r in cases),
        maximum_relative_tail_probability_error=max(r['max_relative_tail_probability_error'] for r in cases),
        cases=cases)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__ == '__main__':
    main()
