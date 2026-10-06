#!/usr/bin/env python3
"""Integrate the current Mott-corrected single-elastic law for class-II research.

This does not change GPU transport or produce an accepted condensed model.
The current sampler selects the Mott energy row BEFORE its rejection loop:
mix individually normalized angular laws, not unnormalized acceptance curves.
Quadrature uses Rutherford CDF coordinates and subdivides at angular knots.
"""
import argparse
import json
import math
import struct
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

def read_nodes(index):
    lines = index.read_text().splitlines()
    magic, count = lines[0].split()
    if magic != 'MAIECB02' or len(lines) != int(count) + 1:
        raise ValueError('charged index header/count')
    nodes = []
    for line in lines[1:]:
        section, density, filename = line.split()
        path = index.parent / filename
        with path.open('rb') as stream:
            header = stream.read(9600)
        fields = struct.unpack_from('<8sIidIIIIQdddd', header)
        if (fields[0], fields[1]) not in [(b'MAIECH02', 2), (b'MAIECF01', 1)]:
            raise ValueError('charged format')
        if fields[2] != int(section) or fields[3] != float(density):
            raise ValueError('node identity')
        elastic_magic, version, rows = struct.unpack_from('<8sII', header, 80)
        if elastic_magic != b'MAIELS01' or version != 1 or rows != 31:
            raise ValueError('elastic format')
        mass, bc, xc2 = np.asarray(struct.unpack_from('<ddd', header, 96), np.float32).astype(float)
        table = np.frombuffer(header, '<f8', 31 * 38, 120).reshape(31, 38).astype(np.float32).astype(float)
        nodes.append(dict(section=int(section), density=float(density), mass=mass, bc=bc, xc2=xc2, table=table))
    return nodes

def angular_parameters(node, energy):
    m = node['mass']
    p2 = energy * (energy + 2 * m)
    beta = p2 / (p2 + m * m)
    mid = 0.1 * (0.1 + 2 * m)
    midbeta = mid / (mid + m * m)
    if beta >= 0.9999:
        low, weight = (30, 0.0)
    else:
        x = 15 + 15 * (beta - midbeta) / (0.9999 - midbeta) if beta >= midbeta else 15 * (math.log(energy) - math.log(0.001)) / math.log(100) if energy > 0.001 else 0.0
        low, weight = (int(x), x - int(x))
    factor = node['table'][low, 0]
    if weight:
        factor += weight * (node['table'][low + 1, 0] - factor)
    screening = node['xc2'] / (4 * energy * (energy + 2 * m) * node['bc']) * factor
    if not (screening > 0 and 0 <= low <= 30 and (0 <= weight < 1)):
        raise ValueError('angular parameters')
    return (screening, low, weight)

def acceptance(row, q):
    y = np.sqrt(q / 2)
    x = y * 31
    i = np.minimum(x.astype(int), 29)
    values = row[3:]
    linear = values[i] + (x - i) * (values[i + 1] - values[i])
    t = (y - 30 / 31) * 31
    c = values[31:35]
    cubic = -4.5 * (t - 1 / 3) * (t - 2 / 3) * (t - 1) * c[0] + 13.5 * t * (t - 2 / 3) * (t - 1) * c[1] - 13.5 * t * (t - 1 / 3) * (t - 1) * c[2] + 4.5 * t * (t - 1 / 3) * (t - 2 / 3) * c[3]
    return np.clip(np.where(x < 30, linear, cubic), 0.0, 1.0)
QUADRATURE = {n: leggauss(n) for n in (8, 16, 32)}

def edges(a, low=0.0, high=2.0):
    q = np.r_[0.0, np.geomspace(min(a * 2e-06, 1e-06), 2.0, 97), 2 * (np.arange(32) / 31) ** 2]
    return np.unique(np.r_[low, q[(q > low) & (q < high)], high])

def integrate(a, rows, boundaries, order):
    u = boundaries * (1 + a) / (2 * a + boundaries)
    if boundaries[-1] == 2:
        u[-1] = 1.0
    x, w = QUADRATURE[order]
    half = np.diff(u) / 2
    sample = u[:-1, None] + half[:, None] + half[:, None] * x
    q = 2 * a * sample / (1 + a - sample)
    w = half[:, None] * w
    moments = np.stack([np.ones_like(q), q, 3 * q - 1.5 * q * q])
    return np.stack([np.einsum('lij,ij->il', moments, w * (acceptance(row, q) if row is not None else 1)) for row in rows])

def rutherford_check():
    worst = 0.0
    for a in (1e-07, 0.0001, 0.01, 1.0, 100.0):
        actual = integrate(a, [None], edges(a), 32).sum(axis=1)[0]
        with localcontext() as context:
            context.prec = 60
            b = 2 * Decimal(str(a))
            k = b * (b + 2) / 2
            log = ((b + 2) / b).ln()
            first = k * (log + b / (b + 2) - 1)
            second = k * (2 - 2 * b * log + b - b * b / (b + 2))
            expected = np.array([1.0, float(first), float(3 * first - Decimal('1.5') * second)])
        worst = max(worst, float(np.max(abs(actual - expected) / expected)))
    if worst > 1e-08:
        raise AssertionError(f'analytic Rutherford moment error {worst}')
    return worst

def analyze(node, energy, order, soft_fraction):
    a, low, weight = angular_parameters(node, energy)
    rows = [node['table'][low], node['table'][min(low + 1, 30)]]
    weights = np.array([1 - weight, weight])
    boundaries = edges(a)
    pieces = integrate(a, rows, boundaries, order)
    normalizers = pieces[:, :, 0].sum(axis=1)
    if not np.all((normalizers > 0) & (normalizers <= 1 + 1e-10)):
        raise AssertionError('Mott normalization')
    mixture = (pieces * (weights / normalizers)[:, None, None]).sum(axis=0)
    cumulative = np.r_[0.0, np.cumsum(mixture[:, 0])]
    i = min(len(boundaries) - 2, max(0, np.searchsorted(cumulative, soft_fraction) - 1))
    lo, hi = boundaries[i:i + 2]
    start = lo
    for _ in range(40):
        mid = (lo + hi) / 2
        extra = integrate(a, rows, np.array([start, mid]), order)[:, 0, 0]
        cdf = cumulative[i] + np.dot(weights / normalizers, extra)
        if cdf < soft_fraction:
            lo = mid
        else:
            hi = mid
    cutoff = (lo + hi) / 2

    def partial(left, right):
        raw = integrate(a, rows, edges(a, left, right), order).sum(axis=1)
        return (raw * (weights / normalizers)[:, None]).sum(axis=0)
    soft, hard = (partial(0, cutoff), partial(cutoff, 2))
    total = mixture.sum(axis=0)
    closure = float(np.max(abs(soft + hard - total) / np.maximum(total, 1e-30)))
    if closure > 1e-07 or abs(soft[0] - soft_fraction) > 1e-08:
        raise AssertionError(f'angular split closure {closure}')
    return dict(energy_MeV=energy, screening=a, energy_low=low, energy_high_weight=weight, acceptance_normalizers=normalizers.tolist(), cutoff_q=cutoff, cutoff_angle_rad=2 * math.asin(math.sqrt(cutoff / 2)), total_moments=total.tolist(), soft_moments=soft.tolist(), hard_moments=hard.tolist(), split_relative_closure=closure)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--energies', type=int, default=33)
    p.add_argument('--soft-fraction', type=float, default=0.9)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.energies < 2 or not 0 < args.soft_fraction < 1:
        p.error('invalid grid or fraction')
    analytic_error = rutherford_check()
    nodes = read_nodes(args.index)
    energies = np.unique(np.r_[np.geomspace(0.0001001, 0.65, args.energies), 0.001, 0.01, 0.1, 0.4])
    results, worst = ([], 0.0)
    for i, node in enumerate(nodes):
        for energy in energies:
            coarse = analyze(node, float(energy), 16, args.soft_fraction)
            fine = analyze(node, float(energy), 32, args.soft_fraction)
            a = np.r_[coarse['total_moments'], coarse['soft_moments'], coarse['hard_moments'], coarse['cutoff_q']]
            b = np.r_[fine['total_moments'], fine['soft_moments'], fine['hard_moments'], fine['cutoff_q']]
            error = float(np.max(abs(a - b) / np.maximum(abs(b), 1e-30)))
            worst = max(worst, error)
            results.append(dict(node=i, section=node['section'], density=node['density'], quadrature_relative_change=error, **fine))
        print(json.dumps(dict(node=i, cases=len(results), maximum_refinement_error=worst)), flush=True)
    report = dict(status='OFFLINE_ANGULAR_MOMENTS_NOT_TRANSPORT_ACCEPTANCE', passed=worst < 1e-07, soft_fraction=args.soft_fraction, analytic_rutherford_relative_error=analytic_error, maximum_refinement_relative_change=worst, results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('quadrature refinement failed')
if __name__ == '__main__':
    main()
