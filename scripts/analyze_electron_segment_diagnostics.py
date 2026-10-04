#!/usr/bin/env python3
"""Describe class-II opportunities at sampled ORIGINAL trajectory states.

This does not simulate candidate paths, measure saved calls/time, or validate
energy-loss coupling. Offline integration uses double; GPU sampling is FP32.
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammainc
from electron_elastic_moments import read_nodes, angular_parameters, edges, integrate

DT = np.dtype([
    ('history', '<u8'), ('birth_rng', '<u8'), ('birth_step', '<u4'),
    ('electron_step', '<u4'), ('valid', '<u4'), ('node', '<u4'),
    *[(name, '<f4') for name in ('energy', 'density', 'elastic_rate', 'dedx',
      'range', 'hard0', 'hard1', 'material_safety', 'scoring_safety', 'rounding')],
    ('position', '<f4', 3), ('reserved', '<u4')])
assert DT.itemsize == 88
NODES = None


def init(index):
    global NODES
    NODES = read_nodes(Path(index))


def soft_probability(key):
    node_index, energy = key
    node = NODES[node_index]
    a, low, weight = angular_parameters(node, energy)
    rows = [node['table'][low], node['table'][min(low + 1, 30)]]
    weights = np.array([1 - weight, weight])
    cutoff = 1 - math.cos(.1)
    values = []
    for order in (16, 32):
        total = integrate(a, rows, edges(a), order)[:, :, 0].sum(axis=1)
        soft = integrate(a, rows, edges(a, high=cutoff), order)[:, :, 0].sum(axis=1)
        values.append(min(.9, float(np.dot(weights / total, soft))))
    if not (0 < values[1] <= .9) or abs(values[0] - values[1]) > 1e-7:
        raise ValueError(f'Unresolved soft probability: {key} {values}')
    return values[1], abs(values[0] - values[1])


def quantiles(x):
    return {str(q): float(np.percentile(x, q)) for q in (0, 10, 50, 90, 99, 100)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    files = sorted(args.root.glob('out/fresh*/electron_segment_samples.bin'))
    if not files:
        raise ValueError('No sample files')
    samples = np.concatenate([np.fromfile(f, DT) for f in files])
    status = {str(v): int(np.count_nonzero(samples['valid'] == v)) for v in np.unique(samples['valid'])}
    if status.get('0', 0):
        raise ValueError(f'Invalid observational samples: {status}')
    d = samples[samples['valid'] == 1]
    keys = [(int(n), float(e)) for n, e in zip(d['node'], d['energy'])]
    unique = list(dict.fromkeys(keys))
    with ProcessPoolExecutor(args.workers, initializer=init,
                             initargs=(str(args.index),)) as pool:
        results = list(pool.map(soft_probability, unique, chunksize=128))
    lookup = dict(zip(unique, results))
    soft = np.array([lookup[k][0] for k in keys])
    np.savez(args.root/'segment_probabilities.npz', probability=soft,
             node=d['node'], energy=d['energy'])
    energy = d['energy'].astype(float)
    elastic = d['elastic_rate'].astype(float)
    loss_cap = .005 * energy / d['dedx']
    base = np.minimum(.05, np.minimum(loss_cap, d['range']))
    material = np.maximum(0, d['material_safety'].astype(float) - d['rounding'])
    scoring = np.maximum(0, d['scoring_safety'].astype(float) - d['rounding'])
    lengths = {'loss_range_step': base, 'plus_material_faces': np.minimum(base, material),
               'plus_material_and_scoring_faces': np.minimum(base, np.minimum(material, scoring))}
    soft_rate = soft * elastic
    hard_rate = d['hard0'].astype(float) + d['hard1'].astype(float) + (1 - soft) * elastic
    assert np.all(hard_rate > 0)
    cuts = [0, .001, .01, .1, 1, np.inf]
    rows = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mask = (energy >= lo) & (energy < hi)
        if not mask.any():
            continue
        row = dict(energy_MeV=[lo, hi if np.isfinite(hi) else None],
                   samples=int(mask.sum()), valid_state_fraction=float(mask.mean()),
                   soft_probability_quantiles=quantiles(soft[mask]), variants={})
        for label, length in lengths.items():
            cap_mu = soft_rate * length
            expected_count = soft_rate * (-np.expm1(-hard_rate * length)) / hard_rate
            # Fifth soft arrival occurs before both the cap and first hard
            # proposal: Erlang survival with competing exponential hazard.
            above_four = (soft_rate / (soft_rate + hard_rate))**5 * gammainc(5, (soft_rate + hard_rate)*length)
            row['variants'][label] = dict(
                length_mm_quantiles=quantiles(length[mask]),
                cap_mean_soft_count_quantiles=quantiles(cap_mu[mask]),
                state_fraction_cap_mean_above_four=float(np.mean(cap_mu[mask] > 4)),
                mean_count_before_hard_or_cap=float(expected_count[mask].mean()),
                mean_probability_count_above_four=float(above_four[mask].mean()))
        row['material_limited_fraction'] = float(np.mean(material[mask] < base[mask]))
        row['either_face_limited_fraction'] = float(np.mean(np.minimum(material, scoring)[mask] < base[mask]))
        rows.append(row)
    final_length = lengths['plus_material_and_scoring_faces']
    expected_count = soft_rate * (-np.expm1(-hard_rate*final_length)) / hard_rate
    above_four = (soft_rate/(soft_rate+hard_rate))**5*gammainc(5,(soft_rate+hard_rate)*final_length)
    report = dict(
        scope='State-weighted feasibility on unchanged FP32 RT07575 trajectories. Not candidate trajectory weighting, measured saved calls, containment proof, dose acceptance or throughput.',
        sampling='Independent hash of immutable birth RNG and electron step, probability 1/16384; no transport RNG consumed.',
        sample_files=[str(f) for f in files], total_samples=len(samples), statuses=status,
        valid_states=len(d), unique_node_energies=len(unique),
        integration_16_32_max_difference=max(v[1] for v in results),
        assumptions=dict(maximum_step_mm=.05, mean_loss_fraction=.005,
                         soft_fraction='min(0.9, current-law CDF at 0.1 rad)',
                         hard_termination='unchanged ionization/brems majorants + hard elastic rate, frozen at sampled initial energy',
                         geometry='radius below all six material/scoring faces, subtract observational FP32 margin; no rejection of lateral displacement'),
        pooled=dict(face_limited_fraction=float(np.mean(final_length < base)),
                    mean_soft_count_before_hard_or_cap=float(expected_count.mean()),
                    mean_probability_count_above_four=float(above_four.mean()),
                    zero_length_state_fraction=float(np.mean(final_length <= 0))),
        energy_bins=rows)
    (args.root/'segment_feasibility.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
