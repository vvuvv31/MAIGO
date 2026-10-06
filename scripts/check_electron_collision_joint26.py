#!/usr/bin/env python3
"""Diagnose empirical26 collision joint laws using completed native events.

This reports joint-CDF differences; it does not certify native equivalence or
replace unresolved rates/outcomes. Local deposits use the walker's canonical
energy fraction1/cosine0. All native branches are weighted by their energy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_electron_response26 import raw_rate
from extract_photon_response import DTYPE
from validate_electron_response26 import masses, weights

ENERGY_THRESHOLDS = np.unique(np.r_[np.geomspace(1e-5, .1, 33), np.linspace(.1, 1, 33)])
ANGLE_THRESHOLDS = np.linspace(-1, 1, 33)

def cdf(energy, cosine, weight):
    edges, angles = ENERGY_THRESHOLDS, ANGLE_THRESHOLDS
    # Upper-tail counters implement F(e,mu) at the same thresholds, including
    # exactly-zero energy and exactly +/-1 directions without bin ambiguity.
    i = np.searchsorted(edges, energy, side='left')
    j = np.searchsorted(angles, cosine, side='left')
    hist = np.zeros((len(edges)+1, len(angles)+1))
    np.add.at(hist, (i, j), weight)
    return hist.cumsum(0).cumsum(1)[:-1, :-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    manifest = json.loads((a.native/'result.json').read_text())
    rows = []
    for group in manifest['results']:
        directory = Path(group['directory'])
        kind = group['rows'][0]['kind']
        section = group['rows'][0]['section']
        raw = np.fromfile(directory/'conditional.phsp', DTYPE)
        with np.load(a.candidate/kind/f'composition{section:02d}.npz') as model:
            requests = np.loadtxt(directory/'requests.txt', ndmin=2)
            for b, proc, e, count in requests:
                b, proc, count = int(b), int(proc), int(count)
                record = next(r for r in group['rows']
                              if r['process']==proc and r['energy']==e)
                event = raw[(raw['kind']==0)&(raw['run']==b)&(raw['process']==proc)]
                child = raw[(raw['kind']==1)&(raw['run']==b)&(raw['process']==proc)]
                if len(event)!=count:
                    raise ValueError('Native event count')
                # Group contains all events at a fixed incoming energy.
                native_energy = np.r_[event['remaining']/e, child['remaining']/e,
                                      np.ones(len(event))]
                native_cosine = np.r_[event['post'][:,2], child['post'][:,2],
                                      np.zeros(len(event))]
                native_weight = np.r_[event['remaining'], child['remaining'],
                                      event['deposit']] / (count*e)
                active = native_weight>0
                if abs(native_weight.sum()-1)>1e-8:
                    raise ValueError('Native branch energy closure')
                native = cdf(native_energy[active], native_cosine[active], native_weight[active])
                domain = model['density_domain'].astype(float)
                x = 0 if domain[0]==domain[1] else (record['density']-domain[0])/(domain[1]-domain[0])
                mix = weights(int(model['density_degree'][0]), x, domain)
                if kind=='charged':
                    edges = model[f'rate{proc}_edges'].astype(float)
                    values = model[f'rate{proc}_values'].astype(float)
                    pieces = np.concatenate([
                        np.broadcast_to(edges[:-1],values.shape[:2])[...,None],
                        np.broadcast_to(edges[1:],values.shape[:2])[...,None],values],axis=-1)
                    mix *= np.array([max(0.,float(raw_rate(row,e))) for row in pieces])
                    mix /= mix.sum()
                prefix = f'ch{b*(2 if kind=="charged" else 3)+proc:04d}_'
                law = (mix[:,None]*masses(model[prefix+'cdf'])).reshape(-1)
                candidate = cdf(model[prefix+'energy'].reshape(-1),
                                model[prefix+'cosine'].reshape(-1),law)
                hotspot = np.unravel_index(np.argmax(abs(candidate-native)), native.shape)
                rows.append(dict(**record, native_events=count,
                                 joint_cdf_absolute=float(abs(candidate-native).max()),
                                 hotspot_energy_fraction=float(ENERGY_THRESHOLDS[hotspot[0]]),
                                 hotspot_cosine=float(ANGLE_THRESHOLDS[hotspot[1]]),
                                 hotspot_native_cdf=float(native[hotspot]),
                                 hotspot_candidate_cdf=float(candidate[hotspot])))
    report = dict(status='EMPIRICAL26_NATIVE_JOINT_DIAGNOSTIC_NOT_ACCEPTANCE',
                  cases=len(rows), maximum_joint_cdf=max(r['joint_cdf_absolute'] for r in rows),
                  rows=rows)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
