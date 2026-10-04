#!/usr/bin/env python3
"""Full electron reference profiles with audited native transport settings.

Radial histogram integration amortizes the azimuth/source-width integrals.
Its discretization is checked by merging adjacent bins; it never rescales a
model dose to agree with the reference. Offline FP64 statistics are unrelated
to the mandatory FP32 GPU dose scorer.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from macro_model import extract
from prototype import NAMES, TAILS, nodes, periodic_roi, save, sha

CENTRES = np.arange(-1.75, 1.8, .1)


def summarize(data, energy, dr=.000125, order=8):
    count = len(data['ledger'])
    exposure = np.bincount(np.arange(count) % 20, minlength=20)*energy
    maximum = max(np.linalg.norm(data['pre'][:, :2], axis=1).max(initial=0),
                  np.linalg.norm(data['post'][:, :2], axis=1).max(initial=0))
    size = 2*(int(np.ceil(maximum/dr))//2+1)
    hist = np.zeros((20, size)); moments = np.zeros((20, 4))
    block = data['event'] % 20
    for t, q in zip(*nodes(order)):
        xyz = data['pre']+t*(data['post']-data['pre'])
        r = np.hypot(xyz[:, 0], xyz[:, 1]); z = xyz[:, 2]
        weight = data['weight']*q
        ids = block*size+(r/dr).astype(np.int64)
        hist += np.bincount(ids, weights=weight, minlength=20*size).reshape(20, size)
        for i, value in enumerate([r*r, z, z*z, r*z]):
            moments[:, i] += np.bincount(block, weights=weight*value, minlength=20)
    hist /= exposure[:, None]; moments /= exposure[:, None]
    radii = (np.arange(size)+.5)*dr
    rois = np.array([periodic_roi(radii, 0.), periodic_roi(radii, 1.8)])
    tails = np.array([(radii > x).astype(float) for x in TAILS])
    metric_blocks = np.column_stack([moments, hist@rois.T, hist@tails.T])
    transform = np.array([periodic_roi(radii, float(x), halfwidth=.05) for x in CENTRES])
    profile_blocks = hist@transform.T
    coarse_radii = (np.arange(size//2)+.5)*2*dr
    coarse_hist = hist.reshape(20, -1, 2).sum(axis=2)
    coarse_profile = coarse_hist@np.array([periodic_roi(coarse_radii, float(x), halfwidth=.05) for x in CENTRES]).T
    coarse_rois = coarse_hist@np.array([periodic_roi(coarse_radii, 0.), periodic_roi(coarse_radii, 1.8)]).T
    charged_fraction = data['ledger'][:, 0].sum()/(count*energy)
    assert abs(profile_blocks.mean(axis=0).sum()-charged_fraction) < 1e-8
    roi_error = abs((coarse_rois.mean(axis=0)-metric_blocks[:, 4:6].mean(axis=0))/metric_blocks[:, 4:6].mean(axis=0))
    profile_error = abs((coarse_profile.mean(axis=0)-profile_blocks.mean(axis=0))/profile_blocks.mean(axis=0))
    assert np.max(roi_error) < 1e-5 and np.max(profile_error) < 1e-5
    return dict(metrics=metric_blocks.mean(axis=0),
                metric_sem=metric_blocks.std(axis=0, ddof=1)/np.sqrt(20),
                profile=profile_blocks.mean(axis=0),
                profile_sem=profile_blocks.std(axis=0, ddof=1)/np.sqrt(20),
                roi_radial_discretization_change=roi_error,
                max_profile_radial_discretization_change=float(profile_error.max()),
                charged_fraction=charged_fraction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    sources = json.loads((a.input/'manifest.json').read_text())
    assert len(sources) == 2 and sources[0]['seed'] != sources[1]['seed']
    summaries = []; audits = []; start = time.monotonic()
    for source in sources:
        data = extract(source)
        assert data['audit']['escaping_electrons'] == 0, 'Finite sphere truncates the charged reference'
        summary = summarize(data, data['audit']['actual_energy_MeV'])
        summaries.append(summary); audits.append(data['audit'])
        print(source['name'], summary['metrics'][[0, 4, 5]], time.monotonic()-start, flush=True)
        del data
    train, test = summaries
    report = dict(status='INDEPENDENT_NATIVE_FULL_ELECTRON_REFERENCE', ct_acceptance=False,
                  train_audit=audits[0], test_audit=audits[1],
                  transport_settings=dict(table=sources[0]['table'],
                      density_g_cm3=sources[0]['density_g_cm3'], energy_MeV=sources[0]['energy_MeV'],
                      max_step_mm=sources[0].get('requested_max_step_mm', 0),
                      production_cut_mm=sources[0]['production_cut_mm']),
                  independent_absolute_errors={key:dict(reference=float(test['metrics'][i]),
                      test_block_standard_error=float(test['metric_sem'][i])) for i, key in enumerate(NAMES)},
                  independent_train_absolute_values={key:float(train['metrics'][i]) for i, key in enumerate(NAMES)},
                  train_charged_fraction=train['charged_fraction'], test_charged_fraction=test['charged_fraction'],
                  test_roi_radial_discretization_change=test['roi_radial_discretization_change'].tolist(),
                  max_test_profile_radial_discretization_change=test['max_profile_radial_discretization_change'],
                  radial_bin_mm=.000125, segment_quadrature_order=8,
                  source_manifest_sha256=sha(a.input/'manifest.json'), script_sha256=sha(Path(__file__)),
                  extractor_sha256=sha(Path(__file__).with_name('macro_model.py')), seconds=time.monotonic()-start)
    save(a.out/'report.json', report)
    np.savez_compressed(a.out/'local_profile.npz', x_mm=CENTRES, reference=test['profile'],
                        training=train['profile'], test_sem=test['profile_sem'], training_sem=train['profile_sem'])
    print('COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
