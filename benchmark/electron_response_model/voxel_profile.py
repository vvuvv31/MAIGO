#!/usr/bin/env python3
"""Reduce audited native voxel histories to exact lab-x chord profiles.

The source slit is already sampled by TOPAS. Do not azimuthally average or
convolve another slit. A 3.6 mm periodic fold uses 36 bins of width 0.1 mm.
These are offline reference statistics, not GPU dose scoring.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import extract
from prototype import save, sha

WIDTH = .1
NBINS = 36
LEFT = -1.8
CENTRES = LEFT + WIDTH*(np.arange(NBINS)+.5)
PEAK = np.arange(16, 20)
VALLEY = np.array([0, 1, 34, 35])


def chord_profile(pre_x, post_x, weights, events, histories):
    """Integrate each nonnegative segment weight exactly over x-bin overlaps."""
    low = np.minimum(pre_x, post_x)
    high = np.maximum(pre_x, post_x)
    length = high-low
    first = np.floor((low-LEFT)/WIDTH).astype(np.int64)
    last = np.floor((high-LEFT)/WIDTH).astype(np.int64)
    profile = np.zeros((histories, NBINS))
    assert np.all(weights >= 0) and np.all(events >= 0) and np.all(events < histories)
    for offset in range(int(np.max(last-first, initial=0))+1):
        selected = first+offset <= last
        ids = np.flatnonzero(selected)
        index = first[ids]+offset
        fraction = np.ones(len(ids))
        moving = length[ids] > 0
        a = np.maximum(low[ids[moving]], LEFT+index[moving]*WIDTH)
        b = np.minimum(high[ids[moving]], LEFT+(index[moving]+1)*WIDTH)
        fraction[moving] = np.maximum(0, b-a)/length[ids[moving]]
        np.add.at(profile, (events[ids], index % NBINS), weights[ids]*fraction)
    expected = np.bincount(events, weights=weights, minlength=histories)
    assert np.max(abs(profile.sum(axis=1)-expected), initial=0) < 1e-10
    return profile


def summarize(profile, energy):
    normalized = profile/energy
    rois = np.column_stack([normalized[:, PEAK].sum(axis=1), normalized[:, VALLEY].sum(axis=1)])
    n = len(profile)
    return dict(histories=n, mean=normalized.mean(axis=0),
                sem=normalized.std(axis=0, ddof=1)/np.sqrt(n),
                roi_mean=rois.mean(axis=0), roi_sem=rois.std(axis=0, ddof=1)/np.sqrt(n))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    sources = json.loads((args.input/'manifest.json').read_text())
    assert len(sources) == 2 and sources[0]['seed'] != sources[1]['seed']
    results = []; audits = []
    for source in sources:
        assert source['geometry'] == 'voxel_box_in_vacuum'
        data = extract(source)
        profile = chord_profile(data['pre'][:, 0], data['post'][:, 0], data['weight'],
                                data['event'], source['histories'])
        assert np.max(abs(profile.sum(axis=1)-data['ledger'][:, 0])) < 1e-10
        result = summarize(profile, data['audit']['actual_energy_MeV'])
        np.savez_compressed(args.out/(source['name']+'_histories.npz'),
                            profile_MeV=profile, ledger_MeV=data['ledger'])
        results.append(result); audits.append(data['audit'])
        print(source['name'], result['roi_mean'], result['roi_sem'], flush=True)
        del data
    train, test = results
    np.savez_compressed(args.out/'local_profile.npz', x_mm=CENTRES,
                        reference=test['mean'], test_sem=test['sem'],
                        training=train['mean'], training_sem=train['sem'])
    report = dict(status='NATIVE_VOXEL_LAB_X_REFERENCE', ct_acceptance=False,
                  scope='Homogeneous voxelized electron reference; photon branch is an explicit sink',
                  profile_method='Uniform along native charged step chord, exact lab-x overlaps; no rotational averaging',
                  source_manifest_sha256=sha(args.input/'manifest.json'),
                  script_sha256=sha(Path(__file__)), extractor_sha256=sha(Path(__file__).with_name('macro_model.py')),
                  audits=audits, roi_names=['peak', 'valley'],
                  train_roi=train['roi_mean'].tolist(), test_roi=test['roi_mean'].tolist(),
                  train_roi_sem=train['roi_sem'].tolist(), test_roi_sem=test['roi_sem'].tolist(),
                  peak_valley_windows_mm=[[-.2,.2], [1.6,2.0]],
                  pitch_mm=3.6, profile_bin_mm=WIDTH, source_settings=sources[0])
    save(args.out/'report.json', report)


if __name__ == '__main__':
    main()
