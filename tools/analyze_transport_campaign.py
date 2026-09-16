#!/usr/bin/env python3
"""Conservative timing/dose gates for benchmark_single_gpu output.

This report cannot authorize promotion: hardware and independent cases are also
required. Profiler timing must never be supplied to this tool.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def paired_stats(base, candidate):
    base, candidate = np.asarray(base), np.asarray(candidate)
    if len(base) != len(candidate) or len(base) < 5:
        raise ValueError('At least five complete pairs required')
    gains = base / candidate - 1
    # Deterministic paired bootstrap; do not confuse it with independent runs.
    rng = np.random.default_rng(20260916)
    samples = rng.choice(gains, (20000, len(gains)), replace=True).mean(axis=1)
    return dict(base_median_s=float(np.median(base)), candidate_median_s=float(np.median(candidate)),
                mean_paired_gain=float(gains.mean()), paired_gain_ci95=np.quantile(samples, [.025, .975]).tolist(),
                paired_gains=gains.tolist())


def analyze(out):
    rows = [r for r in json.loads((out / 'results.json').read_text()) if not r['warmup']]
    base = sorted((r for r in rows if r['kind'] == 'base'), key=lambda r: r['index'])
    cand = sorted((r for r in rows if r['kind'] == 'cand'), key=lambda r: r['index'])
    if len(base) < 5:
        raise ValueError('Five baseline repeats required for the dose envelope')
    def dose(r):
        directory = out / f"run_{r['index']:02d}_{r['kind']}"
        header = dict(line.split(' = ', 1) for line in (directory / 'dose.mhd').read_text().splitlines() if ' = ' in line)
        if header.get('NDims') != '3' or header.get('ElementType') != 'MET_FLOAT':
            raise ValueError('Expected 3D float dose')
        return header, np.fromfile(directory / 'dose.raw', dtype='<f4').astype(np.float64)
    header, ref = dose(base[0])
    lo, hi = ref.copy(), ref.copy()
    for row in base[1:]:
        h, d = dose(row)
        if h != header or d.shape != ref.shape or not np.isfinite(d).all():
            raise ValueError('Dose grid mismatch or invalid dose')
        lo, hi = np.minimum(lo, d), np.maximum(hi, d)
    envelope = float(np.max(hi-lo))
    result = dict(baseline_repeats=len(base), candidate_repeats=len(cand),
                  baseline_envelope_abs=envelope, baseline_envelope_pct_peak=envelope/ref.max()*100,
                  audit_identical=all(r['audit'] == base[0]['audit'] for r in rows),
                  quality_pass=all(r['quality']['accepted'] and not r['quality']['failures'] and not r['quality']['queue_overflow_count'] for r in rows),
                  promotion_allowed=False,
                  outstanding_gates=['matched hardware repetitions', 'other CT cases and 50k water', 'path and sampling equivalence'])
    if cand:
        if [r['index'] for r in base] != [r['index'] for r in cand]:
            raise ValueError('Unmatched run pairs')
        maximum, outside = 0., 0
        for row in cand:
            h, d = dose(row)
            if h != header or d.shape != ref.shape or not np.isfinite(d).all():
                raise ValueError('Dose grid mismatch or invalid dose')
            maximum = max(maximum, float(np.max(np.abs(d-ref))))
            outside = max(outside, int(np.count_nonzero((d < lo) | (d > hi))))
        result.update(max_candidate_diff_abs=maximum, max_candidate_diff_pct_peak=maximum/ref.max()*100,
                      dose_global_envelope_pass=maximum <= envelope,
                      max_voxels_outside_baseline_interval=outside)
        result['timing'] = {key: paired_stats([r[key] for r in base], [r[key] for r in cand])
                            for key in ('wall_s', 'elapsed_s', 'primary_s', 'secondary_s')}
        timing = result['timing']
        result['throughput_gate_pass'] = all(timing[key]['mean_paired_gain'] >= .05 and
                                             timing[key]['paired_gain_ci95'][0] > 0
                                             for key in ('wall_s', 'elapsed_s'))
        result['extend_to_ten_pairs'] = len(base) < 10 and any(
            timing[key]['paired_gain_ci95'][0] <= .05 <= timing[key]['paired_gain_ci95'][1]
            for key in ('wall_s', 'elapsed_s'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = analyze(a.directory)
    (a.directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
