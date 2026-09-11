#!/usr/bin/env python3
"""Evaluate a completed final-stopping run against the frozen TOPAS 3D dose."""
import argparse
import json
from pathlib import Path
import numpy as np
from compile_schneider_ion_stopping import sha
from evaluate_topas10x_gpu_gamma import pass_mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    a = p.parse_args()
    output = a.run / 'gamma_coarse.json'
    if output.exists():
        raise ValueError('Refuse overwrite')
    state = json.loads((a.run / 'execution.json').read_text())
    manifest = json.loads((a.reference / 'manifest.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('Run incomplete')
    if sum(x['histories'] for x in state['completed']) != manifest['histories']:
        raise ValueError('History count mismatch')
    if sha(a.run / 'gpu_sum.raw') != state['aggregate_sha256']:
        raise ValueError('GPU hash mismatch')
    if sha(a.reference / 'topas_sum.raw') != manifest['reference_sum_sha256']:
        raise ValueError('Reference hash mismatch')
    gpu = np.fromfile(a.run / 'gpu_sum.raw', '<f4').reshape(manifest['gpu_shape_zyx'])
    if manifest['mapping'] == 'packed_xneg':
        gpu = np.flip(gpu.transpose(1, 2, 0), axis=2)
    elif manifest['mapping'] != 'native':
        raise ValueError('Unknown mapping')
    ref = np.fromfile(a.reference / 'topas_sum.raw', '<f4').reshape(manifest['topas_shape_zyx'])
    if gpu.shape != ref.shape or not np.all(np.isfinite(gpu) & (gpu >= 0)):
        raise ValueError('Invalid dose')
    mask = ref >= .1 * ref.max()
    pts = np.argwhere(mask)
    results = {}
    for dd, dta in [(3, 3), (2, 2), (1, 1), (3, 0)]:
        for local in [False, True]:
            key = f"{'local' if local else 'global'}_{dd}pct_{dta}mm"
            passed = pass_mask(gpu, ref, pts, np.array(manifest['spacing_zyx']), dd, dta, local)
            results[key] = {'full_mask_percent': 100 * float(passed.mean()),
                            'passed': int(passed.sum()), 'evaluated': len(pts)}
            print(key, results[key]['full_mask_percent'], flush=True)
    bands = []
    for lo, hi in [(10, 20), (20, 50), (50, 80), (80, 101)]:
        selected = (ref >= ref.max()*lo/100) & (ref < ref.max()*hi/100)
        if not selected.any():
            continue
        error = 100 * (gpu[selected].astype(float)-ref[selected]) / ref[selected]
        bands.append({'reference_percent_max': [lo, hi], 'voxels': int(selected.sum()),
                      'mean_local_error_pct': float(error.mean()),
                      'rms_local_error_pct': float(np.sqrt(np.mean(error**2))),
                      'negative_beyond_3pct_percent': 100*float((error < -3).mean()),
                      'positive_beyond_3pct_percent': 100*float((error > 3).mean())})
    report = {'case': manifest['case'], 'histories': manifest['histories'],
              'status': 'RESEARCH_VALIDATION',
              'physics': 'primary midpoint ON; secondary material stopping ON; secondary exact faces ON',
              'table_sha256': state['table_sha256'], 'binary_sha256': state['binary_sha256'],
              'method': 'reference >=10% maximum; 0.5 mm spherical lattice, trilinear interpolation; 0 mm is same-voxel dose difference',
              'dose_scale': 1.0, 'gamma': results, 'dose_bands': bands,
              'dose_sum_ratio': float(gpu.sum(dtype=float)/ref.sum(dtype=float)),
              'overflow_attempts_excluded': state['overflow_attempts'],
              'gpu_sha256': state['aggregate_sha256'], 'reference_sha256': manifest['reference_sum_sha256'],
              'source_sha256': sha(__file__)}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
