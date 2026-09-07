"""Single-variable EM diagnostic. Disabling fluctuations is NOT a physics fix."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from evaluate_topas10x_gpu_gamma import pass_mask
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import config_write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat-seed', type=int)
    args = parser.parse_args()
    reference = Path('/mnt/sda/wuwei/ct_clinical_em_control_20260907')
    out = Path('/mnt/sda/wuwei/ct_clinical_em_no_straggling_20260907')
    if args.repeat_seed is not None:
        out = out.with_name(f'ct_clinical_em_seed{args.repeat_seed}_20260907')
    prior = json.loads((reference / 'comparison.json').read_text())
    report = json.loads((reference / 'gpu/run_report.json').read_text())
    binary = Path('build/oneapi-nvidia-electron-bounds/carbon_mc').resolve()
    for path, expected in report['inputs'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Baseline input changed: ' + path)
    if sha(binary) != report['binary_sha256']:
        raise ValueError('Binary changed')
    if sha(reference / 'dose_topas.bin') != prior['reference_sha256']:
        raise ValueError('Reference changed')
    cfg = yaml.safe_load((reference / 'gpu/run.yaml').read_text())
    if not cfg['enable_energy_straggling'] or cfg['enable_inelastic']:
        raise ValueError('Not the intended EM baseline')
    changed_key = 'enable_energy_straggling' if args.repeat_seed is None else 'random_seed'
    if args.repeat_seed is not None and args.repeat_seed == cfg['random_seed']:
        raise ValueError('Repeat seed must be independent')
    cfg[changed_key] = False if args.repeat_seed is None else args.repeat_seed
    out.mkdir(exist_ok=False)
    config_write(out / 'input.yaml', cfg)
    joint = Path(cfg['ct_electron_joint_response_diagnostic_file'])
    run(out / 'input.yaml', out / 'gpu', binary, joint)
    actual = yaml.safe_load((out / 'gpu/run.yaml').read_text())
    old = yaml.safe_load((reference / 'gpu/run.yaml').read_text())
    changed = {k for k in old.keys() | actual.keys() if old.get(k) != actual.get(k)}
    if changed != {changed_key}:
        raise ValueError('Unpaired configuration: ' + str(changed))
    ref = np.fromfile(reference / 'dose_topas.bin', '<f8').reshape((42, 607, 960))
    dose = np.fromfile(out / 'gpu/dose.raw', '<f4').reshape(ref.shape).astype(float)
    if not np.isfinite(dose).all() or np.any(dose < 0):
        raise ValueError('Invalid dose')
    pts = np.argwhere(ref >= .1 * ref.max())
    ix = tuple(pts.T)
    metrics = dict(sum_ratio=float(dose.sum() / ref.sum()),
                   rms_error_pct_peak=float(np.sqrt(np.mean((dose[ix] - ref[ix]) ** 2)) / ref.max() * 100))
    for dd, dta in ((3, 3), (2, 2), (1, 1), (3, 0)):
        for local in (False, True):
            key = f'{"local" if local else "global"}_{dd}{dta}'
            metrics[key] = float(pass_mask(dose, ref, pts, np.array([2., .5, .5]), dd, dta, local).mean() * 100)
    result = dict(status='ABLATION_ONLY_NOT_A_PRODUCTION_CANDIDATE', metrics=metrics,
                  baseline=prior['metrics'], changed_keys=sorted(changed),
                  dose_sha256=sha(out / 'gpu/dose.raw'), reference_sha256=prior['reference_sha256'],
                  limitation='Same source/seed/binary; disabling fluctuations also changes trajectories and electron response queries. Not a proof of model error.')
    if args.repeat_seed is not None:
        if sha(reference / 'gpu/dose.raw') != prior['gpu_sha256']:
            raise ValueError('Baseline dose changed')
        first = np.fromfile(reference / 'gpu/dose.raw', '<f4').reshape(ref.shape)
        result['status'] = 'GPU_SEED_NOISE_CONTROL'
        result['limitation'] = 'Two GPU seeds only; TOPAS noise remains unmeasured for this spot. No noise-subtracted Gamma claim.'
        result['dose_bands'] = {}
        for lo, hi in ((.1, .2), (.2, .5), (.5, 1.01)):
            mask = (ref >= lo * ref.max()) & (ref < hi * ref.max())
            # Difference/sqrt(2) estimates ONE 300k run, not the mean.
            noise = (dose[mask] - first[mask]) / (np.sqrt(2.) * ref[mask])
            error = (first[mask] - ref[mask]) / ref[mask]
            result['dose_bands'][f'{lo}-{hi}'] = dict(voxels=int(mask.sum()),
                baseline_local_error_rms_pct=float(np.sqrt(np.mean(error**2))*100),
                gpu_single_run_noise_rms_pct=float(np.sqrt(np.mean(noise**2))*100),
                baseline_signed_local_mean_pct=float(error.mean()*100))
    with (out / 'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
