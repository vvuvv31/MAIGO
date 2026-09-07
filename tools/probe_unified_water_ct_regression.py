"""Paired CT regression: new water branch OFF, existing electron-r3 path ON."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import config_write, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args();root=args.out.resolve()
    if not args.analyze_only:root.mkdir(exist_ok=False)
    elif not root.is_dir():raise ValueError('Missing saved paired run')
    source=Path('/mnt/sda/wuwei/ct_single_spot_residual_20260906/high/gpu_step025.yaml')
    cfg=yaml.safe_load(source.read_text())
    cfg.update(number_of_histories=50000,tps_histories_scale=1./6.)
    if args.analyze_only:
        if yaml.safe_load((root/'input.yaml').read_text())!=cfg:raise ValueError('Saved source changed')
    else:config_write(root/'input.yaml',cfg)
    joint=Path('/mnt/sda/wuwei/schneider_electron_ct_runtime_r3_20260906/joint_response.csv')
    binaries={'before':Path('build/oneapi-nvidia-electron-switch/carbon_mc').resolve(),
              'after':Path('build/oneapi-nvidia-unified-water/carbon_mc').resolve()}
    initial={key:sha(value) for key,value in binaries.items()}
    if not args.analyze_only:
        for key,binary in binaries.items():run(root/'input.yaml',root/key,binary,joint)
    for key in binaries:
        report=json.loads((root/key/'run_report.json').read_text())
        if report['binary_sha256']!=initial[key] or sha(root/key/'dose.raw')!=report['dose_sha256']:
            raise ValueError('Frozen binary/dose mismatch')
        for path,h in report['inputs'].items():
            if sha(Path(path))!=h:raise ValueError('Saved input changed '+path)
    a=np.fromfile(root/'before/dose.raw','<f4');b=np.fromfile(root/'after/dose.raw','<f4')
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or a.max()<=0:
        raise RuntimeError('Invalid paired dose')
    delta=float(np.max(np.abs(a.astype(float)-b))/a.max())
    if delta>1.e-6:raise RuntimeError('CT dose regression >1e-6 of peak: '+str(delta))
    ledgers={key:json.loads((root/key/'energy_ledger.json').read_text()) for key in binaries}
    x=ledgers['before']['schneider_diagnostics'];y=ledgers['after']['schneider_diagnostics']
    for key in x:
        # JSON can serialize a whole-number MeV sum as an integer. Units, not
        # the incidental Python numeric type, decide whether it is a count.
        if isinstance(x[key],int) and not key.startswith('E_') and 'MeV' not in key and x[key]!=y[key]:
            raise RuntimeError('Changed CT count '+key)
    if ledgers['after']['unified_water_nuclear_transport']:raise RuntimeError('Water branch activated on CT')
    if ledgers['before']['electron_joint_response']['ordered_path_replays']!=ledgers['after']['electron_joint_response']['ordered_path_replays']:
        raise RuntimeError('Changed electron replay count')
    for key,binary in binaries.items():
        if sha(binary)!=initial[key]:raise RuntimeError('Binary changed during test')
    result=dict(status='CT_50K_PAIRED_REGRESSION_PASS_NOT_FULL_VALIDATION',binary_sha256=initial,
                histories=50000,bitwise_dose_equal=bool(np.array_equal(a,b)),max_dose_difference_fraction_peak=delta,
                integer_nuclear_counts_equal=True,electron_replay_count_equal=True,
                energy_diagnostic_differences={k:dict(before=x[k],after=y[k]) for k in x if x[k]!=y[k]})
    with (root/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
