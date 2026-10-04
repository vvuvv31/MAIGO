#!/usr/bin/env python3
"""Fit to one TOPAS electron run; evaluate against an independent seed."""
from pathlib import Path
import json
import time
import numpy as np
from prototype import extract, reference, fit, model_metrics, error, save, sha, NAMES

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'out/electron_mono_reference_20261002'
OUT = ROOT/'out/electron_mono_model_20261002'


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((INPUT/'manifest.json').read_text())
    train, test = manifest
    assert train['name'] == 'train' and test['name'] == 'test'
    start = time.monotonic()
    a,b,w,e,events,exposure,audit = extract(Path(train['path']).parent,electron_source=train)
    print('train extracted',len(w),flush=True)
    ref = reference(a,b,w,8); ref4 = reference(a,b,w,4)
    print('train reference ready',flush=True)
    candidates = []
    for step in [.04,.02,.01,.005]:
        bounds,cdf = fit(a,b,w,step,order=4)
        metrics = model_metrics(bounds,cdf)
        errors = error(ref,metrics)
        passed = all(abs(errors[k]['relative_difference']) < .005 if errors[k]['relative_difference'] is not None
                     else abs(errors[k]['difference']) < 1e-12 for k in ['r2','z2','peak','valley'])
        candidates.append(dict(step=step,cells=len(cdf),representation_pass=passed,errors=errors))
        print('candidate',step,len(cdf),passed,flush=True)
        if passed:break
    np.savez_compressed(OUT/'model.npz',bounds=bounds,cdf=cdf,energy_MeV=train['energy_MeV'])
    train_deposit_fraction = float(exposure['deposited'].sum()/exposure['electron_energy'].sum())
    report = dict(status='INDEPENDENT_HOMOGENEOUS_ELECTRON_CHECK',ct_acceptance=False,
                  source_manifest_sha256=sha(INPUT/'manifest.json'),script_sha256=sha(Path(__file__)),
                  prototype_sha256=sha(Path(__file__).with_name('prototype.py')),
                  train_audit=audit,train_energy_fraction=train_deposit_fraction,
                  quadrature_4_vs_8=error(ref,ref4),candidates=candidates,
                  runtime_payload_bytes=len(cdf)*24,model_sha256=sha(OUT/'model.npz'))
    save(OUT/'report.json',report)
    del a,b,w,e,events,exposure
    a,b,w,e,events,exposure,audit = extract(Path(test['path']).parent,electron_source=test)
    print('independent test extracted',len(w),flush=True)
    observed = reference(a,b,w,8)
    # Twenty disjoint blocks of independently generated primary histories.
    # Estimate uncertainty in normalized moments and absolute dose fractions.
    normalized = []; absolute = []
    for block in range(20):
        take = events % 20 == block
        values = reference(a[take],b[take],w[take],4)
        incident = exposure['electron_energy'][exposure['event'] % 20 == block].sum()
        normalized.append(values);absolute.append(values*w[take].sum()/incident)
    absolute = np.array(absolute)
    test_fraction = float(exposure['deposited'].sum()/exposure['electron_energy'].sum())
    test_absolute = observed*test_fraction
    model_absolute = metrics*train_deposit_fraction
    se = absolute.std(axis=0,ddof=1)/np.sqrt(len(absolute))
    errors = error(test_absolute,model_absolute)
    for i,name in enumerate(NAMES):
        errors[name]['test_block_standard_error'] = float(se[i])
        errors[name]['relative_95_halfwidth_test_only'] = float(2.093*se[i]/abs(test_absolute[i])) if abs(test_absolute[i]) > 1e-12 else None
    report.update(test_audit=audit,test_energy_fraction=test_fraction,
                  independent_normalized_errors=error(observed,metrics),independent_absolute_errors=errors,
                  point_estimate_peak_valley_within_1pct=all(abs(errors[k]['relative_difference']) <= .01 for k in ['peak','valley']),
                  uncertainty_note='Test block intervals only; training uncertainty and energy interpolation/interface bias are not bounded',
                  seconds=time.monotonic()-start)
    save(OUT/'report.json',report)
    print('complete',json.dumps({k:errors[k] for k in ['r2','peak','valley']}),flush=True)


if __name__ == '__main__':
    main()
