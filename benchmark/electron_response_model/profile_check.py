#!/usr/bin/env python3
"""Local 0.1mm transverse profile check; no fitted dose scaling or registration."""
from pathlib import Path
import json
import numpy as np
from prototype import extract, points, periodic_roi, nodes, save, sha

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'out/electron_mono_reference_20261002'
OUT = ROOT/'out/electron_mono_model_20261002'
DR = .000125
CENTRES = np.arange(-1.75,1.8,.1)


def matrix(n, dr):
    r = (np.arange(n)+.5)*dr
    return np.array([periodic_roi(r,float(x),halfwidth=.05) for x in CENTRES])


def reference_profile(source):
    a,b,w,e,events,exposure,audit = extract(Path(source['path']).parent,electron_source=source)
    rmax = max(np.linalg.norm(a[:,:2],axis=1).max(),np.linalg.norm(b[:,:2],axis=1).max())
    size = 2*(int(np.ceil(rmax/DR))//2+1)
    hist = np.zeros((20,size))
    for r,z,weight in points(a,b,w,8):
        ids = (events%20)*size+(r/DR).astype(np.int64)
        hist += np.bincount(ids,weights=weight,minlength=20*size).reshape(20,size)
    incident = np.bincount(exposure['event']%20,weights=exposure['electron_energy'],minlength=20)
    transform = matrix(size,DR)
    value = transform@hist.sum(axis=0)/incident.sum()
    coarse = matrix(size//2,2*DR)@hist.reshape(20,-1,2).sum(axis=(0,2))/incident.sum()
    blocks = hist@transform.T/incident[:,None]
    se = blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks))
    fraction = exposure['deposited'].sum()/incident.sum()
    assert abs(value.sum()-fraction)<1e-8
    return value,se,coarse


def model_profile(order):
    data = np.load(OUT/'model.npz');b=data['bounds'].astype(float);w=np.diff(np.r_[0.,data['cdf']])
    fraction = json.loads((OUT/'report.json').read_text())['train_energy_fraction']
    value = np.zeros(len(CENTRES))
    for t,weight in zip(*nodes(order)):
        r = .01*np.sinh(b[:,0]+t*(b[:,1]-b[:,0]))
        value += np.array([periodic_roi(r,float(x),halfwidth=.05)@w for x in CENTRES])*weight*fraction
    assert abs(value.sum()-fraction)<1e-8
    return value


def main():
    train_source,test_source = json.loads((INPUT/'manifest.json').read_text())
    train,train_se,train_coarse=reference_profile(train_source)
    test,test_se,test_coarse=reference_profile(test_source)
    model12=model_profile(12);model=model_profile(24)
    difference=(model-test)/test
    uncertainty=2.44*np.hypot(train_se,test_se)/test
    result=dict(scope='Homogeneous lung0.65MeV electron component; ideal periodic grating;0.1mm bins',
                x_mm=CENTRES.tolist(),reference=test.tolist(),model=model.tolist(),
                relative_difference=difference.tolist(),relative_pointwise_uncertainty=uncertainty.tolist(),
                max_abs_relative_difference=float(abs(difference).max()),
                max_abs_model_training_difference=float(abs((model-train)/train).max()),
                max_ref_radial_discretization_change=float(abs((test_coarse-test)/test).max()),
                max_model_quadrature_change=float(abs((model12-model)/model).max()),
                uncertainty_scope='Pointwise estimates, not a simultaneous whole-profile confidence band',
                ct_acceptance=False,source_report_sha256=sha(OUT/'report.json'))
    save(OUT/'local_profile.json',result)
    np.savez_compressed(OUT/'local_profile.npz',x_mm=CENTRES,reference=test,model=model,training=train,
                        training_sem=train_se,test_sem=test_se)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(7,6),sharex=True)
    axes[0].plot(CENTRES,test/.1,label='Independent TOPAS electron histories')
    axes[0].plot(CENTRES,model/.1,'--',label='Joint spatial model')
    axes[0].set_ylabel('Deposited energy fraction / mm');axes[0].legend(fontsize=8)
    axes[1].errorbar(CENTRES,100*difference,yerr=100*uncertainty,fmt='.',capsize=2)
    axes[1].axhline(1,color='gray',ls='--');axes[1].axhline(-1,color='gray',ls='--')
    axes[1].set_ylabel('Relative difference (%)');axes[1].set_xlabel('Transverse position in one pitch (mm)')
    fig.suptitle('Homogeneous lung, 0.65 MeV electrons; not a CT acceptance test')
    fig.tight_layout();fig.savefig(OUT/'local_profile.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k.startswith('max_')}),flush=True)


if __name__ == '__main__':
    main()
