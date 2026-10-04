#!/usr/bin/env python3
"""Distribution and FP32 output checks of the isolated GPU response sampler."""
import json
from pathlib import Path
import re
import numpy as np
from prototype import model_metrics, sha

OUT = Path('out/electron_parametric_v2_20261002')


def main():
    results = []
    for run in json.loads((OUT/'gpu_runs.json').read_text()):
        name, energy_bin = run['name'], run['energy_bin']
        c = next(c for c in json.loads((OUT/(name+'_channels.json')).read_text()) if c['energy_bin'] == energy_bin)
        p = OUT/f'gpu_{name}_{energy_bin}'
        samples = np.fromfile(p.with_suffix('.samples'), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('cell','<u4')])
        assert len(samples) == 2000000
        model = np.load(OUT/(name+'.npz'))
        sl = slice(c['offset'], c['offset']+c['count'])
        bounds, cdf = model['bounds'][sl], model['cdf'][sl]
        x, y, z = [samples[n].astype(float) for n in ['x', 'y', 'z']]
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()
        assert samples['cell'].max() < len(cdf)
        ref = model_metrics(bounds, cdf)
        r = np.hypot(x, y)
        fields = {'r2':(r*r,ref[0]),'z':(z,ref[1]),'z2':(z*z,ref[2]),'r_z':(r*z,ref[3]),
                  'azimuth_x':(x,0.),'azimuth_y':(y,0.),'azimuth_xy':(x*y,0.),'azimuth_x2_minus_y2':(x*x-y*y,0.)}
        checks = {}
        for key, (values, expected) in fields.items():
            se = float(values.std(ddof=1)/np.sqrt(len(values)))
            mean = float(values.mean()); difference = mean-float(expected)
            assert abs(difference) <= 6*se+1e-7*max(1,abs(expected)), (name,energy_bin,key,mean,expected,se)
            checks[key] = dict(mean=mean,expected=float(expected),standard_error=se,z_score=difference/se if se else 0)
        # Group tiny-probability cells into 32 roughly equal probability bins.
        group = np.minimum((cdf*32).astype(int),31)
        observed = np.bincount(group[samples['cell']], minlength=32)/len(samples)
        expected = np.bincount(group,weights=np.diff(np.r_[0.,cdf]),minlength=32)
        se = np.sqrt(expected*(1-expected)/len(samples))
        assert np.all(abs(observed-expected) <= 6*se+1/len(samples))
        text = p.with_suffix('.log').read_text()
        timing = [float(v) for v in re.findall(r'kernel_seconds=([\d.eE+-]+)',text)]
        assert len(timing) == 4 and 'fp32_score_sum=2e+06' in text
        results.append(dict(name=name,energy_bin=energy_bin,checks=checks,cdf_32_groups_pass=True,
                            sample_count=len(samples),median_kernel_seconds=float(np.median(timing[1:])),
                            samples_per_second=float(len(samples)/np.median(timing[1:])),
                            scope='Homogeneous response sampling plus FP32 diagnostic histogram; not full transport throughput'))
    result = dict(accepted=True,results=results,binary_sha256=sha(OUT/'sampler'))
    (OUT/'gpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(accepted=True,range_samples_per_second=[min(r['samples_per_second'] for r in results),max(r['samples_per_second'] for r in results)])))


if __name__ == '__main__':
    main()
