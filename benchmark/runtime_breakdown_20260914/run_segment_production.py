"""Validate production continuation ON/OFF; GPU runs serialized by common runner."""
from pathlib import Path
import json, os, subprocess, sys
import numpy as np
import yaml
repo = Path(__file__).resolve().parents[2]
root = repo / 'scratch/runtime_breakdown_20260914'
runs = repo / 'scratch/unified_em_perf_20260913'
exe = repo / 'build/oneapi-nvidia-secondary-continuation/carbon_mc'
out = Path(__file__).with_name('segment_production_results.json')
rows = []
for case, inputs, tag, reference in [
    ('RT07575', root/'segmented_inputs_50k', 'ct50k', 'runtime_segment_base_50k'),
    ('b1_200', root/'segment_validation_inputs', 'water50k', 'segment_validation_v2_base'),
    ('RT07575', root/'inputs', 'ct1m', 'runtime_segment_base_1m_r1'),
]:
    pair = []
    for enabled in (False, True):
        label = f'segment_production_{tag}_{"on" if enabled else "off"}'
        dest = runs/label/case
        if not (dest/'status.json').exists():
            subprocess.run([sys.executable, 'tools/verify_schneider_v2_1_data.py'], cwd=repo, check=True)
            env = dict(os.environ, CARBON_RUNTIME_BREAKDOWN='1', CARBON_BENCH_PROFILE='0',
                       CARBON_BENCH_CONFIG_ROOT=str(inputs),
                       CARBON_BENCH_CONFIG_OVERRIDES=json.dumps({'secondary_step_chunking': enabled}))
            subprocess.run([sys.executable, 'benchmark/unified_em_performance_20260913/run.py',
                            label, str(exe), case], cwd=repo, env=env, check=True)
        status = json.loads((dest/'status.json').read_text())
        assert status['complete'], dest
        log = (dest/'gpu.log').read_text()
        assert ('[segmented-secondary]' in log) == enabled
        quality = status['quality']
        assert ('secondary_step_chunking_accepted' in json.dumps(quality)) == enabled
        pair.append((dest, status))
    a, sa = pair[0]; b, sb = pair[1]
    ref = runs/reference/case
    sr = json.loads((ref/'status.json').read_text())
    ca = yaml.safe_load((a/'gpu.yaml').read_text()); cb = yaml.safe_load((b/'gpu.yaml').read_text())
    ca.pop('secondary_step_chunking'); cb.pop('secondary_step_chunking'); assert ca == cb
    for key in ('audit', 'steps'): assert sa[key] == sb[key] == sr[key], (tag, key)
    ledgers = [json.loads((p/'out/gpu/energy_ledger.json').read_text()) for p in (a,b,ref)]
    for k,v in ledgers[0].items():
        if isinstance(v,int): assert v == ledgers[1][k] == ledgers[2][k], (tag,k)
    doses = [np.fromfile(p/'dose_gpu.raw', '<f4').astype(float) for p in (a,b,ref)]
    assert all(d.shape == doses[0].shape and np.isfinite(d).all() for d in doses)
    peak = doses[2].max()
    diff = lambda x,y: float(100*np.max(np.abs(x-y))/peak)
    ab = diff(doses[0], doses[1]); off_ref = diff(doses[0],doses[2]); on_ref = diff(doses[1],doses[2])
    assert max(ab,off_ref,on_ref) < .001, (tag,ab,off_ref,on_ref)
    rows.append(dict(case=case,tag=tag,baseline=sa,candidate=sb,
                     max_dose_difference_percent_peak=ab, off_vs_frozen_percent_peak=off_ref,
                     on_vs_frozen_percent_peak=on_ref, gain_percent=100*(sb['throughput']/sa['throughput']-1)))
    out.write_text(json.dumps(rows,indent=2)+'\n')
    print('PRODUCTION',tag,'gain%',rows[-1]['gain_percent'],'dose%',ab,flush=True)

# Also exercise the README's usual build with the accepted all-ion elastic case.
label = 'segment_production_release_elastic1m'
dest = runs/label/'RT07575'
release_exe = repo/'build/oneapi-nvidia-release/carbon_mc'
if not (dest/'status.json').exists():
    subprocess.run([sys.executable, 'tools/verify_schneider_v2_1_data.py'], cwd=repo, check=True)
    env = dict(os.environ, CARBON_RUNTIME_BREAKDOWN='1', CARBON_BENCH_PROFILE='0',
               CARBON_BENCH_CONFIG_ROOT=str(root/'segment_elastic_1m_inputs'),
               CARBON_BENCH_CONFIG_OVERRIDES=json.dumps({'secondary_step_chunking': True}))
    subprocess.run([sys.executable, 'benchmark/unified_em_performance_20260913/run.py',
                    label, str(release_exe), 'RT07575'], cwd=repo, env=env, check=True)
reference = runs/'segment_elastic_base_1m'/'RT07575'
sb = json.loads((dest/'status.json').read_text()); sa = json.loads((reference/'status.json').read_text())
assert sb['complete'] and '[segmented-secondary]' in (dest/'gpu.log').read_text()
for k in ('audit','steps'): assert sa[k] == sb[k], k
ca = yaml.safe_load((reference/'gpu.yaml').read_text()); cb = yaml.safe_load((dest/'gpu.yaml').read_text())
cb.pop('secondary_step_chunking'); assert ca == cb
la = json.loads((reference/'out/gpu/energy_ledger.json').read_text())
lb = json.loads((dest/'out/gpu/energy_ledger.json').read_text())
for k,v in la.items():
    if isinstance(v,int): assert v == lb[k], k
x = np.fromfile(reference/'dose_gpu.raw','<f4').astype(float)
y = np.fromfile(dest/'dose_gpu.raw','<f4').astype(float)
assert x.shape == y.shape and np.isfinite(y).all()
delta = float(100*np.max(np.abs(x-y))/x.max()); assert delta < .001, delta
result = dict(baseline=sa,candidate=sb,max_dose_difference_percent_peak=delta,
              gain_percent=100*(sb['throughput']/sa['throughput']-1))
Path(__file__).with_name('segment_production_release_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('PRODUCTION release elastic 1M',result['gain_percent'],delta,flush=True)
