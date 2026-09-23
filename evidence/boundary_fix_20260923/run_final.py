from pathlib import Path
import os, subprocess, json, hashlib, re, time

root = Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
out = root / 'evidence/boundary_fix_20260923'
binary = root / 'build/oneapi-nvidia-minibeam/carbon_mc'
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
env['CARBON_TEST_OUT_DIR'] = str(out / 'localize_final')
Path(env['CARBON_TEST_OUT_DIR']).mkdir(parents=True,exist_ok=True)
with (out / 'localize_final.log').open('x') as log:
    subprocess.run([str(binary.parent / 'urban_localize')], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print('urban_localize PASS', flush=True)
names = ['navigation', 'terminal'] + [f'{scope}_s{i}' for scope in ['water','water_step025','fullchain'] for i in [1,2,3]]
summary = {'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(), 'runs':{}}
for name in names:
    original = root / f'config/boundary_{name}.yaml'
    config = root / f'config/boundary_final_{name}.yaml'
    config.write_text(original.read_text())
    dest = root / f'out/boundary_final_{name}'
    assert not dest.exists(), dest
    start=time.monotonic()
    logpath = out / f'gpu_final_{name}.log'
    with logpath.open('x') as log:
        run = subprocess.run([str(binary),'--config',str(config)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        print(logpath.read_text()[-8000:],flush=True)
        raise RuntimeError(f'{name} exit {run.returncode}')
    q = json.loads((dest/'quality_report.json').read_text())
    ledger = json.loads((dest/'energy_ledger.json').read_text())
    urban = json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})', logpath.read_text()).group(1))
    result = {'exit_code':run.returncode,'wall_s':time.monotonic()-start,'accepted':q['accepted'], 'failures':q['failures'], 'energy_residual_relative':q['relative_energy_residual'],
              'energy_MeV':{k:ledger[k] for k in ['E_in_MeV','E_dep_MeV','E_esc_MeV','E_beamline_MeV','E_untracked_MeV']},'urban':urban,
              'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
    summary['runs'][name]=result
    (out/'final_run_quality.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name, json.dumps(result),flush=True)
    assert q['accepted'] and not q['failures'] and not any(urban[k] for k in ['fatal','cap','guard','subulp'])
    if name=='navigation':
        assert ledger['E_esc_MeV'] > .5*(ledger['E_in_MeV']-ledger['E_beamline_MeV']), 'transverse exits must retain substantial escaped energy'
print('All final GPU runs PASS runtime checks',flush=True)
