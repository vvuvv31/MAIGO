from pathlib import Path
import os,yaml,json,time,subprocess,re,hashlib,numpy as np
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';binary=r/'build/oneapi-nvidia-minibeam/carbon_mc';before=o/'carbon_mc_before_delta_fix'
env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
base=yaml.safe_load((r/'config/percent1_delta_tail_smoke.yaml').read_text());local=dict(base);local['minibeam_water_delta_response_model']='local'
for k in ['minibeam_water_delta_response_table_file','water_electron_response_sha256','water_electron_response_metadata_sha256']:local.pop(k)
summary={'old_binary_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),'fixed_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'runs':{}}
assert summary['old_binary_sha256']!=summary['fixed_binary_sha256']
def run(name,cfg,exe):
 cp=r/f'config/{name}.yaml';assert not cp.exists();cp.write_text(yaml.safe_dump(cfg,sort_keys=False));lp=o/f'{name}.log';start=time.monotonic()
 with lp.open('x') as log:p=subprocess.run([str(exe),'--config',str(cp)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:print(lp.read_text()[-3500:],flush=True);raise RuntimeError(name)
 dest=r/f'out/{name}';q=json.loads((dest/'quality_report.json').read_text());u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',lp.read_text()).group(1));assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
 ledger=json.loads((dest/'energy_ledger.json').read_text())
 summary['runs'][name]={'histories':cfg['number_of_histories'],'seed':cfg['random_seed'],'wall_s':time.monotonic()-start,'quality':q,'urban':u,'energy':{k:ledger[k] for k in ['E_in_MeV','E_dep_MeV','E_dep_in_grid_MeV','E_dep_outside_grid_MeV','E_esc_MeV','E_beamline_MeV']}}
 (o/'delta_fixed_quality.json').write_text(json.dumps(summary,indent=2)+'\n');print('FIXED',name,summary['runs'][name]['wall_s'],'residual',q['relative_energy_residual'],flush=True)
 return np.fromfile(dest/'dose.raw',dtype='<f4').astype(float)
a=run('percent1_delta_control_before',local,before);b=run('percent1_delta_control_after',local,binary)
reg=float(np.max(abs(a-b))/a.max());assert reg<2e-5
c=run('percent1_delta_fixed_smoke',base,binary);explicit=dict(base);explicit.update(voxel_bins_z=1000,voxel_size_z_mm=.25)
f=run('percent1_delta_fixed_explicit',explicit,binary);grid=float(np.max(abs(c-f))/c.max());assert grid<2e-5
summary['regression_checks']={'baseline_before_after_max_difference_over_peak':reg,'legacy_vs_explicit_z_max_difference_over_peak':grid,'tolerance':2e-5}
(o/'delta_fixed_quality.json').write_text(json.dumps(summary,indent=2)+'\n')
for i in [1,2,3]:
 cfg=dict(base);cfg.update(number_of_histories=1000000,random_seed=2026092800+i,tps_spots_file=str(r/'evidence/valley_diagnosis_20260923/source_1m.csv'))
 run(f'percent1_delta_fixed_s{i}',cfg,binary)
print('ALL_FIXED_DONE',flush=True)
