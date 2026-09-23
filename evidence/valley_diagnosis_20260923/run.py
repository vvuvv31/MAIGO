from pathlib import Path
import os,json,yaml,subprocess,time,hashlib,re
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/valley_diagnosis_20260923';o.mkdir(exist_ok=True)
binary=r/'build/oneapi-nvidia-minibeam/carbon_mc'
expected=json.loads((r/'evidence/boundary_fix_20260923/final_run_quality.json').read_text())['binary_sha256']
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
spot=o/'source_1m.csv';spot.write_text('spot_id,x,y,energy,weight\n0,0,0,3000.0,1000000\n')
base=yaml.safe_load((r/'config/boundary_final_fullchain_s1.yaml').read_text())
summary={'binary_sha256':expected,'runs':{}}
for label,i in [(label,i) for label in ['stats1m','cu005'] for i in [1,2,3]]:
 name=f'valley_{label}_s{i}';cfg=dict(base)
 if label=='stats1m':cfg.update(number_of_histories=1000000,tps_spots_file=str(spot),random_seed=2026092800+i)
 else:cfg.update(minibeam_copper_max_step_mm=.05,random_seed=2026092600+i)
 if i==1 and label=='stats1m':
  cfg.update(minibeam_phase_space_output_file=str(o/'entrance_s1.csv'),
    minibeam_water_primary_plane_output_file=str(o/'planes_s1.csv'),
    minibeam_water_primary_plane_depths_mm='20, 100, 120',enable_minibeam_primary_c12_roi_scoring=True)
 cp=r/f'config/{name}.yaml';assert not cp.exists();cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
 dest=r/f'out/{name}';assert not dest.exists()
 start=time.monotonic();lp=o/f'{name}.log'
 with lp.open('x') as log:run=subprocess.run([str(binary),'--config',str(cp)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:print(lp.read_text()[-5000:],flush=True);raise RuntimeError((name,run.returncode))
 q=json.loads((dest/'quality_report.json').read_text());e=json.loads((dest/'energy_ledger.json').read_text());u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',lp.read_text()).group(1))
 assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
 summary['runs'][name]={'wall_s':time.monotonic()-start,'accepted':q['accepted'],'energy_residual_relative':q['relative_energy_residual'],'urban':u,'energy_MeV':{k:e[k] for k in ['E_in_MeV','E_dep_MeV','E_esc_MeV','E_beamline_MeV']}}
 (o/'run_quality.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(name,json.dumps(summary['runs'][name]),flush=True)
