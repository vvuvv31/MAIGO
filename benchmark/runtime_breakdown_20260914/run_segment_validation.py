from pathlib import Path
import subprocess,sys,os,json,yaml,numpy as np
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';inputs=root/'segment_validation_inputs';out=Path(__file__).resolve().parent;runs=repo/'scratch/unified_em_perf_20260913'
cases=['b1_200','b3_layers','b4_soft_lung','b4_soft_bone','RT07575_elastic']
for case in cases:
 if case=='RT07575_elastic':
  cfg=yaml.safe_load((root/'segmented_inputs_50k/RT07575/gpu.yaml').read_text());extra=yaml.safe_load((repo/'config/rt07575_unified_em_research.yaml').read_text())
  for k in ['all_ion_elastic_file','all_ion_elastic_sha256','elastic_recoil_stopping_file','elastic_recoil_stopping_sha256']:cfg[k]=extra[k]
  cfg['run_mode']='research'
 else:cfg=yaml.safe_load((repo/'benchmark/benchmark20260913'/case/'gpu.yaml').read_text())
 if case=='b1_200':
  cfg.pop('primary_joint_em_data_directory',None)
  unified=yaml.safe_load((repo/'config/unified_water_production.yaml').read_text())
  for k in ['em_model','em_package_file','em_package_sha256','primary_em_model','enable_energy_straggling','enable_secondary_energy_straggling','straggling_scale','ct_secondary_exact_faces']:cfg[k]=unified[k]
 cfg.update(number_of_histories=50000,enable_secondary_unified_em=True,secondary_species_grouping=True)
 folder=inputs/case;folder.mkdir(parents=True,exist_ok=True);(folder/'gpu.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
rows=[]
for case in cases:
 for label,build in [('segment_validation_v2_base','mean_guard_build'),('segment_validation_v2_64','segmented_build')]:
  dest=runs/label/case
  if (dest/'status.json').exists():assert json.loads((dest/'status.json').read_text())['complete'];continue
  subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
  env=dict(os.environ,CARBON_BENCH_PROFILE='0',CARBON_RUNTIME_BREAKDOWN='1',CARBON_BENCH_CONFIG_ROOT=str(inputs),CARBON_BENCH_CONFIG_OVERRIDES='{}')
  subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(root/build/'carbon_mc'),case],cwd=repo,env=env,check=True)
 a=runs/'segment_validation_v2_base'/case;b=runs/'segment_validation_v2_64'/case;sa=json.loads((a/'status.json').read_text());sb=json.loads((b/'status.json').read_text());assert sa['config_sha256']==sb['config_sha256']
 x=np.fromfile(a/'dose_gpu.raw','<f4').astype(float);y=np.fromfile(b/'dose_gpu.raw','<f4').astype(float);assert x.shape==y.shape and np.isfinite(y).all();diff=100*abs(x-y).max()/x.max()
 la=json.loads((a/'out/gpu/energy_ledger.json').read_text());lb=json.loads((b/'out/gpu/energy_ledger.json').read_text());counts={k:[v,lb[k]] for k,v in la.items() if isinstance(v,int)}
 ok=sa['audit']==sb['audit'] and sa['steps']==sb['steps'] and all(v[0]==v[1] for v in counts.values()) and diff<.001
 row=dict(case=case,passed=bool(ok),baseline=sa,candidate=sb,max_dose_difference_percent_peak=diff,ledger_integer_counts=counts);rows.append(row);print('VALIDATION',case,ok,'dose%',diff,flush=True)
 (out/'segment_validation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
assert all(x['passed'] for x in rows),'one or more equivalence gates failed'
