from pathlib import Path
import subprocess,sys,os,json,yaml,numpy as np
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';runs=repo/'scratch/unified_em_perf_20260913';out=Path(__file__).resolve().parent
assert all(x['passed'] for x in json.loads((out/'segment_validation_results.json').read_text()))
cfg=yaml.safe_load((root/'segment_validation_inputs/RT07575_elastic/gpu.yaml').read_text());full=yaml.safe_load((root/'inputs/RT07575/gpu.yaml').read_text());cfg['number_of_histories']=1000000;cfg['tps_spots_file']=full['tps_spots_file'];folder=root/'segment_elastic_1m_inputs/RT07575';folder.mkdir(parents=True,exist_ok=True);(folder/'gpu.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
for label,build in [('segment_elastic_64_1m_r2','segmented_build'),('segment_elastic_base_1m_r2','mean_guard_build')]:
 p=runs/label/'RT07575/status.json'
 if p.exists():assert json.loads(p.read_text())['complete'];continue
 subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
 env=dict(os.environ,CARBON_RUNTIME_BREAKDOWN='1',CARBON_BENCH_PROFILE='0',CARBON_BENCH_CONFIG_ROOT=str(folder.parent),CARBON_BENCH_CONFIG_OVERRIDES='{}')
 subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(root/build/'carbon_mc'),'RT07575'],cwd=repo,env=env,check=True)
a=runs/'segment_elastic_base_1m_r2/RT07575';b=runs/'segment_elastic_64_1m_r2/RT07575';sa=json.loads((a/'status.json').read_text());sb=json.loads((b/'status.json').read_text())
for k in ['config_sha256','audit','steps']:assert sa[k]==sb[k],k
x=np.fromfile(a/'dose_gpu.raw','<f4').astype(float);y=np.fromfile(b/'dose_gpu.raw','<f4').astype(float);diff=float(100*abs(y-x).max()/x.max());assert diff<.001,diff
la=json.loads((a/'out/gpu/energy_ledger.json').read_text());lb=json.loads((b/'out/gpu/energy_ledger.json').read_text());counts={k:[v,lb[k]] for k,v in la.items() if isinstance(v,int)};assert counts and all(v[0]==v[1] for v in counts.values())
result=dict(baseline=sa,candidate=sb,ledger_integer_counts=counts,max_dose_difference_percent_peak=diff,gain_percent=100*(sb['throughput']/sa['throughput']-1));(out/'segment_elastic_1m_r2_results.json').write_text(json.dumps(result,indent=2)+'\n');print('ELASTIC 1M gain%',result['gain_percent'],'dose%peak',diff)
