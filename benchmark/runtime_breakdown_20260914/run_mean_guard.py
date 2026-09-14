from pathlib import Path
import subprocess,os,sys,json,yaml
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';result=repo/'scratch/unified_em_perf_20260913'
def run(label,build,inputs,overrides):
 p=result/label/'RT07575/status.json'
 if p.exists():assert json.loads(p.read_text())['complete'];return
 subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
 env=dict(os.environ,CARBON_RUNTIME_BREAKDOWN='1',CARBON_BENCH_PROFILE='0',CARBON_BENCH_CONFIG_ROOT=str(inputs),CARBON_BENCH_CONFIG_OVERRIDES=json.dumps(overrides))
 subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(root/build/'carbon_mc'),'RT07575'],cwd=repo,env=env,check=True)
for label,build in [('runtime_mean_guard_1m_r1','mean_guard_build'),('runtime_mean_base_1m_r2','build'),('runtime_mean_guard_1m_r2','mean_guard_build')]:run(label,build,root/'inputs',{})
cfg=yaml.safe_load((root/'inputs/RT07575/gpu.yaml').read_text());cfg['number_of_histories']=200000;cfg['tps_spots_file']=str(repo/'scratch/unified_em_ablation_20260913/ct_inputs/200000/RT07575/spots.csv');d=root/'audit_inputs/RT07575';d.mkdir(parents=True,exist_ok=True);(d/'gpu.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
for variant,build in [('base','build'),('guard','mean_guard_build')]:run('runtime_mean_audit_'+variant,build,root/'audit_inputs',{'run_mode':'smoke','enable_primary_loss_query_audit':True})
