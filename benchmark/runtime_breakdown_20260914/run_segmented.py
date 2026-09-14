from pathlib import Path
import os,sys,subprocess,json,csv,yaml,numpy as np
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';runs=repo/'scratch/unified_em_perf_20260913';out=Path(__file__).resolve().parent
cfg=yaml.safe_load((root/'inputs/RT07575/gpu.yaml').read_text());folder=root/'segmented_inputs_50k/RT07575';folder.mkdir(parents=True,exist_ok=True)
with open(cfg['tps_spots_file']) as f:reader=csv.DictReader(f);names=reader.fieldnames;spots=list(reader)
w=np.array([float(x['weight']) for x in spots]);v=w/w.sum()*50000;n=np.floor(v).astype(int);order=np.argsort(-(v-n),kind='stable');n[order[:50000-int(n.sum())]]+=1;assert n.sum()==50000
with (folder/'spots.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
 for row,count in zip(spots,n):
  if count:row['weight']=str(count);writer.writerow(row)
cfg['number_of_histories']=50000;cfg['tps_spots_file']=str(folder/'spots.csv');(folder/'gpu.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
def run(label,build,inputs):
 p=runs/label/'RT07575/status.json'
 if p.exists():assert json.loads(p.read_text())['complete'];return
 subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
 env=dict(os.environ,CARBON_RUNTIME_BREAKDOWN='1',CARBON_BENCH_PROFILE='0',CARBON_BENCH_CONFIG_ROOT=str(inputs),CARBON_BENCH_CONFIG_OVERRIDES='{}')
 subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(root/build/'carbon_mc'),'RT07575'],cwd=repo,env=env,check=True)
def compare(base,candidate):
 a=runs/base/'RT07575';b=runs/candidate/'RT07575';sa=json.loads((a/'status.json').read_text());sb=json.loads((b/'status.json').read_text())
 assert sa['complete'] and sb['complete']
 for key in ['config_sha256','audit','steps']:assert sa[key]==sb[key],(key,sa[key],sb[key])
 x=np.fromfile(a/'dose_gpu.raw','<f4').astype(float);y=np.fromfile(b/'dose_gpu.raw','<f4').astype(float);assert x.shape==y.shape and np.isfinite(y).all();diff=100*abs(x-y).max()/x.max();assert diff<.001,diff
 la=json.loads((a/'out/gpu/energy_ledger.json').read_text());lb=json.loads((b/'out/gpu/energy_ledger.json').read_text());counts={k:[v,lb[k]] for k,v in la.items() if isinstance(v,int)};assert counts and all(v[0]==v[1] for v in counts.values()),counts
 result=dict(base_label=base,candidate_label=candidate,baseline=sa,candidate=sb,max_dose_difference_percent_peak=diff,ledger_integer_counts=counts,gain_percent=100*(sb['throughput']/sa['throughput']-1));print('PAIR',base,'gain%',result['gain_percent'],'dose%peak',diff,flush=True);return result
rows=[]
for label,build in [('runtime_segment_base_50k','mean_guard_build'),('runtime_segment_64_50k','segmented_build')]:run(label,build,folder.parent)
rows.append(compare('runtime_segment_base_50k','runtime_segment_64_50k'));(out/'segmented_results.json').write_text(json.dumps(rows,indent=2)+'\n')
for i in [1,2]:
 pair=[(f'runtime_segment_base_1m_r{i}','mean_guard_build'),(f'runtime_segment_64_1m_r{i}','segmented_build')]
 for label,build in (pair if i==1 else pair[::-1]):run(label,build,root/'inputs')
 rows.append(compare(pair[0][0],pair[1][0]));(out/'segmented_results.json').write_text(json.dumps(rows,indent=2)+'\n')
