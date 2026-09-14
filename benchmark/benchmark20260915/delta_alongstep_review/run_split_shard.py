from pathlib import Path
import csv,json,os,subprocess,sys,yaml,time
import numpy as np
repo=Path(__file__).resolve().parents[3];R=Path(__file__).resolve().parent;inputs=R/'shard_inputs';cfg=yaml.safe_load((repo/'config/rt07575_unified_em_production.yaml').read_text());rows=list(csv.DictReader(open(cfg['tps_spots_file'])));weights=[int(float(x['weight'])) for x in rows];assert sum(weights)==6481909
statuses=[];outdirs=[]
for j in range(2):
 name=f'RT07575_part{j+1}';d=inputs/name;d.mkdir(parents=True,exist_ok=True);counts=[w//2+(j==0 and w%2==1) for w in weights]
 with (d/'spots.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader()
  for row,w in zip(rows,counts):writer.writerow(dict(row,weight=w))
 c=dict(cfg,number_of_histories=sum(counts),tps_spots_file=str(d/'spots.csv'),random_seed=cfg['random_seed']+j*1000003);(d/'gpu.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
 label='delta_alongstep_current_shard01_split';dest=repo/'scratch/unified_em_perf_20260913'/label/name
 if not (dest/'status.json').exists():
  subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
  env=dict(os.environ,CARBON_BENCH_CONFIG_ROOT=str(inputs),CARBON_BENCH_CONFIG_OVERRIDES='{}',CARBON_BENCH_PROFILE='0',CARBON_RUNTIME_BREAKDOWN='1')
  subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(repo/'scratch/delta_alongstep_current_20260915/build/carbon_mc'),name],cwd=repo,env=env,check=True)
 s=json.loads((dest/'status.json').read_text());assert s['complete'];assert s['quality']['queue_overflow_count']==0;assert all(a[3]==a[4]==0 and a[6]>0 for a in s['audit']);l=json.loads((dest/'out/gpu/energy_ledger.json').read_text());assert l['histories']==sum(counts)
 statuses.append(dict(histories=sum(counts),status=s));outdirs.append(dest)
assert sum(v['histories'] for v in statuses)==6481909
wall=sum(v['status']['wall_s'] for v in statuses);elapsed=sum(v['status']['elapsed_s'] for v in statuses)
result=dict(histories=6481909,parts=statuses,total_wall_s=wall,total_elapsed_s=elapsed,throughput_hps=6481909/elapsed,wall_throughput_hps=6481909/wall,primary_s=sum(v['status']['primary_s'] for v in statuses),secondary_s=sum(v['status']['secondary_s'] for v in statuses),single_attempt='not attempted; use same two subdivisions as delta-OFF for comparable timing',split='per-spot floor/ceil halves; seed of second part offset by 1000003')
start=time.perf_counter();arrays=[np.fromfile(d/'dose_gpu.raw','<f4').astype('f8') for d in outdirs];assert arrays[0].shape==arrays[1].shape
merged=arrays[0]+arrays[1];assert np.isfinite(merged).all();merged.astype('<f4').tofile(R/'RT07575_delta_alongstep_merged.raw');mhd=(outdirs[0]/'dose_gpu.mhd').read_text();mhd=mhd.replace('dose_gpu.raw','RT07575_delta_alongstep_merged.raw');(R/'RT07575_delta_alongstep_merged.mhd').write_text(mhd);result['merge_seconds']=time.perf_counter()-start
(R/'shard_results.json').write_text(json.dumps(result,indent=2));print('SPLIT SHARD',result['throughput_hps'],wall,'primary',result['primary_s'],'secondary',result['secondary_s'],flush=True)
