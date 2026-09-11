#!/usr/bin/env python3
"""Run ONLY primary-midpoint + section-ion stopping + exact-faces on local GPU.
Freeze inputs and executable, check 50k closure then original full shards.
No ablation variants. Overflow recursively splits histories; no failed-dose merge.
"""
import argparse,csv,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
import yaml
from compile_schneider_ion_stopping import sha
from run_topas10x_gpu_benchmark import config_write
REPO=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,ONEAPI_DEVICE_SELECTOR='cuda:*',LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib')
def save(p,j):p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',required=True,type=Path);p.add_argument('--table',type=Path,default=REPO/'data/schneider/schneider_ion_section_stopping_v1.bin')
 p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--full20',action='store_true');a=p.parse_args()
 if a.out.exists():raise ValueError('New directory required')
 a.out.mkdir(parents=True);m=json.loads((a.reference/'manifest.json').read_text());ex=json.loads((a.reference/'execution.json').read_text())
 assert ex['status']=='complete' and sha(a.reference/'gpu_sum.raw')==ex['aggregate_sha256']
 assert sha(a.reference/'topas_sum.raw')==m['reference_sum_sha256']
 meta=a.table.with_suffix('.metadata.json');binary_pin=sha(a.binary);table_pin=sha(a.table);meta_pin=sha(meta)
 state={'status':'preflight','binary':str(a.binary),'binary_sha256':binary_pin,'table':str(a.table),'table_sha256':table_pin,'metadata_sha256':meta_pin,'case':m['case'],'completed':[],'overflow_attempts':[]}
 # Save the actual working-tree source, not just HEAD.
 subprocess.run(['git','diff','--binary'],cwd=REPO,stdout=(a.out/'worktree.diff').open('w'),check=True)
 state['source_pins']={str(f.relative_to(REPO)):sha(f) for d in ['include/carbon','src'] for f in (REPO/d).rglob('*') if f.is_file()}
 state['runner_sha256']=sha(__file__);state['reference_sha256']=sha(a.reference/'topas_sum.raw')
 save(a.out/'execution.json',state)
 subprocess.run(['python3','tools/verify_schneider_v2_1_data.py'],cwd=REPO,check=True)
 subprocess.run(['python3','tools/verify_schneider_ion_stopping.py',str(a.table)],cwd=REPO,check=True)
 total=np.zeros(m['gpu_shape_zyx'],dtype=np.float64)
 def config(source):
  c=yaml.safe_load(source.read_text())
  for key in ['ct_primary_midpoint_stopping_diagnostic','ct_secondary_exact_faces_diagnostic','ct_secondary_schneider_sp_diagnostic']:c.pop(key,None)
  c.update(ct_primary_midpoint_stopping=True,ct_secondary_exact_faces=True,
   ct_secondary_ion_section_stopping_file=str(a.table),ct_secondary_ion_section_stopping_sha256=table_pin,
   ct_secondary_ion_section_stopping_metadata_sha256=meta_pin,run_mode='research',LET=False)
  return c
 def run(d,c,merge,depth=0):
  d.mkdir();config_write(d/'config.yaml',c);(d/'data').symlink_to(REPO/'data',target_is_directory=True)
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Local GPU busy')
  if sha(a.binary)!=binary_pin or sha(a.table)!=table_pin or sha(meta)!=meta_pin:raise RuntimeError('Input changed')
  pins={k:sha(v) for k,v in c.items() if isinstance(v,str) and Path(v).is_file()};save(d/'input_pins.json',pins)
  start=time.monotonic()
  with (d/'gpu.log').open('w') as log:
   r=subprocess.run([str(a.binary),'--config',str(d/'config.yaml'),'--device','cuda','--voxel-dose-mhd',str(d/'dose.mhd')],cwd=d,env=ENV,stdout=log,stderr=subprocess.STDOUT)
  save(d/'timing.json',{'returncode':r.returncode,'wall_seconds':time.monotonic()-start})
  qp=d/'out/config/quality_report.json'
  if not qp.exists():
   state.update(status='failed',failed_directory=str(d),returncode=r.returncode,
    error=(d/'gpu.log').read_text()[-2000:])
   save(a.out/'execution.json',state)
   raise RuntimeError('No quality output: '+str(d)+'; see gpu.log')
  q=json.loads(qp.read_text())
  if q['queue_overflow_count'] or q.get('queue_overflow_energy_MeV',0):
   state['overflow_attempts'].append(str(d));save(a.out/'execution.json',state)
   if depth>=5:raise RuntimeError('Overflow split limit')
   with Path(c['tps_spots_file']).open() as f:rows=list(csv.DictReader(f))
   for half in range(2):
    selected=[dict(row,weight=int(row['weight'])//2+(int(row['weight'])%2 if half else 0)) for row in rows];selected=[x for x in selected if x['weight']]
    if not selected:continue
    spots=d/f'split_{half}.csv'
    with spots.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(selected)
    cc=dict(c,tps_spots_file=str(spots),number_of_histories=sum(x['weight'] for x in selected),random_seed=int(c['random_seed'])+100003*(half+1)+depth)
    run(d/f'split_{half}',cc,merge,depth+1)
   return
  if r.returncode or not q['accepted'] or q['failures']:raise RuntimeError('Run rejected: '+str(d)+' '+str(q['failures']))
  ledger=json.loads((d/'out/config/energy_ledger.json').read_text());assert ledger['histories']==c['number_of_histories']
  assert ledger['ct_primary_midpoint_stopping'] is True and ledger['ct_secondary_exact_faces'] is True
  assert ledger['ct_secondary_ion_section_stopping_sha256']==table_pin
  dose=np.fromfile(d/'dose.raw','<f4').reshape(total.shape)
  if not np.all(np.isfinite(dose)&(dose>=0)):raise RuntimeError('Invalid dose')
  record={'directory':str(d),'histories':c['number_of_histories'],'dose_sha256':sha(d/'dose.raw'),'config_sha256':sha(d/'config.yaml'),'quality_sha256':sha(qp),'wall_seconds':time.monotonic()-start}
  if merge:total[:]+=dose;state['completed'].append(record)
  else:state['closure50k']=record
  save(a.out/'execution.json',state);print(d.name,'PASS',record['histories'],round(record['wall_seconds'],2),flush=True)
 first=config(Path(m['shards'][0]['directory'])/'config.yaml')
 # Fixed total allocator preserves spot weights for the closure run.
 smoke=dict(first,number_of_histories=50000,tps_spot_weight_mode='mu',tps_histories_scale=1.0,history_chunk_size=50000)
 run(a.out/'closure50k',smoke,False)
 if not a.full20:state['status']='closure_complete';save(a.out/'execution.json',state);return
 state['status']='running';save(a.out/'execution.json',state)
 for i,t in enumerate(m['shards'],1):run(a.out/f'shard_{i:02d}',config(Path(t['directory'])/'config.yaml'),True)
 assert sum(x['histories'] for x in state['completed'])==m['histories']
 total.astype('<f4').tofile(a.out/'gpu_sum.raw');state['aggregate_sha256']=sha(a.out/'gpu_sum.raw');state['status']='complete';save(a.out/'execution.json',state)
 # Same-position metric needs no search; evaluate exact stored float32 dose.
 g=np.fromfile(a.out/'gpu_sum.raw','<f4').reshape(m['gpu_shape_zyx'])
 if m['mapping']=='packed_xneg':g=np.flip(g.transpose(1,2,0),axis=2)
 elif m['mapping']!='native':raise ValueError('Unknown mapping')
 ref=np.fromfile(a.reference/'topas_sum.raw','<f4').reshape(m['topas_shape_zyx']);mask=ref>=.1*ref.max()
 rr=ref[mask].astype(float);gg=g[mask].astype(float);delta=gg-rr
 summary={'case':m['case'],'histories':m['histories'],'global_3pct_0mm':100*float((abs(delta)<=.03*ref.max()).mean()),'local_3pct_0mm':100*float((abs(delta)<=.03*rr).mean()),'mean_local_error_pct':float((100*delta/rr).mean()),'rms_local_error_pct':float(np.sqrt(np.mean((100*delta/rr)**2))),'source_sha256':sha(__file__),'dose_sha256':state['aggregate_sha256']}
 save(a.out/'same_voxel_comparison.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
