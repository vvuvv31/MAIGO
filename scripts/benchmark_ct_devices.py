#!/usr/bin/env python3
"""Reproducible FP32 comparison of the four CT presets; runtime audits are mandatory."""
from pathlib import Path
import argparse,csv,json,os,re,subprocess,time
import numpy as np
import yaml

BATCHES={'proton_broadbeam':(10,1253150),'carbon_broadbeam':(7,925987),'proton_minibeam':(4,500000),'carbon_minibeam':(10,200000)}
CASES={'proton_broadbeam':'proton_ct_broadbeam','carbon_broadbeam':'carbon_ct_broadbeam','proton_minibeam':'proton_ct_minibeam','carbon_minibeam':'carbon_ct_minibeam'}
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def run(root,binary,lib,out,name,repeat,smoke=False):
 d=out/(name+('_smoke' if smoke else f'_r{repeat}'));d.mkdir(parents=True,exist_ok=False)
 for folder in ['data','inputs']:(d/folder).symlink_to(root/folder,target_is_directory=True)
 cfg=yaml.safe_load((root/'config'/f'{CASES[name]}.yaml').read_text())
 cfg['random_seed']=1001600000+repeat
 if name.endswith('_minibeam'):cfg['number_of_histories']=20000 if smoke else 2000000
 elif smoke:
  with (root/cfg['tps_spots_file']).open() as f:
   rr=csv.DictReader(f);fields=rr.fieldnames;row=next(r for r in rr if float(r['weight'])>0)
  row['weight']='50000'
  with (d/'smoke_spots.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
  cfg['tps_spots_file']='smoke_spots.csv';cfg['number_of_histories']=50000
 batches,histories=(2,int(cfg['number_of_histories'])//2) if smoke else BATCHES[name]
 expected=batches*histories
 (d/'case.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
 env.update(LD_LIBRARY_PATH=str(lib)+':'+env.get('LD_LIBRARY_PATH',''),CARBON_RUNTIME_BREAKDOWN='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 cmd=[str(binary),'--config','case.yaml','--batch-count',str(batches),'--histories-per-batch',str(histories),'--batch-output','batches']
 start=time.perf_counter()
 with (d/'telemetry.csv').open('w') as tel,(d/'gpu.log').open('w') as log:
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv','-l','1'],stdout=tel,stderr=subprocess.DEVNULL)
  try:code=subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  finally:monitor.terminate();monitor.wait()
 elapsed=time.perf_counter()-start
 text=(d/'gpu.log').read_text();hist=re.findall(r'^Histories: (\d+)$',text,re.M)
 outputs=[d/'batches'/f'batch{i:06d}' for i in range(batches)]
 quality=[json.loads((p/'quality_report.json').read_text()) for p in outputs if (p/'quality_report.json').exists()]
 stages={}
 for key,value in re.findall(r'\[runtime-stage\] (\S+) seconds=([\d.eE+-]+)',text):stages[key]=stages.get(key,0.)+float(value)
 failures=[a['code'] for q in quality for a in q.get('failures',[])]
 diagnostic=(name=='carbon_minibeam') # Dose-accuracy gate is deferred, runtime audits are not.
 actual=sum(map(int,hist)) if hist else None
 audits=re.findall(r'^\[unified-em-audit\] (.*)$',text,re.M)
 completion=re.findall(r'\[independent-fp32-completion\] expected=(\d+) completed=(\d+)',text)
 work=[line for line in text.splitlines() if line.startswith(('[independent-charged-work]','[independent-photon-work]','[electron-joint-pilot-work]'))]
 runtime_ok=(actual==expected and len(hist)==batches and all(int(v)==histories for v in hist) and len(quality)==batches and
             all(q.get('accepted') and q.get('queue_overflow_count')==0 and q.get('physical_relative_energy_residual',float('inf'))<1e-4 for q in quality) and
             len(audits)==batches and all(int(a.split()[0])==0 for a in audits) and
             len(completion)==batches and all(a==b for a,b in completion) and len(work)==3*batches and
             all(int(v)==0 for line in work for v in re.findall(r'(?:coverage_missing|invalid|photon_gap)=(\d+)',line)) and code==0 and not failures)
 batch_times=[float(v) for v in re.findall(r'\[batch-wall\] index=\d+ seconds=([\d.eE+-]+)',text)]
 runtime_ok=runtime_ok and len(batch_times)==batches
 loop=stages.get('transport_loop_including_scoring_and_queue_transfers')
 r=dict(case=name,repeat=repeat,smoke=smoke,seed=cfg['random_seed'],requested_histories=expected,actual_histories=actual,binary=str(binary),config=str(d/'case.yaml'),exit_code=code,wall_seconds=elapsed,transport_seconds=loop,runtime_checks_pass=runtime_ok,quality_accepted=len(quality)==batches and all(q.get('accepted') for q in quality),diagnostic_only=diagnostic,batch_quality=quality,batch_count=batches,histories_per_batch=histories,batch_wall_seconds=batch_times,resident_histories_per_second=(batches-1)*histories/sum(batch_times[1:]) if runtime_ok else None,stages=stages,em_audits=audits,end_to_end_histories_per_second=expected/elapsed if runtime_ok else None,transport_histories_per_second=expected/loop if runtime_ok and loop else None)
 if runtime_ok:
  dose_sum=0.;batch_max=[]
  for p in outputs:
   dose=np.memmap(p/'dose.raw',np.float32,'r');assert np.isfinite(dose).all() and (dose>=0).all()
   dose_sum+=float(dose.sum(dtype=float));batch_max.append(float(dose.max()))
  r['dose_sum']=dose_sum;r['batch_dose_max']=batch_max
 save(d/'execution.json',r)
 print(json.dumps({k:r[k] for k in ['case','repeat','smoke','actual_histories','wall_seconds','transport_seconds','runtime_checks_pass','quality_accepted','diagnostic_only','end_to_end_histories_per_second']}),flush=True)
 if not runtime_ok:raise RuntimeError('Unexpected run failure: '+str(d))
 return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES));ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--smoke-only',action='store_true')
 a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 hardware=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True,check=True).stdout;(out/'hardware.txt').write_text(hardware)
 results=[]
 for name in a.cases:results.append(run(root,a.binary.resolve(),a.lib.resolve(),out,name,0,True))
 save(out/'summary.json',dict(results=results))
 for repeat in range(1,(0 if a.smoke_only else a.repeats)+1):
  for name in (a.cases if repeat%2 else a.cases[::-1]):
   results.append(run(root,a.binary.resolve(),a.lib.resolve(),out,name,repeat))
   save(out/'summary.json',dict(results=results))
 print('BENCHMARK COMPLETE',flush=True)
if __name__=='__main__':main()
