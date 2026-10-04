from pathlib import Path
import os,sys,json,re,time,subprocess,fcntl
from common import ROOT,REPO,sha,save_json
from aggregate import aggregate
sys.path.insert(0,str(REPO/'scripts'))
from compare_minibeam import AUDITS
lock=(ROOT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.loads((ROOT/'manifest.json').read_text());identity=json.loads((ROOT/'build.json').read_text());binary=ROOT/'inputs/carbon_mc'
assert sha(binary)==identity['binary_sha256']
for job in m['jobs']:
 assert sha(ROOT/(job['name']+'.yaml'))==job['config_sha256']
 assert sha(ROOT/'sources'/(job['name']+'.txt'))==job['source_sha256']
env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','');env['CARBON_RUNTIME_BREAKDOWN']='1'
assert not any(k.startswith('CARBON_') and k!='CARBON_RUNTIME_BREAKDOWN' for k in env)
old=Path(m['previous_campaign']);previous={}
for p in sorted(old.glob('group[0-9][0-9][0-9].log')):
 for part in p.read_text().split('[plan-shard] ')[1:]:
  name=re.search(r'(batch\d+)\.yaml',part)[1]
  if '[plan-shard-done]' in part:previous[name]=[l for l in part.splitlines() if l.startswith(tuple(AUDITS))]
assert len(previous)==119
start=time.perf_counter();started=time.time();completed=[];checks=[];chunk=[]
proc=None
try:
 with (ROOT/'gpu.log').open('w') as logfile:
  proc=subprocess.Popen([str(binary),'--plan-manifest',str(ROOT/'plan.manifest')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  save_json(ROOT/'campaign_status.json',dict(state='running',pid=proc.pid,runner_pid=os.getpid(),started_unix=started,completed_parts=0,requested_histories=m['histories']))
  for line in proc.stdout:
   logfile.write(line);logfile.flush()
   if line.startswith('[plan-shard] '):chunk=[]
   chunk.append(line.rstrip('\n'))
   if not line.startswith('[batch-wall]'):continue
   job=m['jobs'][len(completed)];text='\n'.join(chunk);name=job['name'];q=json.loads((ROOT/f'out/{name}/quality_report.json').read_text())
   assert q['accepted'] and not q['failures'] and q['queue_overflow_count']==0,(name,q)
   assert f"Histories: {job['histories']}" in text
   assert 'legacy_loaded=0 legacy_fallback=0' in text
   assert '[electron-elastic-batch]' not in text
   cc=re.findall(r'\[independent-fp32-completion\] expected=(\d+) completed=(\d+)',text);assert len(cc)==1 and cc[0][0]==cc[0][1]
   work={}
   for prefix in ['[independent-charged-work]','[independent-photon-work]','[electron-joint-pilot-work]']:
    lines=[l for l in chunk if l.startswith(prefix)];assert len(lines)==1
    w=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',lines[0]));assert not w.get('invalid',0) and not w.get('coverage_missing',0) and not w.get('photon_gap',0);work[prefix]=w
   primary_equal=[l for l in chunk if l.startswith(tuple(AUDITS))]==previous[name];assert primary_equal,(name,'primary/EM audit mismatch')
   checks.append(dict(name=name,histories=job['histories'],energy_residual=q['relative_energy_residual'],queue_overflow=0,quality_accepted=True,primary_EM_equal=primary_equal,completed_electron_tasks=int(cc[0][0]),work=work,batch_seconds=float(re.search(r'seconds=([\d.eE+-]+)',line)[1])))
   completed.append(job);wall=time.perf_counter()-start;histories=sum(j['histories'] for j in completed)
   save_json(ROOT/'batch_checks.json',checks)
   status=dict(state='running',pid=proc.pid,runner_pid=os.getpid(),started_unix=started,completed_parts=len(completed),completed_histories=histories,requested_histories=m['histories'],wall_seconds=wall,histories_per_second=histories/wall,remaining_seconds=(m['histories']-histories)/(histories/wall))
   save_json(ROOT/'campaign_status.json',status);print(json.dumps(status),flush=True)
  rc=proc.wait();wall=time.perf_counter()-start
 assert rc==0 and len(completed)==119,(rc,len(completed))
 save_json(ROOT/'execution.json',dict(exit_code=rc,wall_seconds=wall,binary_sha256=identity['binary_sha256'],histories=m['histories'],parts=len(completed)))
 summary=aggregate(completed,m,'results',wall)
 log=(ROOT/'gpu.log').read_text();batch=[float(v) for v in re.findall(r'\[batch-wall\].*seconds=([\d.eE+-]+)',log)];electron=[float(v) for v in re.findall(r'\[electron-queue-time\] seconds=([\d.eE+-]+)',log)]
 assert len(batch)==len(electron)==119
 timing=dict(process_seconds=wall,process_primary_per_second=m['histories']/wall,resident_primary_per_second=sum(j['histories'] for j in m['jobs'][1:])/sum(batch[1:]),electron_seconds=sum(electron),electron_mean_batch_seconds=sum(electron)/119,batch_seconds=batch,electron_batch_seconds=electron)
 save_json(ROOT/'results/timing.json',timing)
 save_json(ROOT/'campaign_status.json',dict(state='complete',completed_parts=119,completed_histories=m['histories'],requested_histories=m['histories'],wall_seconds=wall,histories_per_second=m['histories']/wall))
 print(json.dumps(dict(state='complete',timing=timing,GTV=summary['gpu_metrics']['GTV'])),flush=True)
except BaseException as e:
 if proc and proc.poll() is None:proc.terminate();proc.wait()
 save_json(ROOT/'campaign_status.json',dict(state='failed',error=str(e),completed_parts=len(completed),completed_histories=sum(j['histories'] for j in completed)))
 raise
