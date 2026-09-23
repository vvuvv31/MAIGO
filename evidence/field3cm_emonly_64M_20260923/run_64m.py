from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,subprocess,sys,time,traceback
import numpy as np
R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_64M_20260923'
M=json.loads((O/'manifest.json').read_text());P=json.loads((O/'preflight.json').read_text())
lock=(O/'serial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def dump(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
cg=next(v.split('::',1)[1] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::'))
C=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert 29999990000<=int((C/'memory.max').read_text())<=30000000000
assert (C/'memory.swap.max').read_text().strip()=='0' and (C/'memory.oom.group').read_text().strip()=='1'
assert not os.environ.get('SLURM_JOB_ID')
assert sha(O/'bin/topas')==P['topas_binary_sha256'] and sha(O/'bin/carbon_mc')==M['gpu_binary_sha256']
for v in M['files']+M['runtime_dependencies']+M['reused_dose_files']:assert sha(v['path'])==v['sha256'],v['path']
stages=[(e,i) for i in M['new_batches'] for e in ['topas','gpu']]
state=dict(status='RUNNING',started_at=now(),mode='direct_serial',systemd_unit=os.environ.get('MAIGO_SYSTEMD_UNIT'),
    pid=os.getpid(),cgroup=str(C),memory_max_bytes=int((C/'memory.max').read_text()),memory_swap_max_bytes=0,
    target_histories_per_engine=M['histories_per_engine'],reused_histories_per_engine=12800000,
    planned_stages=[f'{e}_b{i}' for e,i in stages]+['analysis'],completed_stages=[],current_stage=None,
    completed_histories={'topas':12800000,'gpu':12800000},peak_memory_bytes=0)
memory=(O/'memory_usage.tsv').open('a',buffering=1)
if memory.tell()==0:memory.write('time\tstage\tmemory_current_bytes\tmemory_peak_bytes\n')
def sample():
    state.update(updated_at=now(),current_memory_bytes=int((C/'memory.current').read_text()),
        peak_memory_bytes=max(state['peak_memory_bytes'],int((C/'memory.peak').read_text())))
    remaining=0
    for e in ['topas','gpu']:
        done=[s['wall_s'] for s in state['completed_stages'] if s.get('engine')==e and not s.get('reused_after_restart')]
        average=sum(done)/len(done) if done else M['baseline_batch_wall_s'][e]
        count=sum(1 for ee,ii in stages if ee==e and not any(s['name']==f'{ee}_b{ii}' for s in state['completed_stages']))
        remaining+=count*average
    if state.get('current_stage_started_monotonic') is not None:
        remaining=max(0,remaining-(time.monotonic()-state['current_stage_started_monotonic']))
    state['estimated_remaining_s']=remaining
    state['estimated_finish_at']=(datetime.datetime.now().astimezone()+datetime.timedelta(seconds=remaining)).isoformat()
    memory.write(f"{state['updated_at']}\t{state['current_stage']}\t{state['current_memory_bytes']}\t{state['peak_memory_bytes']}\n")
    dump(O/'serial_status.json',state)
def execute(cmd,cwd,log):
    with log.open('x') as f:p=subprocess.Popen(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    state['child_pid']=p.pid;sample()
    while True:
        try:rc=p.wait(timeout=10);break
        except subprocess.TimeoutExpired:sample()
    state.pop('child_pid',None)
    if rc:raise RuntimeError(f'{cmd[0]} exited {rc}: '+log.read_text()[-3000:])
def check_existing(engine,i,qpath):
    q=json.loads(qpath.read_text());assert q['accepted'] and q['histories']==M['histories_per_batch']
    assert sha(q['dose_path'])==q['dose_sha256']
    expected=P['topas_binary_sha256'] if engine=='topas' else M['gpu_binary_sha256']
    assert q[engine+'_binary_sha256']==expected
    return q
try:
    for engine,i in stages:
        name=f'{engine}_b{i}';case=Path(M['cases'][name]);d=case if engine=='topas' else case.parent
        for v in M['files']:
            if Path(v['path']).parent==d:assert sha(v['path'])==v['sha256'],v['path']
        state.update(current_stage=name,current_stage_started_monotonic=time.monotonic(),current_stage_started_at=now());sample()
        qp=d/'quality.json'
        if qp.exists():
            q=check_existing(engine,i,qp)
            state['completed_stages'].append(dict(name=name,engine=engine,wall_s=q['wall_s'],finished_at=now(),reused_after_restart=True))
            state['completed_histories'][engine]+=q['histories'];continue
        print('STAGE_START',name,now(),flush=True);start=time.monotonic();gpu_guard=None
        if engine=='topas':
            assert sha(O/'bin/topas')==P['topas_binary_sha256']
            execute([str(O/'bin/topas'),'run.txt'],d,d/'run.log')
            text=(d/'run.log').read_text();n=M['histories_per_batch']
            assert 'History-to-spot map is event-ID based (MT-safe).' in text
            assert 'geant4-11-03-patch-02' in text
            assert f'Particle source CarbonPBS: Total number of histories: {n}\n' in text
            dose=d/'dose.bin';dt='<f8'
            q=dict(name=name,threads=128,topas_binary_sha256=sha(O/'bin/topas'))
        else:
            assert sha(O/'bin/carbon_mc')==M['gpu_binary_sha256']
            for v in M['runtime_dependencies']:assert sha(v['path'])==v['sha256'],v['path']
            gpu_guard=open('/tmp/maigo-review-gpu.lock','a')
            while True:
                try:fcntl.flock(gpu_guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:state['waiting_for_gpu']='lock';sample();time.sleep(10)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                state['waiting_for_gpu']='occupied';sample();time.sleep(10)
            state.pop('waiting_for_gpu',None)
            dest=R/'out'/case.stem;assert not dest.exists(),str(dest)
            execute([str(O/'bin/carbon_mc'),'--config',str(case)],R,d/'run.log')
            text=(d/'run.log').read_text();assert f'spots: 256/256 active; total histories: {M["histories_per_batch"]}' in text
            qq=json.loads((dest/'quality_report.json').read_text())
            urban=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',text).group(1))
            assert qq['accepted'] and not qq['failures'] and not any(urban[k] for k in ['fatal','cap','guard','subulp'])
            dose=dest/'dose.raw';dt='<f4'
            q=dict(name=case.stem,gpu_binary_sha256=M['gpu_binary_sha256'],urban=urban,relative_energy_residual=qq['relative_energy_residual'])
        a=np.fromfile(dose,dtype=dt);assert len(a)==1000000 and np.isfinite(a).all() and a.min()>=0 and a.max()>0
        q.update(histories=M['histories_per_batch'],accepted=True,wall_s=time.monotonic()-start,
            dose_path=str(dose),dose_sha256=sha(dose),slurm_job_id=None,finished_at=now())
        dump(qp,q)
        if gpu_guard:gpu_guard.close()
        state['completed_stages'].append(dict(name=name,engine=engine,wall_s=q['wall_s'],finished_at=now()))
        state['completed_histories'][engine]+=q['histories']
        state['current_stage_started_monotonic']=None;sample();print('STAGE_COMPLETE',name,json.dumps(q),flush=True)
    assert state['completed_histories']=={'topas':64000000,'gpu':64000000}
    for v in M['files']+M['runtime_dependencies']+M['reused_dose_files']:assert sha(v['path'])==v['sha256'],v['path']
    state.update(status='ANALYZING',simulation_complete=True,current_stage='analysis',current_stage_started_monotonic=None);sample()
    log=O/'logs'/('analysis_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
    execute([sys.executable,str(O/'analyze_64m.py')],R,log)
    state.update(status='COMPLETE',current_stage=None,finished_at=now(),estimated_remaining_s=0);sample()
    print('ALL_COMPLETE',json.dumps(state),flush=True)
except BaseException as e:
    state.update(status='FAILED',error=str(e),finished_at=now());sample();traceback.print_exc();raise
