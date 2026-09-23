from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time,traceback

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923'
lock=(O/'serial.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cg=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
C=Path('/sys/fs/cgroup')/cg.lstrip('/')
cap=int((C/'memory.max').read_text())
assert 29999990000<=cap<=30000000000,cap
assert (C/'memory.swap.max').read_text().strip()=='0'
assert (C/'memory.oom.group').read_text().strip()=='1'

def now():return datetime.datetime.now().astimezone().isoformat()
def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

stages=[('preflight',[sys.executable,str(O/'run_field.py'),'preflight'])]
for i in range(1,5):
    stages.extend([(f'topas_b{i}',[sys.executable,str(O/'run_field.py'),'topas',str(i)]),
                   (f'gpu_b{i}',[sys.executable,str(O/'run_field.py'),'gpu_batch',str(i)])])
    if i==1:stages.append(('preview_b1',[sys.executable,str(O/'preview_field.py')]))
stages.append(('final_comparison',[sys.executable,str(O/'analyze_field.py')]))
state=dict(status='RUNNING',started_at=now(),mode='direct_serial',pid=os.getpid(),
           systemd_unit=os.environ.get('MAIGO_SYSTEMD_UNIT'),cgroup=str(C),
           memory_max_bytes=cap,memory_high_bytes=int((C/'memory.high').read_text()),
           memory_swap_max_bytes=0,planned_stages=[s[0] for s in stages],completed_stages=[],
           current_stage=None,peak_memory_bytes=0)

def sample(log):
    cur=int((C/'memory.current').read_text());peak=int((C/'memory.peak').read_text())
    state.update(current_memory_bytes=cur,peak_memory_bytes=max(state['peak_memory_bytes'],peak),updated_at=now())
    log.write(f"{state['updated_at']}\t{state['current_stage']}\t{cur}\t{peak}\n");log.flush()
    dump(O/'serial_status.json',state)

with (O/'memory_usage.tsv').open('a',buffering=1) as memory:
    if memory.tell()==0:memory.write('time\tstage\tmemory_current_bytes\tmemory_peak_bytes\n')
    try:
        for name,cmd in stages:
            state['current_stage']=name;sample(memory)
            print('STAGE_START',name,now(),flush=True);start=time.monotonic()
            p=subprocess.Popen(cmd,cwd=R)
            while True:
                try:rc=p.wait(timeout=2)
                except subprocess.TimeoutExpired:sample(memory);continue
                break
            sample(memory)
            if rc:raise RuntimeError(f'{name} exited with code {rc}')
            state['completed_stages'].append(dict(name=name,wall_s=time.monotonic()-start,finished_at=now()))
            print('STAGE_COMPLETE',name,flush=True)
        state.update(status='COMPLETE',current_stage=None,finished_at=now());sample(memory)
        print('ALL_COMPLETE',json.dumps(state),flush=True)
    except BaseException as e:
        state.update(status='FAILED',error=str(e),finished_at=now());sample(memory)
        traceback.print_exc();raise
