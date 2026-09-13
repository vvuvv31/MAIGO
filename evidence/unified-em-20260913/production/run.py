from pathlib import Path
import json,subprocess,time,os,hashlib,re
r=Path(__file__).resolve().parents[3];root=r/'scratch/unified_em_production_20260913';exe=r/'scratch/unified_joint_em_20260913/build/carbon_mc';env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib',ONEAPI_DEVICE_SELECTOR='cuda:*')
for name in ['water','ct_layers']:
 d=root/name;start=time.time()
 with (d/'gpu.log').open('w') as f:p=subprocess.run([str(exe),'--config',str(d/'gpu.yaml'),'--device','cuda','--voxel-dose-mhd',str(d/'dose.mhd')],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
 qp=d/'out/gpu/quality_report.json';q=json.loads(qp.read_text()) if qp.exists() else {};audit=re.findall(r'\[unified-em-audit\]([^\n]+)',(d/'gpu.log').read_text())
 ok=p.returncode==0 and q.get('accepted',False) and q.get('status')=='pass' and q.get('run_mode')=='production' and q.get('queue_overflow_count',-1)==0 and bool(audit) and all(int(x.split()[0])==0 for x in audit)
 ok=ok and any(x['code']=='unified_material_em_accuracy_pending' for x in q.get('approximations',[]))
 status=dict(complete=ok,returncode=p.returncode,elapsed_s=time.time()-start,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),config_sha256=hashlib.sha256((d/'gpu.yaml').read_bytes()).hexdigest(),quality=q,audit=audit)
 (d/'status.json').write_text(json.dumps(status,indent=2));print(name,ok,status['elapsed_s'],flush=True)
 if not ok:raise RuntimeError((d/'gpu.log').read_text()[-2500:])
