"""Run fresh GPU doses sequentially on the local RTX 2080 Ti."""
from pathlib import Path
import json,subprocess,time,os,hashlib,re,fcntl
R=Path(__file__).resolve().parent;repo=R.parents[3];exe=R/'carbon_mc'
manifest=json.loads((R/'input_manifest.json').read_text());assert hashlib.sha256(exe.read_bytes()).hexdigest()==manifest['binary_sha256']
subprocess.run(['python3',str(repo/'tools/verify_schneider_v2_1_data.py')],check=True)
if any(v=='g4_material_joint_v1' for v in manifest['models'].values()):subprocess.run(['python3',str(repo/'tools/verify_unified_em_data.py')],check=True)
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='cuda:*',LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
for key in ['CARBON_JOINT_EM_DATA','CARBON_DIAGNOSTIC_PRIMARY_START_DEDX']:env.pop(key,None)
for name,*_ in json.loads((R/'cases.json').read_text()):
 d=R/name
 assert hashlib.sha256((d/'gpu.yaml').read_bytes()).hexdigest()==manifest['sha256'][name+'/gpu.yaml']
 if (d/'gpu_status.json').exists() and json.loads((d/'gpu_status.json').read_text()).get('complete'):continue
 binary=manifest.get('case_binaries',{}).get(name,dict(path='carbon_mc',sha256=manifest['binary_sha256']));exe=R/binary['path']
 assert hashlib.sha256(exe.read_bytes()).hexdigest()==binary['sha256']
 subprocess.run(['python3',str(repo/'tools/verify_schneider_v2_1_data.py')],check=True)
 lock=(repo/'scratch/unified_em_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 start=time.time()
 with (d/'gpu.log').open('w') as log:p=subprocess.run([str(exe),'--config',str(d/'gpu.yaml'),'--device','cuda','--voxel-dose-mhd',str(d/'dose_gpu.mhd')],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 qpath=d/'out/gpu/quality_report.json';q=json.loads(qpath.read_text()) if qpath.exists() else {}
 tag='unified-em-audit' if manifest['models'][name]=='g4_material_joint_v1' else 'joint-em-audit'
 audit=[list(map(int,a.split())) for a in re.findall(r'\['+tag+r'\]([^\n]+)',(d/'gpu.log').read_text())]
 ok=p.returncode==0 and q.get('accepted',False) and q.get('queue_overflow_count',-1)==0
 if manifest['models'][name]!='legacy':ok=ok and bool(audit) and all(a[0]==0 for a in audit)
 status=dict(complete=ok,returncode=p.returncode,elapsed_s=time.time()-start,model=manifest['models'][name],primary_em_model='legacy' if manifest['models'][name]=='g4_material_joint_v1' else manifest['models'][name],binary_sha256=binary['sha256'],config_sha256=hashlib.sha256((d/'gpu.yaml').read_bytes()).hexdigest(),quality=q,audit=audit)
 (d/'gpu_status.json').write_text(json.dumps(status,indent=2));print(name,ok,status['elapsed_s'],flush=True)
 if not ok:raise RuntimeError('Failed dose must not be analyzed; inspect and split overflow: '+name)
