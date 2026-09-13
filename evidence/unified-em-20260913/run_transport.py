import subprocess,pathlib,time,json,os,re
r=pathlib.Path(__file__).resolve().parents[2];root=r/'scratch/unified_joint_em_20260913';env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib',ONEAPI_DEVICE_SELECTOR='cuda:*')
subprocess.run(['python3',str(r/'tools/verify_schneider_v2_1_data.py')],check=True)
for case in ['b1_100','b1_200','b1_300','b2_250','b3_layers','b4_soft_bone','b4_soft_lung']:
 d=root/'transport'/case;cfg=d/'gpu.yaml';cfg.write_text(cfg.read_text().replace('number_of_histories: 1000\n','number_of_histories: 50000\n'));t=time.time()
 with (d/'gpu.log').open('w') as f:p=subprocess.run([str(root/'build/carbon_mc'),'--config',str(cfg),'--device','cuda','--voxel-dose-mhd',str(d/'dose_gpu.mhd')],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
 qpath=d/'out/gpu/quality_report.json';q=json.load(qpath.open()) if qpath.exists() else {};audit=re.findall(r'\[unified-em-audit\]([^\n]+)',(d/'gpu.log').read_text())
 ok=p.returncode==0 and q.get('accepted',False) and q.get('queue_overflow_count',-1)==0 and len(audit)>0 and all(int(a.split()[0])==0 for a in audit)
 status=dict(complete=ok,returncode=p.returncode,histories=50000,elapsed_s=time.time()-t,audit=audit,quality=q);(d/'status.json').write_text(json.dumps(status,indent=2));print(case,ok,status['elapsed_s'],flush=True)
 if not ok:raise RuntimeError((d/'gpu.log').read_text()[-2000:])
