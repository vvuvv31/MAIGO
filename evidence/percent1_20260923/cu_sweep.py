from pathlib import Path
import os,sys,yaml,json,subprocess,shutil,time,re,hashlib,concurrent.futures,numpy as np
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');parent=r/'evidence/percent1_20260923';o=parent/'cu_step025';o.mkdir(exist_ok=True);mode=sys.argv[1]
binary=r/'build/oneapi-nvidia-minibeam/carbon_mc';ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
if mode=='gpu':
 while not (parent/'delta_fixed_quality.json').exists() or len(json.loads((parent/'delta_fixed_quality.json').read_text())['runs'])<7:time.sleep(10)
 expected=json.loads((parent/'delta_fixed_quality.json').read_text())['fixed_binary_sha256'];assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
 env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
 base=yaml.safe_load((r/'config/review_fullchain_topas_matched_1m.yaml').read_text());rows=[]
 for i in [1,2,3]:
  cfg=dict(base);cfg.update(random_seed=2026092800+i,minibeam_copper_max_step_mm=.025);name=f'percent1_cu0025_s{i}';cp=r/f'config/{name}.yaml';assert not cp.exists();cp.write_text(yaml.safe_dump(cfg,sort_keys=False));start=time.monotonic();logpath=o/f'{name}.log'
  with logpath.open('x') as f:p=subprocess.run([str(binary),'--config',str(cp)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:print(logpath.read_text()[-3000:],flush=True);raise RuntimeError(name)
  q=json.loads((r/f'out/{name}/quality_report.json').read_text());u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',logpath.read_text()).group(1));assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
  rows.append({'name':name,'histories':1000000,'seed':cfg['random_seed'],'wall_s':time.monotonic()-start,'quality':q,'urban':u});(o/'gpu_quality.json').write_text(json.dumps({'binary_sha256':expected,'runs':rows},indent=2)+'\n');print('CU025_GPU',name,rows[-1]['wall_s'],flush=True)
elif mode=='topas':
 def one(i):
  n=1000000;seed=2026100400+i;d=o/f'topas_1m_s{i}';d.mkdir(exist_ok=False)
  for f in ['run_energy_scan.txt','run1.txt','beam_model_single_center_e250.csv']:shutil.copyfile(ref/f,d/f)
  ap=(ref/'aperture.txt').read_text();ap,k=re.subn(r'(d:Ge/[^\n]+/MaxStepSize\s*=\s*)0\.05(\s+mm)',r'\g<1>0.025\2',ap);assert k>0;(d/'aperture.txt').write_text(ap)
  (d/'spots_single_center.csv').write_text(f'spot_id,x,y,energy,weight\n0,0,0,3000,{n}\n')
  s=(ref/'run_single_center_em_only_water75ev_e250_10m.txt').read_text().replace('spots_single_center_e250_10m.csv','spots_single_center.csv');s+=f'\ni:Ts/Seed = {seed}\ns:Sc/DoseAtPhantomP/IfOutputFileAlreadyExists = "Exit"\n';(d/'run.txt').write_text(s);start=time.monotonic()
  with (d/'run.log').open('x') as f:p=subprocess.run(['bash','-c','source /software/env_topas.sh && exec /software/topas/bin/topas run.txt'],cwd=d,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:print((d/'run.log').read_text()[-3000:],flush=True);raise RuntimeError(d)
  log=(d/'run.log').read_text();assert f'Particle source CarbonPBS: Total number of histories: {n}\n' in log
  dose=np.fromfile(d/'dose.bin',dtype='<f8');assert len(dose)==1000000 and np.all(np.isfinite(dose)) and min(dose)>=0
  row={'name':d.name,'histories':n,'seed':seed,'threads':128,'wall_s':time.monotonic()-start,'aperture_maxstep_overrides':k,'note':'Water max step unchanged at 0.05 mm; TOPAS uses independent seeds; source/seed pairing is used only for the GPU Cu0.05-to-0.025 contrast.'};(d/'quality.json').write_text(json.dumps(row,indent=2)+'\n');print('CU025_TOPAS',json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(one,[1,2,3]))
 (o/'topas_quality.json').write_text(json.dumps(rows,indent=2)+'\n')
else:raise ValueError(mode)
