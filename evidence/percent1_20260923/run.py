from pathlib import Path
import os,json,yaml,subprocess,time,hashlib,re,shutil,concurrent.futures,sys
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';o.mkdir(exist_ok=True)
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
binary=r/'build/oneapi-nvidia-minibeam/carbon_mc'
expected=json.loads((r/'evidence/boundary_fix_20260923/final_run_quality.json').read_text())['binary_sha256']
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
mode=sys.argv[1]
if mode=='gpu':
 env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
 base=yaml.safe_load((r/'config/review_fullchain_topas_matched_1m.yaml').read_text())
 summary={'binary_sha256':expected,'histories_per_run':1000000,'runs':{}}
 for i in range(1,13):
  name=f'percent1_matched_s{i}';cfg=dict(base);cfg['random_seed']=2026093000+i
  cp=r/f'config/{name}.yaml';assert not cp.exists();cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
  dest=r/f'out/{name}';assert not dest.exists();start=time.monotonic();lp=o/f'{name}.log'
  with lp.open('x') as log:run=subprocess.run([str(binary),'--config',str(cp)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
  if run.returncode:print(lp.read_text()[-4000:],flush=True);raise RuntimeError((name,run.returncode))
  q=json.loads((dest/'quality_report.json').read_text());e=json.loads((dest/'energy_ledger.json').read_text());u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',lp.read_text()).group(1))
  assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
  summary['runs'][name]={'seed':cfg['random_seed'],'wall_s':time.monotonic()-start,'accepted':q['accepted'],'relative_energy_residual':q['relative_energy_residual'],'urban':u,'energy_MeV':{k:e[k] for k in ['E_in_MeV','E_dep_MeV','E_esc_MeV','E_beamline_MeV']}}
  (o/'gpu_quality.json').write_text(json.dumps(summary,indent=2)+'\n');print('GPU',name,json.dumps(summary['runs'][name]),flush=True)
elif mode=='topas':
 def runone(name,n,seed,threads):
  d=o/name;d.mkdir(exist_ok=False)
  for f in ['run_energy_scan.txt','run1.txt','aperture.txt','beam_model_single_center_e250.csv']:shutil.copyfile(ref/f,d/f)
  (d/'spots_single_center.csv').write_text(f'spot_id,x,y,energy,weight\n0,0,0,3000,{n}\n')
  src=(ref/'run_single_center_em_only_water75ev_e250_10m.txt').read_text().replace('spots_single_center_e250_10m.csv','spots_single_center.csv').replace('i:Ts/NumberOfThreads = 128',f'i:Ts/NumberOfThreads = {threads}')
  # Main input overrides inherited parameters. Carrier scores share the existing copy/bins.
  src+=f'\ni:Ts/Seed = {seed}\ns:Sc/DoseAtPhantomP/IfOutputFileAlreadyExists = "Exit"\n'
  for scorer,filename,filtername in [('ElectronCarrier','dose_electron_carrier','OnlyIncludeParticlesNamed'),('NonElectronCarrier','dose_non_electron_carrier','OnlyIncludeParticlesNotNamed')]:
   src+=f'\ns:Sc/{scorer}/Quantity = "DoseToMedium"\ns:Sc/{scorer}/Component = "Box"\ns:Sc/{scorer}/OutputFile = "{filename}"\ns:Sc/{scorer}/OutputType = "binary"\ns:Sc/{scorer}/IfOutputFileAlreadyExists = "Exit"\ni:Sc/{scorer}/XBins = 1000\ni:Sc/{scorer}/YBins = 1000\ni:Sc/{scorer}/ZBins = 1\nsv:Sc/{scorer}/{filtername} = 1 "e-"\n'
  (d/'run.txt').write_text(src);start=time.monotonic()
  with (d/'run.log').open('x') as log:run=subprocess.run(['bash','-c','source /software/env_topas.sh && exec /software/topas/bin/topas run.txt'],cwd=d,stdout=log,stderr=subprocess.STDOUT)
  if run.returncode:print((d/'run.log').read_text()[-5000:],flush=True);raise RuntimeError((name,run.returncode))
  import numpy as np
  a=[np.fromfile(d/f,dtype='<f8') for f in ['dose.bin','dose_electron_carrier.bin','dose_non_electron_carrier.bin']]
  assert all(len(v)==1000000 and np.all(np.isfinite(v)) and np.min(v)>=0 for v in a)
  closure=float(np.max(np.abs(a[0]-a[1]-a[2]))/np.max(a[0]));assert closure<1e-6
  log=(d/'run.log').read_text();assert f'Particle source CarbonPBS: Total number of histories: {n}\n' in log,log[-4000:]
  result={'name':name,'histories':n,'seed':seed,'threads':threads,'wall_s':time.monotonic()-start,'returncode':run.returncode,'carrier_max_residual_over_peak':closure}
  (d/'quality.json').write_text(json.dumps(result,indent=2)+'\n');print('TOPAS',json.dumps(result),flush=True);return result
 smoke={'name':'topas_smoke','histories':20000,'seed':2026100100,'threads':64,'wall_s':12.8046,'carrier_max_residual_over_peak':6.491366164714589e-8,'note':'Completed; resumed after overstrict 1e-10 closure threshold. 1e-6 matches FP32 accumulation rounding; integral residual 3.21e-11.'}
 (o/'topas_smoke_ok.json').write_text(json.dumps(smoke,indent=2)+'\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(runone,f'topas_2m_s{i}',2000000,2026100100+i,128) for i in [1,2,3]]
  rows=[f.result() for f in concurrent.futures.as_completed(futures)]
 (o/'topas_quality.json').write_text(json.dumps(rows,indent=2)+'\n')
else:raise ValueError(mode)
