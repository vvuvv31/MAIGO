from pathlib import Path
import numpy as np,json,hashlib,struct,yaml,os,subprocess,time,re
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';src=Path('/mnt/sda/wuwei/water75ev_c12_response_20260922');d=o/'delta_diagnostic_table';d.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
m=json.loads((src/'table.metadata.json').read_text());assert sha(src/'table.bin')==m['data_sha256'] and sha(src/'table.paths.bin')==m['path_sha256']
cd=np.dtype([('low','<f8'),('high','<f8'),('fraction','<f8'),('unresolved','<f8'),('offset','<u4'),('count','<u4')]);sd=np.dtype([('cdf','<f8'),('pre','<f8',3),('post','<f8',3)]);nd=np.dtype([('previous','<u4'),('reserved','<u4'),('point','<f8',3)])
magic,ver,nc,ns=struct.unpack('<8sIII',(src/'table.bin').open('rb').read(20));assert magic==b'WELSEG01' and ver==1
c=np.fromfile(src/'table.bin',dtype=cd,count=nc,offset=20);s=np.memmap(src/'table.bin',dtype=sd,mode='r',offset=20+40*nc,shape=(ns,))
pm,pv,pns,nn=struct.unpack('<8sIII',(src/'table.paths.bin').open('rb').read(20));assert pm==b'WELPTH01' and pns==ns
heads=np.memmap(src/'table.paths.bin',dtype='<u4',mode='r',offset=20,shape=(ns,));nodes=np.memmap(src/'table.paths.bin',dtype=nd,mode='r',offset=20+4*ns,shape=(nn,))
sel=[];out_samples=[];offset=0;aud=[];rng=np.random.default_rng(2026100301)
for i,ch in enumerate(c):
 if not ch['count']:continue
 a=s[int(ch['offset']):int(ch['offset']+ch['count'])];k=32768;u=(np.arange(k)+rng.random(k))/k;idx=np.searchsorted(a['cdf'],u,side='right');assert idx.max()<len(a)
 chosen=np.array(a[idx]);chosen['cdf']=(np.arange(k)+1)/k
 weights=np.diff(np.r_[0.,a['cdf']]);p=(a['pre']+a['post'])/2;ps=(chosen['pre']+chosen['post'])/2
 original=np.sum(weights[:,None]*p*p,axis=0);reduced=np.mean(ps*ps,axis=0)
 aud.append({'energy_low_MeVu':float(ch['low']),'original_segments':len(a),'selected':k,'midpoint_second_moment_original_mm2':original.tolist(),'midpoint_second_moment_reduced_mm2':reduced.tolist()})
 sel.append(idx+int(ch['offset']));out_samples.append(chosen);c[i]['offset']=offset;c[i]['count']=k;m['channels'][i]['offset']=offset;m['channels'][i]['count']=k;offset+=k
selected=np.concatenate(sel);outs=np.concatenate(out_samples);outheads=np.array(heads[selected]);needed=np.zeros(nn,bool);front=np.unique(outheads)
while len(front):
 front=front[~needed[front]]
 if not len(front):break
 needed[front]=True;front=np.unique(nodes['previous'][front]);front=front[front!=0xffffffff]
keep=np.flatnonzero(needed);assert len(keep)<20000000 and len(outs)<10000000
remap=np.full(nn,0xffffffff,dtype='<u4');remap[keep]=np.arange(len(keep),dtype='<u4');outnodes=np.array(nodes[keep]);pr=outnodes['previous'];mask=pr!=0xffffffff;pr[mask]=remap[pr[mask]];outheads=remap[outheads];assert np.all((pr==0xffffffff)|(pr<np.arange(len(pr))))
# Zero-length channels after nonzero channels are not present in this table.
assert all(ch['offset']==v['offset'] and ch['count']==v['count'] for ch,v in zip(c,m['channels']))
with (d/'table.bin').open('xb') as f:f.write(struct.pack('<8sIII',magic,ver,nc,len(outs)));c.tofile(f);outs.tofile(f)
with (d/'table.paths.bin').open('xb') as f:f.write(struct.pack('<8sIII',pm,pv,len(outs),len(outnodes)));outheads.tofile(f);outnodes.tofile(f)
m.update(data_sha256=sha(d/'table.bin'),data_size_bytes=(d/'table.bin').stat().st_size,path_sha256=sha(d/'table.paths.bin'),path_size_bytes=(d/'table.paths.bin').stat().st_size,path_nodes=len(outnodes))
m['diagnostic_compression']={'algorithm':'Independent stratified inverse-CDF sampling, 32768 equal-weight deposits per nonempty energy channel; exact ancestor-graph closure; unchanged loss fractions, escape fractions and energies. For diagnostic sensitivity only, not production or 1% certification.','seed':2026100301,'source_table':str(src/'table.bin'),'source_metadata_sha256':sha(src/'table.metadata.json'),'moments':aud}
(d/'table.metadata.json').write_text(json.dumps(m,indent=2)+'\n');print('TABLE',len(outs),len(outnodes),flush=True)
# Avoid sharing the GPU with the independent-batch experiment.
while True:
 q=o/'gpu_quality.json'
 if q.exists() and len(json.loads(q.read_text())['runs'])==12:break
 time.sleep(15)
env=dict(os.environ);env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
base=yaml.safe_load((r/'config/review_fullchain_topas_matched_1m.yaml').read_text());summary={}
for label,n,seed in [('smoke',20000,2026100300)]+[(f's{i}',1000000,2026092800+i) for i in [1,2,3]]:
 name=f'percent1_delta_{label}';cfg=dict(base);spot=o/f'delta_source_{n}.csv';spot.write_text(f'spot_id,x,y,energy,weight\n0,0,0,3000,{n}\n')
 cfg.update(number_of_histories=n,random_seed=seed,tps_spots_file=str(spot),minibeam_water_delta_response_model='water_response_v1',minibeam_water_delta_response_table_file=str(d/'table.bin'),water_electron_response_sha256=m['data_sha256'],water_electron_response_metadata_sha256=sha(d/'table.metadata.json'))
 cp=r/f'config/{name}.yaml';assert not cp.exists();cp.write_text(yaml.safe_dump(cfg,sort_keys=False));start=time.monotonic();lp=o/f'{name}.log'
 with lp.open('x') as f:p=subprocess.run([str(r/'build/oneapi-nvidia-minibeam/carbon_mc'),'--config',str(cp)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:print(lp.read_text()[-5000:],flush=True);raise RuntimeError(name)
 q=json.loads((r/f'out/{name}/quality_report.json').read_text());u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',lp.read_text()).group(1));assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
 summary[name]={'wall_s':time.monotonic()-start,'histories':n,'seed':seed,'quality':q,'urban':u};(o/'delta_quality.json').write_text(json.dumps(summary,indent=2)+'\n');print('DELTA',name,summary[name]['wall_s'],flush=True)
