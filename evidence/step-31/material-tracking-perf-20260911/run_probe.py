import sys,json,time,subprocess,os,struct,hashlib,re
from pathlib import Path
import yaml,numpy as np
repo=Path('/mnt/sdb/wuwei/MAIGO');sys.path.insert(0,str(repo/'tools'))
from run_topas10x_gpu_benchmark import config_write
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
n=int(sys.argv[1]);out=Path('/mnt/sda/wuwei/material_tracking_perf_20260911/20022516')/f'n{n}';out.mkdir(parents=True,exist_ok=False)
c=yaml.safe_load(Path('/mnt/sda/wuwei/final_stopping_20260911/20022516_extended/shard_01/config.yaml').read_text())
index=repo/'benchmark/benchmark20260908/material_response_states_450_candidate/material_response_runtime_index.json'
assert sha(index)=='79bb2a62c8080ee9850845dd5b104936cc06db007f9c19c3fb10b20b42fd271e'
with Path(c['ct_grid_file']).open('rb') as f:
 magic,ver,nx,ny,nz,ox,oy,oz,sx,sy,sz=struct.unpack('<5I6f',f.read(44));rho=np.fromfile(f,'<f4',nx*ny*nz);sec=np.fromfile(f,'u1',rho.size)
tables=json.loads(index.read_text())['tables'];selected=set()
for section in np.unique(sec):
 ds=np.unique(rho[sec==section]);nodes=sorted((float(np.float32(t['material_identity']['schneider_identity']['density_g_cm3'])),i) for i,t in enumerate(tables) if t['material_identity'].get('schneider_identity',{}).get('material_section')==int(section))
 assert nodes and ds[0]>=nodes[0][0] and ds[-1]<=nodes[-1][0],('density coverage',section,ds[0],ds[-1])
 v=np.array([x[0] for x in nodes])
 for d in ds:
  hi=int(np.searchsorted(v,d));selected.add(nodes[hi][1])
  if v[hi]!=d:selected.add(nodes[hi-1][1])
raw=sum(s['rows']*312 for i in selected for s in tables[i]['transport_states']['sources']);assert raw<64*1024**3
binary=repo/'build/oneapi-nvidia-stopping-final/carbon_mc'
state={'histories':n,'binary_sha256':sha(binary),'index_sha256':sha(index),'raw_bank_GiB':raw/1024**3,'selected_tables':len(selected),'runs':[]}
(out/'summary.json').write_text(json.dumps(state,indent=2))
print('preflight',n,state['raw_bank_GiB'],len(selected),flush=True)
subprocess.run(['python3',str(repo/'tools/verify_schneider_v2_1_data.py')],cwd=repo,check=True)
subprocess.run(['python3',str(repo/'tools/verify_schneider_ion_stopping.py'),c['ct_secondary_ion_section_stopping_file']],cwd=repo,check=True)
c.update(number_of_histories=n,tps_spot_weight_mode='mu',tps_histories_scale=1.0,history_chunk_size=n,voxel_bins_x=nx,voxel_bins_y=ny,voxel_bins_z=nz,voxel_size_x_mm=sx,voxel_size_y_mm=sy,voxel_size_z_mm=sz,phantom_length_mm=nz*sz,depth_bin_width_mm=sz,scorer_area_mm2=nx*sx*ny*sy)
for name in ['baseline','material']:
 d=out/name;d.mkdir();cfg=dict(c)
 if name=='material':
  cfg['ct_schneider_delta_tail_file']='';cfg.update(material_electron_response_index_file=str(index),material_electron_response_index_sha256=sha(index),material_electron_response_memory_mode='host_mapped',material_electron_response_host_budget_MiB=98304,material_electron_response_device_budget_MiB=8192,material_electron_short_range_mm=0)
 config_write(d/'config.yaml',cfg);(d/'data').symlink_to(repo/'data',target_is_directory=True)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 start=time.monotonic()
 with (d/'gpu.log').open('w') as log:
  p=subprocess.run(['/usr/bin/time','-v','-o',str(d/'resources.txt'),str(binary),'--config',str(d/'config.yaml'),'--device','cuda','--voxel-dose-mhd',str(d/'dose.mhd')],cwd=d,env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='cuda:*',LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib'),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 text=(d/'gpu.log').read_text();record={'mode':name,'wall_seconds':time.monotonic()-start,'returncode':p.returncode,'directory':str(d)}
 match=re.search(r'Kernel time: primary=([\d.e+-]+) s secondary=([\d.e+-]+) s neutral=([\d.e+-]+) s charged-after-neutral=([\d.e+-]+)',text)
 if match:record.update(primary_kernel_seconds=float(match[1]),total_kernel_seconds=sum(map(float,match.groups())))
 qpath=d/'out/config/quality_report.json'
 if qpath.exists():
  q=json.loads(qpath.read_text());record['quality']=q
  assert q['queue_overflow_count']==0,'Overflow: reject and split paired source before retry'
  assert all(x['code']=='unvalidated_material_electron_response' for x in q['failures']),q['failures']
 else:record['error']=text[-2500:]
 state['runs'].append(record);(out/'summary.json').write_text(json.dumps(state,indent=2));print(name,record,flush=True)
 if not qpath.exists():raise RuntimeError('No quality output')
