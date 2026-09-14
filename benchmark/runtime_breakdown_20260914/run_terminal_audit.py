from pathlib import Path
import subprocess,sys,os,json,numpy as np
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';runs=repo/'scratch/unified_em_perf_20260913';out=Path(__file__).resolve().parent
for label,build in [('segment_terminal_base','terminal_base_build'),('segment_terminal_64','terminal_segment_build')]:
 d=runs/label/'RT07575'
 if (d/'status.json').exists():assert json.loads((d/'status.json').read_text())['complete'];continue
 subprocess.run([sys.executable,'tools/verify_schneider_v2_1_data.py'],cwd=repo,check=True)
 env=dict(os.environ,CARBON_BENCH_PROFILE='0',CARBON_RUNTIME_BREAKDOWN='1',CARBON_BENCH_CONFIG_ROOT=str(root/'segmented_inputs_50k'),CARBON_BENCH_CONFIG_OVERRIDES='{}')
 subprocess.run([sys.executable,'benchmark/unified_em_performance_20260913/run.py',label,str(root/build/'carbon_mc'),'RT07575'],cwd=repo,env=env,check=True)
ints=['parent','generation','z','a','steps','flags','bin','voxel'];dtype=np.dtype([('stream','<u8'),('rng','<u8')]+[(k,'<u4') for k in ints]+[('values','<f4',(16,))]);assert dtype.itemsize==112
items=[];statuses=[]
for label in ['segment_terminal_base','segment_terminal_64']:
 p=runs/label/'RT07575';statuses.append(json.loads((p/'status.json').read_text()));v=np.fromfile(p/'secondary_terminal.bin',dtype);v=v[(v['flags']&1)!=0]
 order=np.lexsort(tuple(v[k] for k in ['a','z','generation','parent','stream']));items.append(v[order])
a,b=items;assert a.shape==b.shape
keys=['stream','parent','generation','z','a'];assert all(np.array_equal(a[k],b[k]) for k in keys)
identity=np.empty(len(a),dtype=np.dtype([(k,dtype[k]) for k in keys]));
for k in keys:identity[k]=a[k]
assert len(np.unique(identity))==len(a),'duplicate particle identities'
integer_mismatches={k:int(np.count_nonzero(a[k]!=b[k])) for k in ['rng']+ints}
x=a['values'];y=b['values'];names=['energy','x','y','z','dx','dy','dz','pending_depth','pending_voxel','delta_remaining','delta_threshold','delta_rate0','delta_rate1','delta_density','tally_all','tally_fov']
float_mismatches={name:dict(bitwise_mismatches=int(np.count_nonzero(x[:,i].view('<u4')!=y[:,i].view('<u4'))),max_absolute_difference=float(np.max(np.abs(x[:,i].astype(float)-y[:,i].astype(float))))) for i,name in enumerate(names)}
result=dict(tracks=len(a),identities_equal=True,integer_mismatches=integer_mismatches,float_mismatches=float_mismatches,statuses=statuses)
(out/'segment_terminal_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='statuses'},indent=2))

assert all(v==0 for v in integer_mismatches.values()),'integer terminal state mismatch'
assert all(v['bitwise_mismatches']==0 for v in float_mismatches.values()),'floating terminal state mismatch'
for key in ['config_sha256','audit','steps']:assert statuses[0][key]==statuses[1][key]
