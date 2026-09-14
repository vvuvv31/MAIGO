from pathlib import Path
import numpy as np,json,re
out=Path(__file__).resolve().parent;repo=out.parents[1];run=repo/'scratch/unified_em_perf_20260913/runtime_life_probe_1m/RT07575'
meta=re.findall(r'\[queue-life\] (primary|secondary) (\d+) (\d+)',(run/'gpu.log').read_text());data=np.fromfile(run/'em_track_lifetimes.bin','<u4');offset=0;batches=[]
for kind,idx,n in meta:
 n=int(n);v=data[offset:offset+n].astype(np.int64);offset+=n
 if kind=='secondary':batches.append(v)
assert offset==len(data)
def capacity(v):return int(np.pad(v,(0,(-len(v))%32)).reshape(-1,32).max(axis=1).sum()*32) if len(v) else 0
baseline=sum(capacity(v) for v in batches);work=sum(int(v.sum()) for v in batches)
rows=[]
for block in [32,64,128]:
 for cutoff in [1024,8192,34816]:
  cap=0;segments=0;compactions=0;resumes=0;tail_work=0;transport_entries=0;trace=[]
  for batch,v0 in enumerate(batches):
   v=v0.copy()
   while len(v):
    tail=len(v)<cutoff;step=v if tail else np.minimum(v,block);cap+=capacity(step);segments+=1;transport_entries+=len(v)
    if tail:tail_work+=int(v.sum());break
    remain=v-step;alive=remain[remain>0]
    if not len(alive):break
    compactions+=1;resumes+=len(alive);trace.append(dict(batch=batch,input_tracks=len(v),survivors=len(alive)));v=alive
  rows.append(dict(block=block,tail_cutoff=cutoff,em_calls=work,warp_capacity_proxy=cap,warp_useful_fraction=work/cap,proxy_capacity_reduction_percent=100*(1-cap/baseline),transport_launches=segments,compaction_boundaries=compactions,survivor_resumptions=resumes,transport_entries=transport_entries,tail_em_work_share_percent=100*tail_work/work,state_traffic_GB={str(size):2*size*resumes/1e9 for size in [128,256,512]},boundaries=trace))
print('baseline_capacity_proxy',baseline,'work',work)
for x in rows:print({k:v for k,v in x.items() if k!='boundaries'})
(out/'segment_compaction_model.json').write_text(json.dumps(dict(baseline_capacity_proxy=baseline,em_calls=work,rows=rows),indent=2)+'\n')
