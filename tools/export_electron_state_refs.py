"""Lossless fixed-layout state-reference export for C++/device consumers."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from compile_electron_state_catalog import sha

p=argparse.ArgumentParser()
p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--metadata-sha256',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if sha(a.metadata)!=a.metadata_sha256:
    raise ValueError('Response metadata SHA mismatch')
m=json.loads(a.metadata.read_text());s=m['transport_states']
raw=a.metadata.parent/s['filename']
if raw.stat().st_size!=s['size_bytes'] or sha(raw)!=s['sha256']:
    raise ValueError('State reference NPZ changed')
ns=sum(c['count'] for c in m['channels']);nn=m['path_nodes']
sources=s['sources']
if not sources or len(sources)>0xffffffff:
    raise ValueError('Invalid source count')
for source in sources:
    if sha(source['path'])!=source['sha256']:
        raise ValueError('Raw transport source changed')
limit=np.array([s['rows'] for s in sources],dtype='<u8')
z=np.load(raw,allow_pickle=False)
sentinel=np.iinfo(np.uint64).max
for ids,rows,count in [('sample_source_id','sample_source_row',ns),('source_id','node_source_row',nn)]:
    if z[ids].shape!=(count,) or z[rows].shape!=(count,) or np.any(z[ids]>=len(sources)):
        raise ValueError('Invalid state reference shape/source')
    if np.any(z[rows]>=limit[z[ids]]):
        raise ValueError('State reference out of raw source')
edge=z['incoming_edge_source_row'];post=z['node_is_post_state']
if edge.shape!=(nn,) or post.shape!=(nn,) or np.any(post>1):
    raise ValueError('Invalid node states')
if np.any((edge!=sentinel)&(edge>=limit[z['source_id']])):
    raise ValueError('Incoming edge outside source')
sample=np.empty(ns,dtype=[('source','<u4'),('row','<u8')])
sample['source']=z['sample_source_id'];sample['row']=z['sample_source_row']
node=np.zeros(nn,dtype=[('source','<u4'),('row','<u8'),('edge','<u8'),('post','u1'),('reserved','u1',(3,))])
node['source']=z['source_id'];node['row']=z['node_source_row'];node['edge']=edge;node['post']=post
meta_path=a.output.with_suffix('.metadata.json')
if a.output.exists() or meta_path.exists():
    raise ValueError('Existing state-reference output')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('xb') as stream:
    stream.write(struct.pack('<8sIIQQ',b'ESTREF01',1,len(sources),ns,nn))
    stream.write(limit.tobytes());stream.write(sample.tobytes());stream.write(node.tobytes())
meta=dict(schema_version=1,status='STATE_REFERENCE_MAP_NOT_BOUNDARY_MODEL',
    data_filename=a.output.name,data_sha256=sha(a.output),data_size_bytes=a.output.stat().st_size,
    sources=sources,samples=ns,nodes=nn,
    response_metadata_path=str(a.metadata.resolve()),response_metadata_sha256=a.metadata_sha256,
    state_npz_sha256=s['sha256'],exporter_sha256=sha(__file__))
with meta_path.open('x') as stream:
    stream.write(json.dumps(meta,indent=2)+'\n')
print(ns,'sample refs,',nn,'node refs:',a.output,flush=True)
