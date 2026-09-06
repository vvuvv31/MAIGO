"""Explicit isolated A/B control: SAME CDF/energy/data, collapse only paths.

Keeps schema 2 and unvalidated status. Not a compatibility downgrade/fallback.
Never overwrite the ordered candidate or the authoritative physics stack.
"""
import argparse,json,shutil,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def make(source,out):
    csv=source/'joint_response.csv';meta=csv.with_suffix('.metadata.json');m=json.loads(meta.read_text())
    if m['schema_version']!=2 or m['status']!='UNVALIDATED_INTERFACE_DIAGNOSTIC':raise ValueError('Requires isolated ordered candidate')
    binary=source/m['ordered_path_file']
    if sha(csv)!=m['data_sha256'] or sha(binary)!=m['ordered_path_sha256']:raise ValueError('Input SHA')
    with binary.open('rb') as f:
        magic,version,n,nv=struct.unpack('<8sIII',f.read(20))
        if magic!=b'ELPATH01' or version!=1 or binary.stat().st_size!=20+8*n+24*nv:raise ValueError('Path schema')
        ranges=np.fromfile(f,dtype='<u4',count=2*n).reshape(n,2);v=np.fromfile(f,dtype='<f8').reshape(nv,3)
    expected=np.r_[0,np.cumsum(ranges[:,1],dtype=np.uint64)[:-1]]
    if not np.array_equal(ranges[:,0],expected) or ranges[:,1].sum()!=nv or np.any(ranges[:,1]==0):raise ValueError('Path ranges')
    net=np.add.reduceat(v,ranges[:,0],axis=0)
    out.mkdir(exist_ok=False);shutil.copyfile(csv,out/csv.name)
    target=out/'ordered_paths.bin'
    with target.open('xb') as f:
        f.write(struct.pack('<8sIII',b'ELPATH01',1,n,n));f.write(np.column_stack((np.arange(n),np.ones(n))).astype('<u4').tobytes());f.write(net.astype('<f8').tobytes())
    m.update(ordered_path_file=target.name,ordered_path_sha256=sha(target),ordered_path_size_bytes=target.stat().st_size,
        diagnostic_ablation='net-chord only; same CSV and sampling as ordered counterpart',
        ordered_parent_metadata_sha256=sha(meta),ablation_compiler_sha256=sha(Path(__file__)))
    with (out/meta.name).open('x') as f:json.dump(m,f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();make(a.source,a.out)
