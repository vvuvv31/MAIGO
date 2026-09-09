"""Build exact next-step links for continuation; never resample stored steps."""
import argparse
import json
from pathlib import Path
import numpy as np
from electron_transport_state import map_transport_states
from compile_electron_state_catalog import sha


def generating_step_links(rows, heads):
    """Child CSR keyed by exact generating raw row, not spatial proximity."""
    dtype = np.dtype([(n, '<i4') for n in ('run', 'event', 'track', 'step')])
    keys = np.empty(len(rows), dtype=dtype)
    for name in dtype.names:
        keys[name] = rows[name]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError('Duplicate generating-step identity')
    children = heads[rows['parent'][heads] != 0]
    if np.any(rows['parent_valid'][children] != 1) or np.any(rows['parent_step_id'][children] <= 0):
        raise ValueError('Child missing exact generating-step identity')
    query = np.empty(len(children), dtype=dtype)
    query['run'] = rows['run'][children]
    query['event'] = rows['event'][children]
    query['track'] = rows['parent'][children]
    query['step'] = rows['parent_step_id'][children]
    found = np.searchsorted(sorted_keys, query)
    if np.any(found == len(rows)):
        raise ValueError('Generating parent step absent from source')
    if np.any(sorted_keys[found] != query):
        raise ValueError('Generating parent step absent from source')
    parents = order[found]
    if np.any(rows['track'][parents] == rows['track'][children]):
        raise ValueError('Self-parent track')
    permutation = np.argsort(parents, kind='stable')
    offsets = np.zeros(len(rows) + 1, dtype='<u8')
    offsets[1:] = np.bincount(parents, minlength=len(rows))
    np.cumsum(offsets, out=offsets)
    return offsets, children[permutation].astype('<u8')


def compile_links(source,output):
    if output.exists():raise ValueError('Track links already exist')
    raw_path=Path(source['path'])
    if sha(raw_path)!=source['pins'][str(raw_path)]:raise ValueError('Changed state payload')
    rows=map_transport_states(raw_path)
    order=np.lexsort((rows['step'],rows['track'],rows['event'],rows['run']))
    same=np.ones(max(0,len(rows)-1),dtype=bool)
    for key in ('run','event','track'):
        same &= rows[key][order[1:]]==rows[key][order[:-1]]
    starts=np.r_[0,np.flatnonzero(~same)+1]
    if np.any(rows['step'][order[starts]]!=1):raise ValueError('Missing first track step')
    if np.any((np.diff(rows['step'][order])!=1)&same):raise ValueError('Missing/duplicate track steps')
    sentinel=np.iinfo(np.uint64).max
    links=np.full(len(rows),sentinel,dtype='<u8')
    links[order[:-1][same]]=order[1:][same]
    heads=order[starts].astype('<u8')
    if np.count_nonzero(links==sentinel)!=len(heads):raise ValueError('Track terminal count mismatch')
    child_offsets, child_heads = generating_step_links(rows, heads)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        np.savez(stream,next_source_row=links,track_first_source_row=heads,
                 child_offsets=child_offsets,child_first_source_row=child_heads)
    metadata=dict(schema_version=2,status='EXACT_TRACK_LINKS_NOT_TRANSPORT_VALIDATION',
        source_path=str(raw_path),source_sha256=sha(raw_path),rows=len(rows),tracks=len(heads),
        terminal_sentinel=int(sentinel),data_filename=output.name,data_sha256=sha(output),
        compiler_sha256=sha(__file__),children=len(child_heads),
        semantics='Same-track links and child CSR by exact (run,event,parent,parent_step_id); no continuation sampling law implied.')
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(source['tag'],len(rows),'rows',len(heads),'tracks linked',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    matches=[s for s in json.loads(a.sources.read_text())['sources'] if s['tag']==a.tag]
    if len(matches)!=1:raise ValueError('Missing/ambiguous source')
    compile_links(matches[0],a.output)
