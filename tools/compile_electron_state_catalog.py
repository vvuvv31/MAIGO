"""Lossless row catalog for V4 continuation data; no fitted transport model.

All electron/photon rows remain in their pinned original payload. The catalog
groups exact row references by PDG and PRE kinetic energy, preserving track,
parent and step keys for correlated continuation. It does not rescale energies,
discard tails, or declare a production sampling law.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from electron_transport_state import map_transport_states


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compile_catalog(source,output):
    output=Path(output)
    if output.exists():raise ValueError('Existing catalog; do not overwrite')
    for path,pin in source['pins'].items():
        if sha(path)!=pin:raise ValueError('Changed source '+path)
    raw_path=Path(source['path']);rows=map_transport_states(raw_path)
    if len(rows)!=source['entries']:raise ValueError('Source entry count changed')
    # Units are MeV of the electron/photon, NOT MeV/u of the generating ion.
    edges=np.r_[0.,np.geomspace(1e-9,32.,257)]
    nb=len(edges)-1
    pdg=rows['pdg'];energy=rows['pre_ke_MeV']
    ids=np.flatnonzero(np.isin(pdg,[11,22]))
    e=np.asarray(energy[ids])
    if not len(ids) or not np.isfinite(e).all() or np.any(e<0) or np.any(e>edges[-1]):
        raise ValueError('Missing/invalid or out-of-domain continuation energy')
    if np.any(~np.isin(pdg,[1000060120,11,22])):raise ValueError('Unsupported source species must not be dropped')
    bins=np.searchsorted(edges,e,side='right')-1
    bins[e==edges[-1]]=nb-1
    keys=(pdg[ids]==22).astype(np.int64)*nb+bins
    order=np.argsort(keys,kind='stable');references=ids[order].astype('<u8')
    counts=np.bincount(keys,minlength=2*nb).astype('<u8')
    offsets=np.r_[np.uint64(0),np.cumsum(counts,dtype=np.uint64)].astype('<u8')
    if int(offsets[-1])!=len(ids):raise ValueError('State reference closure')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        np.savez(stream,energy_edges_MeV=edges.astype('<f8'),pdg=np.array([11,22],dtype='<i4'),
                 offsets=offsets,source_rows=references)
    meta=dict(schema_version=1,status='UNVALIDATED_CONTINUATION_ROW_CATALOG',
        source=source,data_filename=output.name,data_sha256=sha(output),data_size_bytes=output.stat().st_size,
        indexed_rows=len(ids),channels=2*nb,empty_channels=int(np.count_nonzero(counts==0)),
        energy_coordinate='particle_pre_kinetic_MeV',energy_min_MeV=float(e.min()),energy_max_MeV=float(e.max()),
        source_row_semantics='Zero-based row in pinned V4 binary; exact states and ancestry remain in source.',
        sampling_law='Not defined by catalog; uniform step selection is not a validated continuation model.',
        compiler_sha256=sha(__file__))
    output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(source['tag'],len(ids),'rows catalogued',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=json.loads(args.sources.read_text())
    matches=[s for s in manifest.get('sources',manifest.get('indexed',[])) if s['tag']==args.tag]
    if len(matches)!=1:raise ValueError('Missing or ambiguous source tag')
    compile_catalog(matches[0],args.output)
