#!/usr/bin/env python3
"""Verify the accepted derived delta moments; never substitute older/water data."""
import argparse,hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/em/unified_em_delta_moments_v2.json'
SHA='c551bc52fa30ff7e3ea229c8b89792e8ecd2b0fb12f18fcad50d1c6f206bc7cc'
SOURCE_SHA='8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(path,source=None):
    path=Path(path);m=json.loads(MANIFEST.read_text())
    if m['sha256']!=SHA or m['source_package_sha256']!=SOURCE_SHA:raise ValueError('Unapproved delta manifest')
    if not path.exists():raise FileNotFoundError(f'{path}: generate with python3 tools/build_delta_moments.py')
    if path.stat().st_size!=m['bytes'] or digest(path)!=SHA:raise ValueError('Delta moments size/SHA mismatch')
    with path.open('rb') as f:magic,version,nodes=struct.unpack('<8sII',f.read(16))
    if (magic,version,nodes)!=(b'EMDMOMT2',2,m['nodes']) or 16+8*nodes!=m['bytes']:raise ValueError('Invalid delta moments layout')
    if source is not None and digest(source)!=SOURCE_SHA:raise ValueError('Delta source package SHA mismatch')
    print(f'OK: condensed delta moments, {nodes} nodes, SHA256={SHA}')
    return m
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('table',nargs='?',type=Path,default=ROOT/'data/em/unified_em_delta_moments_v2.bin');p.add_argument('--source',type=Path,default=ROOT/'data/em/unified_em_v1.bin');a=p.parse_args();verify(a.table,a.source)
