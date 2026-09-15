#!/usr/bin/env python3
"""Verify the unified research EM binary against its material/species manifest."""
import argparse,hashlib,json,math,struct
from pathlib import Path

def verify(package):
    meta=json.loads(package.with_suffix('.json').read_text())
    if meta['schema']!='EMJOINT1' or meta['version']!=1:raise ValueError('Unsupported EM schema')
    digest=hashlib.sha256()
    with package.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=meta['package_sha256']:raise ValueError('EM SHA256 mismatch')
    if package.stat().st_size!=meta['package_bytes']:raise ValueError('EM size mismatch')
    with package.open('rb') as f:
        magic,version,nm,ns,nr,nn,nv=struct.unpack('<8s6I',f.read(32))
        if magic!=b'EMJOINT1' or version!=1 or ns!=18 or nr!=nm*ns:raise ValueError('Invalid EM header')
        if (nm,nr,nn,nv)!=(len(meta['materials']),meta['records'],meta['nodes'],meta['segments']):raise ValueError('Manifest/header mismatch')
        if 32+8*nm+8*ns+104*nr+52*nn+24*nv!=meta['package_bytes']:raise ValueError('Invalid layout')
        sections=set()
        for m in meta['materials']:
            section,rho=struct.unpack('<if',f.read(8));sections.add(section)
            if section!=m['section'] or not math.isclose(rho,m['density_g_cm3'],rel_tol=1e-6):raise ValueError('Material manifest mismatch')
        if sections!=set(range(-1,25)):raise ValueError('Missing water/Schneider section')
        species=[struct.unpack('<2I',f.read(8)) for _ in range(ns)]
        required={(1,1),(1,2),(1,3),(2,3),(2,4),(2,6),(3,6),(3,7),(4,6),(4,7),(4,9),(4,10),(5,8),(5,10),(5,11),(6,10),(6,11),(6,12)}
        if set(species)!=required or species!=[tuple(x) for x in meta['species_za']]:raise ValueError('Ion registry mismatch')
    print(f'OK: {nm} material/density nodes x {ns} ions, SHA256={digest.hexdigest()}')
    print('Integrity verified; production execution uses the AGENTS exception, while density cut-onset/patient accuracy validation remains pending.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',nargs='?',type=Path,default=Path(__file__).resolve().parents[1]/'data/em/unified_em_v1.bin')
    parser.add_argument("--core-only",action="store_true",help="Verify the source package before generating delta moments")
    args=parser.parse_args();verify(args.package)
    if not args.core_only:
        from verify_delta_moments import verify as verify_moments
        verify_moments(args.package.parent/"unified_em_delta_moments_v2.bin",args.package)
