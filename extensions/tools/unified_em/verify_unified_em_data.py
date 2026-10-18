#!/usr/bin/env python3
"""Verify the unified research EM binary against its material/species manifest."""
import argparse,json,math,struct
from pathlib import Path

def verify(package):
    meta=json.loads(package.with_suffix('.metadata.json').read_text())
    if meta['schema'] not in ('EMJOINT1','EMFAC001','EMCNT001') or meta['version']!=1:raise ValueError('Unsupported EM schema')
    if package.stat().st_size!=meta['package_bytes']:raise ValueError('EM size mismatch')
    with package.open('rb') as f:
        magic,version,nm,ns,nr,nn,nv=struct.unpack('<8s6I',f.read(32))
        if magic.decode('ascii')!=meta['schema'] or version!=1 or not 26<=nm<=256 or ns!=18 or nr!=nm*ns:raise ValueError('Invalid EM header')
        if (nm,nr,nn,nv)!=(len(meta['materials']),meta['records'],meta['nodes'],meta['segments']):raise ValueError('Manifest/header mismatch')
        if magic==b'EMJOINT1' and 32+8*nm+8*ns+104*nr+52*nn+24*nv!=meta['package_bytes']:raise ValueError('Invalid layout')
        sections=set()
        for m in meta['materials']:
            section,rho=struct.unpack('<if',f.read(8));sections.add(section)
            if section!=m['section'] or not math.isclose(rho,m['density_g_cm3'],rel_tol=1e-6):raise ValueError('Material manifest mismatch')
        if sections!=set(range(-1,25)):raise ValueError('Missing water/Schneider section')
        species=[struct.unpack('<2I',f.read(8)) for _ in range(ns)]
        required={(1,1),(1,2),(1,3),(2,3),(2,4),(2,6),(3,6),(3,7),(4,6),(4,7),(4,9),(4,10),(5,8),(5,10),(5,11),(6,10),(6,11),(6,12)}
        if set(species)!=required or species!=[tuple(x) for x in meta['species_za']]:raise ValueError('Ion registry mismatch')
        if magic==b'EMCNT001':
            if nm!=26 or meta.get('density_response_states')!=0:raise ValueError('Continuous EM must have26 compositions and zero density response states')
            base_end=32+8*nm+8*ns+104*nr+52*nn+24*nv
            f.seek(base_end);degree,scalar_count,descriptors,rows=struct.unpack('<4I',f.read(16))
            if not 2<=degree<=64 or not 2<=scalar_count<=4096 or descriptors!=468 or rows>1000000:raise ValueError('Invalid continuous extension')
            if base_end+16+64*descriptors+26*scalar_count*32+4*rows*(degree+1)!=meta['package_bytes']:raise ValueError('Invalid continuous layout')
        if magic==b'EMFAC001':
            from verify_density_encoding import verify_em
            verify_em(f,nm,ns,nr,nn,nv)
    label = 'compositions, continuous density' if magic == b'EMCNT001' else 'material/density nodes'
    print(f'OK: {nm} {label} x {ns} ions')
    print('Stored layout/fields and registry verified; physics fields are checked by the C++ loader, dose by separate reference comparisons.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',nargs='?',type=Path,default=Path(__file__).resolve().parents[3]/'data/em/ions/continuous26.bin')
    args=parser.parse_args();verify(args.package)
