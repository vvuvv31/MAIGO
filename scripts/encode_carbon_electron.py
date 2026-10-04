#!/usr/bin/env python3
"""Copy current C12 v7 artifacts to lossless zstd storage without changing identity.

Run prepare_material_electron.py --carbon-compact first for a newly extracted
bank. This utility verifies every plain block against its pinned SHA, copies
its existing manifest unchanged and writes only .bin.zst blocks in a new cache.
The runtime independently checks all decompressed bytes against that manifest.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
from prepare_material_electron import fingerprint
from migrate_carbon_electron import Encoder,key

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    doc=json.loads(a.index.read_text())
    if doc['projectile']!=[6,12]:raise ValueError('C12 index required')
    tables=sorted(doc['tables'],key=lambda t:(t['material_identity']['section_id'],t['material_identity']['runtime_density_g_cm3']))
    encoder=Encoder();version=fingerprint();a.cache.mkdir(parents=True,exist_ok=True);report=[]
    for i,table in enumerate(tables):
        identity=key(table,version,7);source=a.source/identity;dest=a.cache/identity
        if dest.exists():raise FileExistsError(f'Refuse to replace artifact: {dest}')
        raw=(source/'manifest').read_bytes();pin=(source/'manifest.sha256').read_text().strip()
        if hashlib.sha256(raw).hexdigest()!=pin:raise RuntimeError('Manifest SHA mismatch')
        lines=raw.decode().splitlines()
        if lines[0].split()[:2]!=['MAIGO_CARBON_EM_7',identity]:raise RuntimeError('Artifact version/identity mismatch')
        stage=a.cache/(identity+'.partial.'+str(os.getpid())+'.'+str(time.time_ns()));stage.mkdir();total=stored=0
        for b,line in enumerate(lines[1:]):
            name,n,w,digest=line.split();size=int(n)*int(w)
            if int(n)<0 or int(w)<=0:raise RuntimeError('Invalid block descriptor')
            src=source/f'{b}.bin';dst=stage/f'{b}.bin.zst'
            if src.stat().st_size!=size:raise RuntimeError('Block size mismatch')
            if encoder.copy(src,dst,size)!=digest:raise RuntimeError('Block SHA mismatch')
            total+=size;stored+=dst.stat().st_size
        (stage/'manifest').write_bytes(raw);(stage/'manifest.sha256').write_text(pin+'\n');stage.rename(dest)
        report.append(dict(table=i,key=identity,raw_bytes=total,stored_bytes=stored));print(json.dumps(report[-1]),flush=True)
    (a.cache/'encoding.json').write_text(json.dumps(dict(version=version,entries=report),indent=2)+'\n')
if __name__=='__main__':main()
