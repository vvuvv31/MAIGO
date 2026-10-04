#!/usr/bin/env python3
"""Append independently extracted C12 birth nodes; preserve all original node bytes."""
from pathlib import Path
import argparse,hashlib,json,struct,shutil

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--extension',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 old=a.base/'birth_bank.bin';m=json.loads((a.base/'extraction_provenance.json').read_text());assert sha(old)==m['bank_sha256']
 with old.open('rb') as f:h=struct.unpack('<8sIIII',f.read(24))
 assert h==(b'MAIBIR02',2,63,6,12)
 nodes=[]
 for d in sorted(a.extension.glob('node*')):
  x=json.loads((d/'complete.json').read_text());assert x['source_primary_MeV']==3060 and x['histories']>=64
  assert sha(d/'birth.bin')==x['sha256'];nodes.append((d,x))
 assert [x['node'] for _,x in nodes]==list(range(63,82))
 a.out.mkdir(parents=True,exist_ok=False);dest=a.out/'birth_bank.bin'
 with dest.open('xb') as f:
  f.write(struct.pack('<8sIIII',b'MAIBIR02',2,82,6,12))
  with old.open('rb') as base:base.seek(24);shutil.copyfileobj(base,f)
  for d,_ in nodes:
   with (d/'birth.bin').open('rb') as src:shutil.copyfileobj(src,f)
 record=dict(schema='MAIBIR02',projectile=[6,12],bank_sha256=sha(dest),bytes=dest.stat().st_size,base_bank_sha256=sha(old),base_nodes=63,extension_nodes=19,extension=[x for _,x in nodes],scope='Research C12 birth: original63 nodes unchanged plus19 native64-history nodes at255MeV/u, 0.05mm production cut, <=0.25mm primary step. Independent electron/photon propagation remains in shared package; broadbeam dose acceptance pending.',assembler_sha256=sha(__file__))
 (a.out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(record['bank_sha256'],record['bytes'])
if __name__=='__main__':main()
