#!/usr/bin/env python3
"""Convert SHA-verified audited raw v4 C12 artifacts to exact bitmap/graph32 v5.

Both roots must be on the same filesystem for immutable common-block hardlinks.
Old artifacts are preserved. Full GPU validation is required before promotion.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
from prepare_material_electron import ROOT,fingerprint
from migrate_carbon_electron import key
V4='285a84dde87e93f75b1302553dc07e0ba76b44229387478041fb66455fb3fc7e'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--begin',type=int,default=0);p.add_argument('--end',type=int);a=p.parse_args()
 doc=json.loads(a.index.read_text());assert doc['projectile']==[6,12];version='9a344a25b97b2184fd6a568fd00de2bd6ea7145721d8ac4e84134152cd818305';build=ROOT/'out/bitmap_tools'/version;build.mkdir(parents=True,exist_ok=True);exe=build/'migrate'
 if not exe.exists():subprocess.run(['g++','-O3','-std=c++20','-DCARBON_OPENSSL_SHA256=1',f'-DCARBON_MATERIAL_PREPARED_VERSION="{version}"','-I',str(ROOT/'include'),str(ROOT/'scripts/migrate_carbon_bitmap.cpp'),'-lcrypto','-lzstd','-o',str(exe)],check=True)
 a.cache.mkdir(parents=True,exist_ok=True);tables=sorted(doc['tables'],key=lambda t:(t['material_identity']['section_id'],t['material_identity']['runtime_density_g_cm3']));results=[]
 for i in range(a.begin,min(a.end if a.end is not None else len(tables),len(tables))):
  table=tables[i];old=key(table,V4,4);new=key(table,version,5);src=a.source/old;dest=a.cache/new;stage=a.cache/(new+'.partial.'+str(time.time_ns()));assert not dest.exists()
  head=(src/'manifest').read_text().splitlines()[0].split();assert int(head[2])==table['material_identity']['section_id'] and float(head[3])==table['material_identity']['runtime_density_g_cm3']
  started=time.monotonic();result=subprocess.run([str(exe),str(src),str(stage),old,new],check=True,text=True,capture_output=True);row=json.loads(result.stdout);stage.rename(dest);row.update(table=i,key=new,source_key=old,seconds=time.monotonic()-started);results.append(row);print(json.dumps(row),flush=True)
 (a.cache/f'migration_{a.begin}_{a.end}.json').write_text(json.dumps({'version':version,'source_version':V4,'index_sha256':hashlib.sha256(a.index.read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
if __name__=='__main__':main()
