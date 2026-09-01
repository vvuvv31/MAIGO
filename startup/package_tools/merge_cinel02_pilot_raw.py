#!/usr/bin/env python3
"""Stream valid pilot campaigns into one collision-identity-safe CINEL02 raw file."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cinel02
from audit_cinel02_unsupported import iter_records

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--summary',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-output',type=Path,required=True);p.add_argument('--campaign-uuid',required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.summary.open())); a.output.parent.mkdir(parents=True,exist_ok=True)
 interactions=products=0
 with a.output.open('w+b') as out:
  out.write(b'\0'*cinel02.RAW_HEADER_SIZE)
  for run_index,row in enumerate(rows,1):
   name='z{}a{}_e{}_h{}_{}'.format(row['projectile_z'],row['projectile_a'],row['source_energy_MeV_per_u'],row['histories'],row['version'])
   for path in sorted((a.root/name).glob('raw/*/worker_*.cinel02')):
    for record,event_products in iter_records(path):
     record['run_id']=run_index
     out.write(cinel02.pack_record(record,event_products)); interactions+=1; products+=len(event_products)
  size=out.tell(); out.seek(0); out.write(cinel02.RAW_HEADER.pack(cinel02.RAW_MAGIC,cinel02.RAW_VERSION,cinel02.RAW_HEADER_SIZE,0x01020304,0,interactions,products,size,0,0))
 first=rows[0]; first_name='z{}a{}_e{}_h{}_{}'.format(first['projectile_z'],first['projectile_a'],first['source_energy_MeV_per_u'],first['histories'],first['version'])
 contracts=list((a.root/first_name).glob('raw/*/cinel02.contract.json'))
 metadata=json.loads(contracts[0].read_text()); metadata['campaign_uuid']=a.campaign_uuid
 metadata.setdefault('provenance',{})['merge']='102 primary-only projectile/source-energy pilot campaigns; run_id reassigned 1..102'
 a.metadata_output.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
 print({'interactions':interactions,'products':products,'bytes':size,'campaign_uuid':a.campaign_uuid})
if __name__=='__main__':main()
