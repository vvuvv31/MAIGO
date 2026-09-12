from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import csv,json,re,shutil,hashlib
root=Path(__file__).resolve().parent
remote='/LustreData6/home/wuwei/maigo_elastic_ct_20260912'
old=Path('/mnt/sda/wuwei/elastic_production_validation_20260911/matched_full_references')
manifest=json.loads((old/'campaign.json').read_text());report={}
for case in ['RT07575','20022516']:
 entry=next(x for x in manifest['cases'] if x['case']==case)
 src=old/case/'replica_01';shared=root/'shared'/case;shared.mkdir(parents=True,exist_ok=True)
 for name in ['HUtoMaterialSchneider.txt','beam_model.csv']:shutil.copyfile(src/name,shared/name)
 if not (shared/'dicom').exists():shutil.copytree((src/'dicom').resolve(),shared/'dicom')
 with (src/'spots.csv').open() as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 totals=[0]*len(rows)
 for replica in entry['replicas']:
  d=Path(replica['directory']);text=(d/'run.txt').read_text();scale=Decimal(re.search(r'HistoriesScale\s*=\s*([\d.]+)',text).group(1))
  rr=list(csv.DictReader((d/'spots.csv').open()));assert len(rr)==len(rows)
  for i,r in enumerate(rr):
   assert all(r[k]==rows[i][k] for k in fields if k!='weight')
   totals[i]+=int((Decimal(r['weight'])*scale).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
 assert sum(totals)==entry['histories'],(case,sum(totals),entry['histories'])
 shards=[[0]*len(rows) for _ in range(5)];offset=0
 for i,n in enumerate(totals):
  for j in range(5):shards[j][i]=n//5
  for j in range(n%5):shards[(offset+j)%5][i]+=1
  offset=(offset+n%5)%5
 counts=[]
 for j,weights in enumerate(shards):
  run=root/case/f'shard_{j+1:02d}';run.mkdir(parents=True,exist_ok=True)
  with (run/'spots.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r,n in zip(rows,weights):w.writerow(dict(r,weight=str(n)))
  text=(src/'run.txt').read_text();text=re.sub(r'NumberOfThreads\s*=\s*\d+','NumberOfThreads = 56',text);text=re.sub(r'HistoriesScale\s*=\s*[\d.]+','HistoriesScale = 1.0',text);text=re.sub(r'i:Ts/Seed\s*=\s*\d+',f'i:Ts/Seed = {7501+j if case=="RT07575" else 2001+j}',text)
  text=text.replace('"dicom"',f'"{remote}/shared/{case}/dicom"').replace('includeFile = HUtoMaterialSchneider.txt',f'includeFile = {remote}/shared/{case}/HUtoMaterialSchneider.txt').replace('"beam_model.csv"',f'"{remote}/shared/{case}/beam_model.csv"')
  (run/'run.txt').write_text(text);counts.append(sum(weights))
 assert sum(counts)==entry['histories'] and max(counts)-min(counts)<=1
 report[case]={'total_histories':sum(counts),'shard_histories':counts,'partitioning':'Exact aggregate per-spot integer histories preserved, split evenly over five nodes; scale=1'}
(root/'migration_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
