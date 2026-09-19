"""Sequential single-thread table export; one seed track per ion, killed after export."""
from pathlib import Path
import argparse,subprocess,json,hashlib,time,os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--topas',type=Path,required=True)
args=parser.parse_args();R=args.campaign.resolve();exe=args.topas.resolve()
sha=hashlib.sha256(exe.read_bytes()).hexdigest()
for z,a in json.loads((R/'extraction_manifest.json').read_text())['species_za']:
 base=R/f'ion_{z}_{a}';d=base/'dump_v5';d.mkdir(exist_ok=True);status=d/'status.json'
 if status.exists():
  old=json.loads(status.read_text())
  if old['complete'] and old['binary_sha256']==sha:continue
  raise RuntimeError('Refuse to overwrite failed or different-binary extraction')
 start=time.time()
 with (d/'topas.log').open('w') as log:p=subprocess.run([str(exe),str(base/'topas.txt')],cwd=d,stdout=log,stderr=subprocess.STDOUT)
 catalog=d/'catalog.csv';expected=len(json.loads((R/'extraction_manifest.json').read_text())['materials'])
 count=len(catalog.read_text().splitlines())-1 if catalog.exists() else 0
 ok=p.returncode==0 and count==expected and 'Finalization:' in (d/'topas.log').read_text()
 status.write_text(json.dumps({'complete':ok,'binary_sha256':sha,'material_density_count':count,'elapsed_s':time.time()-start,'returncode':p.returncode},indent=2))
 print(z,a,ok,count,time.time()-start,flush=True)
 if not ok:raise RuntimeError('Incomplete material table extraction')
