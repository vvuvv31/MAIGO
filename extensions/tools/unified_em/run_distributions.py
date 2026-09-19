from pathlib import Path
import argparse,subprocess,json,os,hashlib
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--topas',type=Path,required=True)
args=parser.parse_args();R=args.campaign.resolve();exe=args.topas.resolve()
for z,a in json.loads((R/'extraction_manifest.json').read_text())['species_za']:
 d=R/f'ion_{z}_{a}'/'distribution_probe_v2';d.mkdir(exist_ok=True)
 with (d/'topas.log').open('w') as log:
  p=subprocess.run([str(exe),str(d.parent/'topas.txt')],cwd=d,env=dict(os.environ,EM_DISTRIBUTION_PROBE='1'),stdout=log,stderr=subprocess.STDOUT)
 count=len(list(d.glob('*/distribution_probe.csv')))
 ok=p.returncode==0 and count==4
 (d/'status.json').write_text(json.dumps({'complete':ok,'material_density_count':count,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2))
 print(z,a,ok,count,flush=True)
 if not ok:raise RuntimeError('Distribution probe failed')
