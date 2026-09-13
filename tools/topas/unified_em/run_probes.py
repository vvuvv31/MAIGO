from pathlib import Path
import subprocess,json,os,hashlib
R=Path(__file__).resolve().parent;exe=R/'build/topas'
for z,a in json.loads((R/'extraction_manifest.json').read_text())['species_za']:
 d=R/f'ion_{z}_{a}'/'mean_probe_v1';d.mkdir(exist_ok=True)
 with (d/'topas.log').open('w') as log:
  p=subprocess.run([str(exe),str(d.parent/'topas.txt')],cwd=d,env=dict(os.environ,EM_FINITE_PROBE='1'),stdout=log,stderr=subprocess.STDOUT)
 count=len(list(d.glob('*/mean_probe.csv')))
 ok=p.returncode==0 and count==len(json.loads((R/'extraction_manifest.json').read_text())['materials'])
 (d/'status.json').write_text(json.dumps({'complete':ok,'material_density_count':count,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2))
 print(z,a,ok,count,flush=True)
 if not ok:raise RuntimeError('Mean probe failed')
