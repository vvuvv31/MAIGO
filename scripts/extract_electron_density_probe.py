#!/usr/bin/env python3
"""Independent native charged parameters at every adjacent-node midpoint."""
import concurrent.futures,json,os,re,subprocess
from pathlib import Path
import numpy as np
from extract_photon_response import sha,save
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'out/electron_standalone_20261003'
def one(task):
 out,spec=task;d=out/f'pair{spec["low"]:02d}_{spec["high"]:02d}';d.mkdir();src=BASE/f'electron_boundary_parameters82_v5/node{spec["low"]:02d}'
 text=(src/'topas.txt').read_text();name=spec['material_name'];pattern=r'(d:Ma/'+re.escape(name)+r'/Density = )[^\n]+'
 text,n=re.subn(pattern,lambda m:m[1]+f'{spec["density"]:.17g} g/cm3',text);assert n==1
 (d/'topas.txt').write_text(text);binary=BASE/'electron_boundary_native_v5/topas'
 env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 with (d/'topas.log').open('x') as log:r=subprocess.run([str(binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0 and 'ELECTRON_BOUNDARY_PROBE=PASS' in (d/'topas.log').read_text()
 opts=dict(l.split('=',1) for l in (d/'boundary_options.txt').read_text().splitlines());assert np.float32(float(opts['density_g_cm3']))==np.float32(spec['density'])
 result=dict(**spec,path=str(d),binary_sha256=sha(binary),input_sha256=sha(d/'topas.txt'),parameter_sha256=sha(d/'boundary_parameters.csv'),hard_rate_sha256=sha(d/'hard_rate_cubics.csv'),options=opts)
 save(d/'report.json',result);return result

def main():
 out=BASE/'density_midpoints57';out.mkdir(exist_ok=False);r=json.loads((BASE/'charged_complete512_v2/manifest.json').read_text())['results'];groups={}
 for x in r:groups.setdefault(x['material_identity']['section_id'],[]).append(x)
 cases=[]
 for sec,rows in sorted(groups.items()):
  rows.sort(key=lambda x:x['material_identity']['runtime_density_g_cm3'])
  for lo,hi in zip(rows,rows[1:]):
   ident=lo['material_identity'];rho=float(np.float32(.5*(ident['runtime_density_g_cm3']+hi['material_identity']['runtime_density_g_cm3'])))
   cases.append(dict(low=lo['node'],high=hi['node'],section=sec,density=rho,material_name=ident['material_name'],composition_sha256=ident['composition']['composition_sha256']))
 assert len(cases)==57;save(out/'inputs.json',dict(cases=cases,generator_sha256=sha(Path(__file__))))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(one,[(out,s) for s in cases]))
 save(out/'report.json',dict(results=results,scope='Actual native intermediate densities, original composition and cuts; no interpolated rates used as native reference'))
if __name__=='__main__':main()
