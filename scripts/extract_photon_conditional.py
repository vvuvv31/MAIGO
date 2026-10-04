#!/usr/bin/env python3
"""Native process-conditional photon outcomes, with explicit support audit."""
import argparse,concurrent.futures,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
from extract_photon_response import ROOT,CARDS,DTYPE,sha,save
BINARY=ROOT/'out/electron_standalone_20261003/photon_conditional_native/topas'
RATES=ROOT/'out/electron_standalone_20261003/photon_bank100k'

def analyze(d,spec,expected_pdg=22):
 r=np.fromfile(d/'conditional.phsp',DTYPE);assert r.nbytes==(d/'conditional.phsp').stat().st_size and np.isin(r['kind'],[0,1]).all()
 zero_child=(r['kind']==1)&(r['remaining']==0)
 for k in DTYPE.names[6:]:assert np.isfinite(r[k][~zero_child] if k=='post' else r[k]).all(),k
 c=r[r['kind']==0];child=r[r['kind']==1];assert np.array_equal(c['event'],np.arange(len(c))) and np.all(c['step']==1) and np.all(c['pdg']==expected_pdg)
 assert np.all(c['energy']>0) and np.all(c['remaining']>=0) and np.all(c['deposit']>=0)
 assert np.max(abs(c['energy']-c['remaining']-c['deposit']-c['children']))<1e-9
 assert np.isin(child['pdg'],[11,22]).all() and np.all(child['remaining']>=0) and np.all(child['event']<len(c))
 assert np.array_equal(child['run'],c['run'][child['event']]) and np.array_equal(child['process'],c['process'][child['event']]) and np.array_equal(child['energy'],c['energy'][child['event']])
 transfer=np.bincount(child['event'],weights=child['remaining'],minlength=len(c));assert np.max(abs(transfer-c['children']))<1e-9
 assert np.max(abs(np.linalg.norm(c['pre'],axis=1)-1),initial=0)<1e-10
 assert np.max(abs(np.linalg.norm(child['post'][child['remaining']>0],axis=1)-1),initial=0)<1e-10
 a=np.atleast_1d(np.genfromtxt(d/'conditional_audit.csv',delimiter=',',names=True));requests=np.loadtxt(d/'requests.txt',ndmin=2)
 assert len(a)==len(requests) and len(c)==int(a['accepted'].sum()) and np.all(a['accepted']==a['requested'])
 assert np.array_equal(a['cell'],requests[:,0]) and np.array_equal(a['process'],requests[:,1]) and np.array_equal(a['energy_MeV'],requests[:,2]) and np.array_equal(a['requested'],requests[:,3])
 for x in a:
  selected=(c['run']==int(x['cell']))&(c['process']==int(x['process']));assert selected.sum()==x['accepted'] and np.all(c['energy'][selected]==x['energy_MeV'])
  assert x['rate_per_mm']>0 and x['max_energy_residual_MeV']<1e-9
 identity=json.loads((CARDS/f'node{spec["node"]:02d}'/'complete.json').read_text())['material_identity'];rho=identity['runtime_density_g_cm3']
 assert np.all(abs(r['density']-rho)<=4*np.finfo(float).eps*rho) and np.all(r['density'].astype(np.float32)==np.float32(rho))
 return dict(collision_samples=len(c),child_births=len(child),zero_energy_children=int(zero_child.sum()),zero_energy_nonfinite_directions=int(np.count_nonzero(~np.isfinite(r['post'][zero_child]).all(axis=1))),requests=len(a),null_collision_count=int(a['null_collisions'].sum()),maximum_residual_MeV=float(a['max_energy_residual_MeV'].max()),material_identity=identity,raw_sha256=sha(d/'conditional.phsp'),audit_sha256=sha(d/'conditional_audit.csv'),raw_bytes=r.nbytes,scope='Independent native GPIL/PostStepDoIt conditional final states, not analog history or flight distribution; all children recorded. Undefined native directions are permitted only for exactly zero-energy children, which carry zero probability in the energy-weighted estimator. Null collisions consume RNG and are counted, not emitted as real interactions.')

def one(task):
 d,spec=task;d.mkdir(parents=True,exist_ok=False)
 rate=np.loadtxt(RATES/f'node{spec["node"]:02d}/photon_rates.csv',delimiter=',',skiprows=1)
 energies=np.geomspace(1e-5,.65,257) if spec['mono_energy'] is None else np.array([spec['mono_energy']])
 anchors=energies.astype(np.float32).astype(float)
 edges=np.r_[anchors[0],np.sqrt(anchors[:-1]*anchors[1:]),.65] if spec['mono_energy'] is None else None
 requests=[]
 for cell,e in enumerate(energies):
  for proc in range(3):
   requested_energy=e
   if np.interp(e,rate[:,0],rate[:,proc+1])<=0:
    if edges is None:continue
    # A process threshold can lie inside a cell whose centre has zero rate.
    # Use an actually evaluated native rate node inside the same energy cell;
    # never borrow a different process or move the runtime particle's energy.
    supported=rate[(rate[:,0]>=edges[cell])&(rate[:,0]<edges[cell+1])&(rate[:,proc+1]>0),0]
    if not len(supported):continue
    requested_energy=float(supported[len(supported)//2])
   requests.append((cell,proc,requested_energy,spec['samples']))
 (d/'requests.txt').write_text(''.join(f'{i} {p} {e:.17g} {n}\n' for i,p,e,n in requests))
 text=(RATES/f'node{spec["node"]:02d}/topas.txt').read_text();text='\n'.join(l for l in text.splitlines() if 'BeamEnergySpectrum' not in l)+'\n'
 text=re.sub(r'i:Ts/Seed = .*',f'i:Ts/Seed = {spec["seed"]}',text)
 text=re.sub(r'i:So/Beam/NumberOfHistoriesInRun = .*','i:So/Beam/NumberOfHistoriesInRun = 1',text)
 text=re.sub(r'd:So/Beam/BeamEnergy = .*','d:So/Beam/BeamEnergy = 0.00001 MeV',text)
 text=text.replace('"PhotonResponseNtuple"','"PhotonConditionalDump"')
 (d/'topas.txt').write_text(text)
 env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 env.pop('MAIGO_PHOTON_XS',None)
 start=time.perf_counter()
 with (d/'topas.log').open('x') as log:r=subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
 result=dict(spec,binary_sha256=sha(BINARY),input_sha256=sha(d/'topas.txt'),requests_sha256=sha(d/'requests.txt'),rate_source_sha256=sha(RATES/f'node{spec["node"]:02d}/photon_rates.csv'),exit_code=r.returncode,seconds=time.perf_counter()-start)
 save(d/'run.json',result)
 assert r.returncode==0 and 'PHOTON_CONDITIONAL_DUMP=PASS' in (d/'topas.log').read_text(),d
 result['audit']=analyze(d,spec);save(d/'report.json',result);print(d.name,json.dumps(result['audit']),flush=True);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',type=int,nargs='+',default=[57,12,32]);p.add_argument('--samples',type=int,default=512);p.add_argument('--workers',type=int,default=3);p.add_argument('--mono-energy',type=float);a=p.parse_args()
 assert a.samples>0 and a.workers>0 and all(0<=n<82 for n in a.nodes) and len(set(a.nodes))==len(a.nodes)
 assert a.mono_energy is None or 1e-5<=a.mono_energy<=.65
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 tasks=[(out/f'node{n:02d}',dict(node=n,samples=a.samples,seed=114600001+n*1009,mono_energy=a.mono_energy)) for n in a.nodes]
 save(out/'inputs.json',dict(script_sha256=sha(Path(__file__)),tasks=[s for _,s in tasks]))
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
 save(out/'report.json',dict(status='NATIVE_PROCESS_CONDITIONAL_DATA_NOT_CT_ACCEPTANCE',results=results))
if __name__=='__main__':main()
