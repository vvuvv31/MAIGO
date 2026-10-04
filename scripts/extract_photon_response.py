#!/usr/bin/env python3
"""Fresh photon rates and collision final states, independent of replay banks."""
import argparse,concurrent.futures,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/electron_response_model'))
from prototype import sha,save
from macro_model import DTYPE as FULL
BINARY=ROOT/'out/electron_standalone_20261003/photon_native/topas'
CARDS=ROOT/'out/electron_standalone_20261003/birth_bank64'
DTYPE=np.dtype([(k,'<i4') for k in ['run','event','step','kind','pdg','process']]+[(k,'<f8') for k in ['energy','remaining','deposit','children']]+[('pre','<f8',3),('post','<f8',3),('length','<f8'),('density','<f8')])
assert DTYPE.itemsize==120


def analyze(d,spec):
    r=np.fromfile(d/'photons.phsp',DTYPE);assert r.nbytes==(d/'photons.phsp').stat().st_size and np.isin(r['kind'],[0,1,2]).all()
    for k in DTYPE.names[6:]:assert np.isfinite(r[k]).all(),k
    events=r[r['kind']==2];collisions=r[r['kind']==0];children=r[r['kind']==1]
    assert len(events)==spec['histories'] and np.array_equal(np.sort(events['event']),np.arange(spec['histories']))
    assert np.all(events['energy']>0) and np.all(events['remaining']>=0)
    assert np.max(abs(events['energy']-events['remaining']-events['deposit']))<1e-8 and events['children'].max()<1e-9
    assert np.isin(collisions['process'],[0,1,2]).all() and np.all(collisions['pdg']==22)
    assert np.max(abs(collisions['energy']-collisions['remaining']-collisions['deposit']-collisions['children']),initial=0)<1e-9
    assert np.isin(children['pdg'],[11,22]).all() and np.all(children['remaining']>=0)
    key=lambda x:(int(x['run']),int(x['event']),int(x['step']))
    rowmap={key(x):i for i,x in enumerate(collisions)};assert len(rowmap)==len(collisions)
    parent=np.array([rowmap[key(x)] for x in children],dtype=np.int64)
    born=np.bincount(parent,weights=children['remaining'],minlength=len(collisions))
    assert np.max(abs(born-collisions['children']),initial=0)<1e-9
    assert np.max(abs(np.linalg.norm(collisions['pre'],axis=1)-1),initial=0)<1e-10
    assert np.max(abs(np.linalg.norm(children['post'],axis=1)-1),initial=0)<1e-10
    identity=json.loads((CARDS/f'node{spec["node"]:02d}'/'complete.json').read_text())['material_identity'];rho=identity['runtime_density_g_cm3']
    assert np.all(abs(r['density']-rho)<=4*np.finfo(float).eps*rho) and np.all(r['density'].astype(np.float32)==np.float32(rho))
    report=dict(collisions=len(collisions),child_births=len(children),source_histories=len(events),source_escape_count=int(np.count_nonzero(events['remaining'])),source_escape_energy_MeV=float(events['remaining'].sum()),source_incident_MeV=float(events['energy'].sum()),raw_bytes=r.nbytes,raw_sha256=sha(d/'photons.phsp'),process_counts={name:int(np.count_nonzero(collisions['process']==i)) for i,name in enumerate(['phot','compt','Rayl'])},max_step_energy_residual_MeV=float(events['children'].max()),max_event_energy_residual_MeV=float(np.max(abs(events['energy']-events['remaining']-events['deposit']))),material_identity=identity)
    ratefile=d/'photon_rates.csv'
    if ratefile.exists():
        rate=np.loadtxt(ratefile,delimiter=',',skiprows=1);assert np.isfinite(rate).all() and np.all(rate[:,1:]>=0) and np.all(np.diff(rate[:,0])>0) and np.all(rate[:,-1]==0)
        report['rates']=dict(rows=len(rate),sha256=sha(ratefile),minimum_energy_MeV=float(rate[0,0]),maximum_energy_MeV=float(rate[-1,0]))
        first=collisions[collisions['step']==1]
        # Exactly monoenergetic first flights provide an independent extinction
        # and process-frequency check, separate from collision-final-state fits.
        if spec['mono_energy'] is not None and len(first)==spec['histories']:
            energy=float(first['energy'][0]);coeff=np.array([np.interp(energy,rate[:,0],rate[:,k]) for k in range(1,4)])
            total=coeff.sum();optical=first['length']*total
            report['first_flight']=dict(actual_energy_MeV=energy,rate_per_mm=total,mean_optical_depth=float(optical.mean()),SE_mean_optical_depth=float(optical.std(ddof=1)/np.sqrt(len(optical))),process_probability=(coeff/total).tolist(),observed_probability=[float(np.mean(first['process']==i)) for i in range(3)])
    if spec['verify_full']:
        full=np.fromfile(d/'full.phsp',FULL);primary=full[(full['parent']==0)&np.isin(full['step'],collisions['step'])]
        pmap={key(x):x for x in primary}
        maxdirection=0
        for x in collisions:
            y=pmap[key(x)]
            assert x['energy']==y['energy'] and x['remaining']==y['remaining'] and x['deposit']==y['dep'] and x['length']==y['length']
            maxdirection=max(maxdirection,float(abs(x['pre']-y['direction']).max()),float(abs(x['post']-y['post_direction']).max()))
        fs=full[(full['parent']==1)&(full['step']==1)]
        assert len(fs)==len(children)
        # Row order of children can differ from native track scheduling.
        observed=sorted((int(x['event']),int(x['step']),int(x['pdg']),float(x['remaining'])) for x in children)
        expected=sorted((int(x['event']),int(x['parent_step']),int(x['pdg']),float(x['birth'])) for x in fs)
        assert observed==expected and maxdirection==0
        report['full_record_audit']=dict(all_primary_collision_states_exact=True,all_child_birth_energies_exact=True,raw_sha256=sha(d/'full.phsp'))
    return report


def one(task):
    d,spec=task;d.mkdir(parents=True,exist_ok=False)
    card=CARDS/f'node{spec["node"]:02d}'/'topas.txt';text=card.read_text()
    text=re.sub(r'i:Ts/NumberOfThreads = .*','i:Ts/NumberOfThreads = 1',text)
    text=re.sub(r'i:Ts/Seed = .*',f'i:Ts/Seed = {spec["seed"]}',text)
    text=re.sub(r'sv:Ph/Default/Modules = .*','sv:Ph/Default/Modules = 1 "g4em-standard_opt4"',text)
    text=re.sub(r'd:Ge/World/HL([XYZ]) = .*',r'd:Ge/World/HL\1 = 20000000 mm',text)
    text=re.sub(r'd:Ge/ROI/HL([XYZ]) = .*',r'd:Ge/ROI/HL\1 = 10000000 mm',text)
    text=re.sub(r's:So/Beam/BeamParticle = .*','s:So/Beam/BeamParticle = "gamma"',text)
    text=re.sub(r'i:So/Beam/NumberOfHistoriesInRun = .*',f'i:So/Beam/NumberOfHistoriesInRun = {spec["histories"]}',text)
    text=re.sub(r's:Sc/Steps/Quantity = .*','s:Sc/Steps/Quantity = "PhotonResponseNtuple"',text)
    text=re.sub(r's:Sc/Steps/OutputFile = .*','s:Sc/Steps/OutputFile = "photons"',text)
    text=re.sub(r'd:Ge/ROI/MaxStepSize = .*','d:Ge/ROI/MaxStepSize = 0.05 mm',text)
    if spec['mono_energy'] is not None:text=re.sub(r'd:So/Beam/BeamEnergy = .*',f'd:So/Beam/BeamEnergy = {spec["mono_energy"]:.17g} MeV',text)
    else:
        energies=np.geomspace(1e-5,.65,257)
        text+='\ns:So/Beam/BeamEnergySpectrumType = "Discrete"\ndv:So/Beam/BeamEnergySpectrumValues = 257 '+' '.join(format(e,'.17g') for e in energies)+' MeV\nuv:So/Beam/BeamEnergySpectrumWeights = 257 '+' '.join([format(1/257,'.17g')]*257)+'\n'
    if spec['verify_full']:
        text+='\ns:Sc/Full/Quantity = "PrimaryElectronTransportNtuple"\ns:Sc/Full/Component = "ROI"\ns:Sc/Full/OutputType = "binary"\ns:Sc/Full/OutputFile = "full"\ns:Sc/Full/IfOutputFileAlreadyExists = "Exit"\nb:Sc/Full/RecordTransportState = "True"\n'
    (d/'topas.txt').write_text(text)
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    if spec['rates']:env['MAIGO_PHOTON_XS']='1'
    else:env.pop('MAIGO_PHOTON_XS',None)
    start=time.perf_counter()
    with (d/'topas.log').open('x') as log:r=subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(spec,binary_sha256=sha(BINARY),input_sha256=sha(d/'topas.txt'),exit_code=r.returncode,seconds=time.perf_counter()-start)
    save(d/'run.json',result)
    log=(d/'topas.log').read_text();counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',log)
    assert r.returncode==0 and counts and int(counts[-1])==spec['histories'],d
    assert sum(map(float,re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',log)))==0
    result['audit']=analyze(d,spec);save(d/'report.json',result);print(d.name,json.dumps(result['audit']),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',type=int,nargs='+',default=[57,12,32]);p.add_argument('--histories',type=int,default=100000);p.add_argument('--workers',type=int,default=3);p.add_argument('--mono-energy',type=float);p.add_argument('--verify-full',action='store_true');p.add_argument('--no-rates',action='store_true');a=p.parse_args()
    assert a.histories>0 and a.workers>0 and all(0<=n<82 for n in a.nodes) and len(set(a.nodes))==len(a.nodes)
    assert a.mono_energy is None or 1e-5<=a.mono_energy<=.65
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    tasks=[(out/f'node{n:02d}',dict(node=n,histories=a.histories,seed=114300001+n*1009,mono_energy=a.mono_energy,verify_full=a.verify_full,rates=not a.no_rates)) for n in a.nodes]
    save(out/'inputs.json',dict(script_sha256=sha(Path(__file__)),tasks=[s for _,s in tasks],scope='Native opt4, original electron GS; primary photon collision states/children and attenuation; no legacy trajectory data'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    save(out/'report.json',dict(status='AUDITED_NATIVE_PHOTON_DATA_NOT_GPU_ACCEPTANCE',results=results))
if __name__=='__main__':main()
