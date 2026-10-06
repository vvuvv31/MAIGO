#!/usr/bin/env python3
"""Independent native conditional collisions for the eight failed compositions."""
import argparse,concurrent.futures,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes,raw_rate
from extract_photon_response import DTYPE
from validate_electron_response26 import weights,masses
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    nodes=collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged');tasks=[];cards={}
    for card in (ROOT/'out/electron_standalone_20261003/photon_conditional_bank512').glob('node*/topas.txt'):
        text=card.read_text();m=re.search(r'd:Ma/[^/]+/Density = ([\d.eE+-]+) g/cm3',text)
        if m:cards[float(m[1])]=text
    for section in [0,1,15,16,17,18,19,23]:
        group=sorted((n for n in nodes if n['section']==section),key=lambda n:n['density'])
        for j,n in enumerate(group):
            requests=[]
            # Sample at the source outcome bins' geometric anchors, separating
            # the continuous-density error from the existing energy-bin law.
            for desired in [.08,.1,.2,.4,.6]:
                bin=min(int(np.searchsorted(n['edges'],desired)-1),len(n['edges'])-2)
                energy=float(np.sqrt(n['edges'][bin]*n['edges'][bin+1]))
                for process in range(2):
                    if raw_rate(n['rates'][process],energy)>0:requests.append((bin,process,energy,a.samples))
            for kind in ['charged','photon']:
                d=a.out/f'{kind}_c{section:02d}_cut{j}';d.mkdir()
                if kind=='photon':
                    requests=[]
                    ph=next(x for x in collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/photon','photon') if (x['section'],x['density'])==(section,n['density']))
                    for desired in [.003,.01,.08,.2,.4]:
                        b=min(int(np.searchsorted(ph['edges'],desired)-1),len(ph['edges'])-2);e=float(np.sqrt(ph['edges'][b]*ph['edges'][b+1]))
                        for proc in range(3):
                            if np.interp(e,ph['photon_rates'][:,0],ph['photon_rates'][:,1+proc])>0:requests.append((b,proc,e,a.samples))
                    node=ph;binary=ROOT/'out/electron_standalone_20261003/photon_conditional_native/topas'
                    quantity='PhotonConditionalDump'
                else:
                    # Recreate charged requests because the photon list above
                    # is intentionally independent.
                    requests=[]
                    for desired in [.08,.1,.2,.4,.6]:
                        b=min(int(np.searchsorted(n['edges'],desired)-1),len(n['edges'])-2);e=float(np.sqrt(n['edges'][b]*n['edges'][b+1]))
                        for proc in range(2):
                            if raw_rate(n['rates'][proc],e)>0:requests.append((b,proc,e,a.samples))
                    node=n;binary=ROOT/'out/electron_standalone_20261003/electron_collision_native/topas';quantity='ElectronCollisionDump'
                (d/'topas.txt').write_text(cards[n['density']].replace('PhotonConditionalDump',quantity))
                (d/'requests.txt').write_text(''.join(f'{b} {proc} {e:.17g} {count}\n' for b,proc,e,count in requests))
                tasks.append((d,kind,node,binary,requests))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    def run(task):
        d,kind,node,binary,requests=task;start=time.monotonic()
        with (d/'topas.log').open('x') as log:r=subprocess.run([str(binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode or ('ELECTRON_COLLISION_DUMP=PASS' if kind=='charged' else 'PHOTON_CONDITIONAL_DUMP=PASS') not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        raw=np.fromfile(d/'conditional.phsp',DTYPE);events=raw[raw['kind']==0];children=raw[raw['kind']==1]
        if np.max(abs(events['energy']-events['remaining']-events['deposit']-events['children']))>1e-9:raise ValueError('energy closure')
        model=np.load(a.candidate/kind/f'composition{node["section"]:02d}.npz');domain=model['density_domain'].astype(float);degree=int(model['density_degree'][0]);rho=node['density']
        x=0 if domain[0]==domain[1] else (rho-domain[0])/(domain[1]-domain[0]);bw=weights(degree,x,domain)
        audit=np.atleast_1d(np.genfromtxt(d/'conditional_audit.csv',delimiter=',',names=True));rows=[]
        for b,proc,e,count in requests:
            primary=events[(events['run']==b)&(events['process']==proc)];secondary=children[(children['run']==b)&(children['process']==proc)]
            if len(primary)!=count:raise ValueError('count')
            # Energy-weighted walker chooses deposit, mother, or child using
            # branch kinetic energy. All branches retain their native angles.
            total=primary['energy'].sum();energy=(np.sum(primary['remaining']**2)+np.sum(secondary['remaining']**2)+np.sum(primary['deposit']*primary['energy']))/total
            active_mother=primary[primary['remaining']>0];active_children=secondary[secondary['remaining']>0]
            # Local deposits use canonical cosine0 in the packed walker law.
            # Directions for exactly zero-energy native children are undefined.
            cosine=(np.sum(active_mother['remaining']*active_mother['post'][:,2])+np.sum(active_children['remaining']*active_children['post'][:,2]))/total
            prefix=f'ch{b*(2 if kind=="charged" else 3)+proc:04d}_';pmf=masses(model[prefix+'cdf']);ce=model[prefix+'energy'].astype(float)*e;cm=model[prefix+'cosine'].astype(float)
            if kind=='charged':
                edges=model[f'rate{proc}_edges'].astype(float);values=model[f'rate{proc}_values'].astype(float)
                pieces=np.concatenate([np.broadcast_to(edges[:-1],values.shape[:2])[...,None],np.broadcast_to(edges[1:],values.shape[:2])[...,None],values],axis=-1)
                control=np.array([max(0.,float(raw_rate(row,e))) for row in pieces]);mix=bw*control;mix/=mix.sum()
                rate=float(bw@control*rho)
            else:mix=bw;rate=float(np.interp(e,model['mass_rates'][:,0],model['mass_rates'][:,1+proc])*rho)
            weighted=mix[:,None]*pmf
            nominal=audit[(audit['cell']==b)&(audit['process']==proc)]['rate_per_mm'][0]
            newe=float(np.sum(weighted*ce));newm=float(np.sum(weighted*cm))
            rows.append(dict(kind=kind,section=node['section'],density=rho,energy=e,process=proc,native_energy=energy,candidate_energy=newe,mean_energy_relative=newe/energy-1,native_cosine=cosine,candidate_cosine=newm,mean_cosine_absolute=newm-cosine,native_rate=float(nominal),candidate_rate=rate,rate_relative=rate/nominal-1))
        result=dict(directory=str(d),seconds=time.monotonic()-start,events=len(events),rows=rows,status='NATIVE_COLLISION_ENERGY_AUDIT_PASS_NOT_GPU_DOSE_ACCEPTANCE')
        (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,tasks))
    (a.out/'result.json').write_text(json.dumps(dict(results=results,status='INDEPENDENT_NATIVE_COMPONENT_COMPARISON'),indent=2)+'\n')
if __name__=='__main__':main()
