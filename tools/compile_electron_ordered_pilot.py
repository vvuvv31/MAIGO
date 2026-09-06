"""Independent bulk full-family ordered paths; OFFLINE hypothesis only.

Energy-weighted sampling within 5MeV/u parent bins; no interface fit. Includes
all deposit tracks and explicitly accounts for finite-box neutral escape.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
from compare_electron_response_geometries import load_v2_binary,build_roots
from electron_ordered_paths import OrderedPaths
from analyze_longitudinal_holdout import sha

def compile_source(path,hu,samples,seed):
    d=load_v2_binary(path);tracks,root=build_roots(d);g=OrderedPaths(d)
    config=(path.parent/'run.txt').read_text()
    def mm(key):return float(re.search(r'^d:Ge/ZResponse/'+key+r' = ([\d.]+) mm$',config,re.M)[1])
    bounds=np.array([[-mm('HLX'),mm('HLX')],[-mm('HLY'),mm('HLY')],[mm('TransZ')-mm('HLZ'),mm('TransZ')+mm('HLZ')]])
    inset=100 if hu==-1000 else 5;rho=.01131606474518776 if hu==-1000 else 1.0787997245788574
    first={};last={};parents={}
    for i,r in enumerate(d):
        k=tuple(r[:3].astype(int));first.setdefault(k,i);last[k]=i
        if r[3]==0 and np.all(r[10:13]>=bounds[:,0]+inset) and np.all(r[10:13]<=bounds[:,1]-inset) and np.all(r[13:16]>=bounds[:,0]+inset) and np.all(r[13:16]<=bounds[:,1]-inset):parents[k+(int(r[5]),)]=i
    p=d[list(parents.values())];pb=np.minimum((p[:,17]/60).astype(int),36)
    loss=np.bincount(pb,weights=(p[:,17]-p[:,18])*p[:,19],minlength=37)
    local=np.bincount(pb,weights=p[:,16]*p[:,19],minlength=37)
    terminal=np.bincount(pb,weights=np.where((pb==0)&(p[:,18]==0),p[:,16]-p[:,17],0)*p[:,19],minlength=37)
    rb={};birth=np.zeros(37);escape=np.zeros(37)
    for k in tracks:
        if root(k)!=k:continue
        r=d[first[k]];pk=(k[0],k[1],int(r[3]),int(r[29]))
        if pk in parents:
            if r[22]!=1:raise ValueError('Unbound root')
            rb[k]=min(int(d[parents[pk],17]/60),36);birth[rb[k]]+=r[9]*r[19]
    bins=np.full(len(d),-1)
    for i,r in enumerate(d):
        rk=root(tuple(r[:3].astype(int)))
        if rk in rb:bins[i]=rb[rk]
    for k in tracks:
        rk=root(k)
        if rk not in rb:continue
        r=d[last[k]]
        if r[18]>1e-9:
            if r[4]!=22 or not np.any(np.minimum(abs(r[13:16]-bounds[:,0]),abs(r[13:16]-bounds[:,1]))<1e-5):raise ValueError('Unresolved family escape')
        escape[rb[rk]]+=r[18]*r[19]
    sel=(bins>=0)&(d[:,16]>0);dep=np.bincount(bins[sel],weights=d[sel,16]*d[sel,19],minlength=37)
    for value in (local+birth-loss-terminal,local+dep+escape-loss-terminal):
        if np.max(abs(value)/np.maximum(loss,1))>1e-7:raise ValueError('Full family energy closure')
    if np.any((terminal!=0)&((birth!=0)|(dep!=0)|(escape!=0))):raise ValueError('Nonlocal terminal residual')
    rng=np.random.default_rng(seed);vectors=[];offsets=[0];sample_bins=[];weights=[];identity=0.;descendants=0
    for b in range(37):
        ids=np.flatnonzero(sel&(bins==b))
        if not len(ids):continue
        chosen=rng.choice(ids,size=samples,p=d[ids,16]*d[ids,19]/dep[b])
        for i in chosen:
            rk=root(tuple(d[i,:3].astype(int)));r=d[first[rk]]
            fraction=1. if d[i,4]==22 else rng.random()
            path_vectors=g.path(int(i),fraction)
            actual=d[i,10:13]+fraction*(d[i,13:16]-d[i,10:13])-r[6:9]
            error=float(np.linalg.norm(path_vectors.sum(axis=0)-actual));identity=max(identity,error)
            if error>1e-6:raise ValueError('Ordered endpoint identity')
            # Rotate the ENTIRE path into the primary local frame, keeping its
            # internal azimuth correlations. A single random azimuth is applied
            # to the full path at projection, never independently per segment.
            z=r[24:27];z=z/np.linalg.norm(z);helper=np.array([1.,0,0]) if abs(z[0])<.9 else np.array([0.,1,0])
            x=np.cross(helper,z);x/=np.linalg.norm(x);y=np.cross(z,x)
            v=path_vectors@np.stack((x,y,z),axis=1)*rho/10
            vectors.extend(v);offsets.append(len(vectors));sample_bins.append(b);weights.append(dep[b]/samples)
            descendants+=tuple(d[i,:3].astype(int))!=rk
    return dict(vectors=np.array(vectors),offsets=np.array(offsets,dtype=np.int64),bins=np.array(sample_bins),
        weights=np.array(weights),loss=loss,local=local,escape=escape,terminal=terminal),dict(
        hu=hu,samples_per_active_bin=samples,seed=seed,sampled_descendant_deposits=int(descendants),
        maximum_endpoint_residual_mm=identity,maximum_birth_residual_mm=g.maximum_birth_residual_mm,
        full_family_energy_residual_MeV=float(np.max(abs(local+dep+escape-loss-terminal))),raw_sha256=sha(path))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metadata',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--indices',nargs='+',type=int,required=True);p.add_argument('--samples',type=int,default=256)
    a=p.parse_args();m=json.loads(a.metadata.read_text());a.out.mkdir(exist_ok=False);report={}
    if a.samples<1 or len(set(a.indices))!=len(a.indices):raise ValueError('Invalid sampling request')
    for i in a.indices:
        s=m['sources'][i];path=Path(s['path'])
        if sha(path)!=s['sha256']:raise ValueError('Raw SHA mismatch')
        campaign=path.parents[2]/'manifest.json'
        campaign_inputs=json.loads(campaign.read_text())['inputs']
        config=path.parent/'run.txt'
        if sha(config)!=campaign_inputs[str(config)]:raise ValueError('Changed source configuration')
        arrays,info=compile_source(path,s['hu'],a.samples,914730+i)
        output=a.out/f'source_{i:03d}.npz';np.savez(output,**arrays)
        info.update(raw_path=str(path),payload_sha256=sha(output),config_sha256=sha(config),
                    header_sha256=sha(path.with_suffix('.header')),campaign_manifest_sha256=sha(campaign));report[output.name]=info
        print(i,info,flush=True)
    with (a.out/'manifest.json').open('x') as f:json.dump(dict(status='OFFLINE_ORDERED_PATH_PILOT',
        compiler_sha256=sha(Path(__file__)),path_reconstructor_sha256=sha(Path(__file__).with_name('electron_ordered_paths.py')),
        source_metadata_sha256=sha(a.metadata),sources=report,
        limitations=['source-material electron scattering retained across interface','finite-box photon escape retained local',
                     'Monte Carlo compression requires sample/seed convergence; not a production package']),f,indent=2)
