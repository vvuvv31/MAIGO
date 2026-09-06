"""Energy-conditioned independent bulk response. Diagnostic output, never production.

5 MeV/u parent PRE-step bins are explicit intervals, not nearest-node aliases.
All probabilities use primary energy loss denominators. Preserve signed joint
radial/longitudinal mass displacement; expose neutral/finite-box escape.
"""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from compare_electron_response_geometries import load_v2_binary,build_roots
from analyze_longitudinal_holdout import sha

NENERGY=37
def extract(path,hu):
    d=load_v2_binary(path)
    if set(d[:,1].astype(int))!=set(range(4)):raise ValueError('Incomplete sample')
    if len(np.unique(d[:,[0,1,2,5]],axis=0))!=len(d):raise ValueError('Duplicate steps')
    config=(path.parent/'run.txt').read_text()
    def mm(key):
        m=re.search(r'^d:Ge/ZResponse/'+key+r' = ([\d.]+) mm$',config,re.M)
        if not m:raise ValueError('Unknown response geometry')
        return float(m[1])
    hx,hy,hz,cz=mm('HLX'),mm('HLY'),mm('HLZ'),mm('TransZ')
    bounds=np.array([[-hx,hx],[-hy,hy],[cz-hz,cz+hz]])
    inset=100. if hu==-1000 else 5.
    tracks,root=build_roots(d);first={};last={};parents={}
    for i,r in enumerate(d):
        k=tuple(r[:3].astype(int));first.setdefault(k,i);last[k]=i
        # Select PARENT steps by a fixed bulk inset, before looking at any
        # child's range or escape. Never select only contained electron roots.
        if r[3]==0 and np.all(r[10:13]>=bounds[:,0]+inset) and np.all(r[10:13]<=bounds[:,1]-inset) and np.all(r[13:16]>=bounds[:,0]+inset) and np.all(r[13:16]<=bounds[:,1]-inset):
            parents[k+(int(r[5]),)]=i
    p=d[list(parents.values())];pe=p[:,17]/12
    if pe.max()>185.000001 or pe.min()<0:raise ValueError('Parent domain')
    pb=np.minimum((pe/5).astype(int),NENERGY-1)
    loss=np.bincount(pb,weights=(p[:,17]-p[:,18])*p[:,19],minlength=NENERGY)
    local=np.bincount(pb,weights=p[:,16]*p[:,19],minlength=NENERGY)
    terminal=np.bincount(pb,weights=np.where((pb==0)&(p[:,18]==0),
        (p[:,16]-(p[:,17]-p[:,18]))*p[:,19],0.),minlength=NENERGY)
    minimum=np.full(NENERGY,np.inf);maximum=np.zeros(NENERGY)
    np.minimum.at(minimum,pb,pe);np.maximum.at(maximum,pb,pe)
    rb={};birth=np.zeros(NENERGY);escape=np.zeros(NENERGY)
    for k in tracks:
        if root(k)!=k:continue
        r=d[first[k]]
        if r[22]!=1:raise ValueError('Unbound root')
        pk=(k[0],k[1],int(r[3]),int(r[29]))
        if pk not in parents:continue
        pi=parents[pk]
        e=d[pi,17]/12;rb[k]=min(int(e/5),NENERGY-1)
        birth[rb[k]]+=r[9]*r[19]
    if np.max(abs(local+birth-loss-terminal))>1e-7*max(1.,loss.sum()):raise ValueError('Birth closure')
    for k in tracks:
        rk=root(k)
        if rk in rb:
            r=d[last[k]]
            if r[18]>1e-9:
                if not np.any(np.minimum(abs(r[13:16]-bounds[:,0]),abs(r[13:16]-bounds[:,1]))<1e-5):
                    raise ValueError('Positive terminal energy away from boundary')
                if r[4]!=22:raise ValueError('Bulk parent cohort has charged escape; enlarge geometry')
            escape[rb[rk]]+=r[18]*r[19]
    rows=[];origins=[];directions=[];bins=[]
    for r in d:
        rk=root(tuple(r[:3].astype(int)))
        if rk not in rb or r[16]<=0:continue
        b=d[first[rk]];rows.append(r);origins.append(b[6:9]);directions.append(b[24:27]);bins.append(rb[rk])
    rows=np.array(rows);origins=np.array(origins);directions=np.array(directions);bins=np.array(bins)
    dep=np.bincount(bins,weights=rows[:,16]*rows[:,19],minlength=NENERGY)
    if np.max(abs(local+dep+escape-loss-terminal))>1e-7*max(1.,loss.sum()):raise ValueError('Family closure')
    if abs(terminal[0])>0 and (birth[0]!=0 or dep[0]!=0 or escape[0]!=0):
        raise ValueError('Terminal-local excess overlaps electron redistribution')
    rho=.01131606474518776 if hu==-1000 else 1.0787997245788574
    if np.max(abs(rows[:,20]/rho-1))>1e-4:raise ValueError('Mixed material')
    start=rows[:,10:13]-origins;end=rows[:,13:16]-origins
    dz0=np.sum(start*directions,axis=1)*rho/10;dz1=np.sum(end*directions,axis=1)*rho/10
    pitch=.00025;counts=np.maximum(1,np.ceil(abs(dz1-dz0)/pitch).astype(int));photons=rows[:,4]==22;counts[photons]=1
    if counts.max()>4096:raise ValueError('Unresolved charged step')
    ids=np.repeat(np.arange(len(rows)),counts)
    sub=(np.arange(len(ids))-np.repeat(np.cumsum(counts)-counts,counts)+.5)/counts[ids]
    sub[photons[ids]]=1.
    point=start[ids]+sub[:,None]*(end-start)[ids]
    long=np.sum(point*directions[ids],axis=1)
    radial=np.linalg.norm(point-long[:,None]*directions[ids],axis=1)*rho/10;long*=rho/10
    weight=rows[ids,16]*rows[ids,19]/counts[ids]
    key=np.column_stack((bins[ids],np.floor(long/pitch).astype(int),np.floor(radial/pitch).astype(int)))
    unique,inverse=np.unique(key,axis=0,return_inverse=True)
    energy=np.bincount(inverse,weights=weight)
    moments=np.column_stack((energy,np.bincount(inverse,weights=weight*long),np.bincount(inverse,weights=weight*radial)))
    return dict(loss=loss,local=local,escape=escape,terminal_local_excess=terminal,minimum=minimum,maximum=maximum,keys=unique,moments=moments,roots=len(rb))

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--previous-report',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    manifest=json.loads(a.campaign.read_text());sources=[]
    for f,pin in manifest['inputs'].items():
        if sha(Path(f))!=pin:raise ValueError('Campaign input changed')
    for k,v in manifest['cases'].items():sources.append((a.campaign.parent/k/'steps.phsp',v['hu']))
    if a.previous_report:
        for k,v in json.loads(a.previous_report.read_text())['cases'].items():
            for f,pin in v['artifacts'].items():
                if sha(Path(f))!=pin:raise ValueError('Previous raw changed')
            # Existing air adds only low bands; existing tissue strengthens all
            # 0-175 bins. Each original history occurs exactly once here.
            f=next(f for f in v['artifacts'] if f.endswith('steps.phsp'))
            sources.append((Path(f),-1000 if k.startswith('air/') else 100))
    if len({str(f.resolve()) for f,hu in sources})!=len(sources):raise ValueError('Duplicate source')
    aggregate={};provenance=[]
    for i,(f,hu) in enumerate(sources):
        r=extract(f,hu);aggregate.setdefault(hu,[]).append(r)
        # Preserve per-source intermediates for independent closure audits;
        # never require rerunning Monte Carlo to debug a compiler gate.
        np.savez(a.out/f'source_{i:03d}.npz',**r)
        provenance.append(dict(path=str(f),sha256=sha(f),hu=hu,roots=r['roots']))
        print(i+1,len(sources),hu,r['roots'],flush=True)
    metadata=dict(schema_version=1,status='UNVALIDATED_INTERFACE_DIAGNOSTIC',
        projectile=[6,12],energy_bin_width_MeVu=5,energy_max_MeVu=185,
        method='measured energy-weighted joint bulk displacement; no interface or patient fit',
        limitations=['finite-box neutral escape retained locally, not a photon model','net mass displacement approximation across composition changes'],
        fixed_parent_inset_mm={'air':100,'tissue':5},charged_escape_allowed=False,
        sources=provenance,compiler_sha256=sha(Path(__file__)),channels=[])
    output=a.out/'joint_response.csv'
    with output.open('x') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['section_id','energy_bin','nonlocal_fraction','cdf','longitudinal_mass_g_cm2','radial_mass_g_cm2'])
        for hu,runs in sorted(aggregate.items()):
            loss=sum(r['loss'] for r in runs);local=sum(r['local'] for r in runs);escape=sum(r['escape'] for r in runs)
            terminal=sum(r['terminal_local_excess'] for r in runs)
            minimum=np.min([r['minimum'] for r in runs],axis=0);maximum=np.max([r['maximum'] for r in runs],axis=0)
            keys=np.concatenate([r['keys'] for r in runs]);moments=np.concatenate([r['moments'] for r in runs])
            unique,inverse=np.unique(keys,axis=0,return_inverse=True)
            pooled=np.column_stack([np.bincount(inverse,weights=moments[:,i]) for i in range(3)])
            section=0 if hu==-1000 else 8
            for eb in range(NENERGY):
                if loss[eb]<=0:continue
                take=unique[:,0]==eb;values=pooled[take]
                mass=values[:,0].sum();fraction=mass/loss[eb]
                if abs((local[eb]+escape[eb]+mass-terminal[eb])/loss[eb]-1)>1e-7:
                    raise ValueError(f'Pooled closure: section={section}, bin={eb}, loss={loss[eb]}, local={local[eb]}, escape={escape[eb]}, deposits={mass}')
                if mass>0:
                    cdf=np.cumsum(values[:,0])/mass;cdf[-1]=1.
                    for row,c in zip(values,cdf):w.writerow([section,eb,fraction,c,row[1]/row[0],row[2]/row[0]])
                else:w.writerow([section,eb,0,1,0,0])
                metadata['channels'].append(dict(section_id=section,energy_bin=eb,energy_min_MeVu=float(minimum[eb]),energy_max_MeVu=float(maximum[eb]),
                    terminal_local_excess_MeV=float(terminal[eb]),
                    raw_kinetic_closure_residual_MeV=float(local[eb]+escape[eb]+mass-loss[eb]),
                    redistribution_inactive=bool(mass==0),
                    parent_loss_MeV=float(loss[eb]),nonlocal_fraction=float(fraction),escape_retained_fraction=float(escape[eb]/loss[eb])))
    metadata.update(data_sha256=sha(output),data_size_bytes=output.stat().st_size,data_filename=output.name)
    (a.out/'joint_response.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
