"""Compile an OFFLINE diagnostic mass-displacement response, never a runtime package.

Normalize to measured primary step energy loss, not dose-fit amplitudes. Keep
correlated radial/longitudinal deposit samples and reject truncated families.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha
from compare_electron_response_geometries import load_v2_binary,build_roots

def extract(folder,bounds,hu):
    d=load_v2_binary(folder/'steps.phsp')
    if set(d[:,1].astype(int))!=set(range(4)):raise ValueError('Incomplete four-history sample')
    if len(np.unique(d[:,[0,1,2,5]],axis=0))!=len(d):raise ValueError('Duplicate steps')
    tracks,root=build_roots(d)
    first={};last={};parents={}
    for i,row in enumerate(d):
        key=tuple(row[:3].astype(int));first.setdefault(key,i);last[key]=i
        if row[3]==0 and 170<=row[17]/12<=175.01:
            parents[key+(int(row[5]),)]=i
    selected=set()
    for key in tracks:
        if root(key)!=key:continue
        row=d[first[key]]
        if row[22]!=1 or row[29]<1:raise ValueError('Invalid parent association')
        pk=(key[0],key[1],int(row[3]),int(row[29]))
        if pk in parents:selected.add(key)
    if not selected:raise ValueError('No selected electron roots')
    p=d[list(parents.values())]
    loss=float(np.sum((p[:,17]-p[:,18])*p[:,19]));local=float(np.sum(p[:,16]*p[:,19]))
    birth=float(sum(d[first[k],9]*d[first[k],19] for k in selected))
    if abs((local+birth)/loss-1)>1e-5:raise ValueError('Primary birth energy does not close')
    rows=[];origins=[];directions=[];escaped=0.
    bounds=np.array(bounds)
    for key in tracks:
        rk=root(key)
        if rk not in selected:continue
        terminal=d[last[key]]
        if terminal[18]>1e-9:
            xyz=terminal[13:16]
            on_boundary=np.any(np.minimum(abs(xyz-bounds[:,0]),abs(xyz-bounds[:,1]))<1e-5)
            if not on_boundary:raise ValueError('Positive terminal KE away from physical boundary')
            escaped+=terminal[18]*terminal[19]
    for row in d:
        rk=root(tuple(row[:3].astype(int)))
        if rk not in selected or row[16]<=0:continue
        b=d[first[rk]]
        rows.append(row);origins.append(b[6:9]);directions.append(b[24:27])
    rows=np.array(rows);origins=np.array(origins);directions=np.array(directions)
    dep=float(np.sum(rows[:,16]*rows[:,19]))
    if abs((local+dep+escaped)/loss-1)>1e-5:raise ValueError('Electron family budget does not close')
    # Even a small nonzero escaped energy is exposed, never silently normalized
    # into contained deposits. Such a response cannot be promoted as complete.
    rho=.01131606474518776 if hu==-1000 else 1.0787997245788574
    if np.max(abs(rows[:,20]/rho-1))>1e-4:raise ValueError('Mixed material response')
    start=rows[:,10:13]-origins;end=rows[:,13:16]-origins
    dz0=np.sum(start*directions,axis=1)*rho/10
    dz1=np.sum(end*directions,axis=1)*rho/10
    pitch=.00025
    counts=np.maximum(1,np.ceil(abs(dz1-dz0)/pitch).astype(int))
    # Photon interactions are discrete at the post-step point, not uniform
    # energy deposition along their potentially macroscopic free flight.
    photons=rows[:,4]==22
    counts[photons]=1
    if counts.max()>4096:raise ValueError('Unresolved deposit segment')
    ids=np.repeat(np.arange(len(rows)),counts)
    sub=(np.arange(len(ids))-np.repeat(np.cumsum(counts)-counts,counts)+.5)/counts[ids]
    sub[photons[ids]]=1.0
    point=start[ids]+sub[:,None]*(end-start)[ids]
    long=np.sum(point*directions[ids],axis=1)
    radial=np.linalg.norm(point-long[:,None]*directions[ids],axis=1)*rho/10
    long=long*rho/10
    weights=rows[ids,16]*rows[ids,19]/counts[ids]
    maximum=np.linalg.norm(bounds[:,1]-bounds[:,0])*rho/10
    if np.max(abs(long))>maximum or radial.max()>maximum:raise ValueError('Response exceeds physical geometry')
    # Sparse JOINT table; never independently sample longitudinal/radial tails.
    bins=np.column_stack((np.floor(long/pitch).astype(int),np.floor(radial/pitch).astype(int)))
    unique,inverse=np.unique(bins,axis=0,return_inverse=True)
    energies=np.bincount(inverse,weights=weights)
    centroids=np.column_stack((np.bincount(inverse,weights=weights*long)/energies,
                               np.bincount(inverse,weights=weights*radial)/energies))
    return dict(histories=len(np.unique(d[:,1])),roots=len(selected),parent_steps=len(parents),
        parent_energy_range_MeVu=[float(p[:,17].min()/12),float(p[:,17].max()/12)],
        loss_MeV=loss,local_MeV=local,electron_birth_MeV=birth,electron_deposit_MeV=dep,
        escaped_MeV=escaped,energy_closure=(local+dep+escaped)/loss-1,
        maximum_longitudinal_mass_g_cm2=float(np.max(abs(long))),
        maximum_radial_mass_g_cm2=float(radial.max()),
        direct_photon_deposit_MeV=float(np.sum(rows[photons,16]*rows[photons,19])),
        mass_bin_pitch_g_cm2=pitch,complete_no_escape=bool(escaped<1e-9),
        artifacts={str(f):sha(f) for f in [folder/'steps.phsp',folder/'steps.header',folder/'run.txt']}),unique,energies,centroids

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retry-root',type=Path)
    a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text());a.out.mkdir(exist_ok=False)
    for f,pin in manifest['inputs'].items():
        if sha(Path(f))!=pin:raise ValueError('Changed input '+f)
    retries={}
    if a.retry_root:
        retry_manifest=json.loads((a.retry_root/'manifest.json').read_text())
        for f,pin in retry_manifest['inputs'].items():
            if sha(Path(f))!=pin:raise ValueError('Changed retry input '+f)
        retries=retry_manifest['cases']
        for name,c in retries.items():
            if name not in manifest['cases'] or c!=manifest['cases'][name]:raise ValueError('Retry geometry mismatch')
            original=a.root/name;replacement=a.retry_root/name
            errors=''.join(p.read_text() for p in original.glob('job_*.err'))
            if 'OOM' not in errors and 'oom_kill' not in errors:raise ValueError('Retry not backed by OOM evidence')
            if (original/'run.txt').read_text().replace(str(original),str(replacement))!=(replacement/'run.txt').read_text():
                raise ValueError('Retry changed physics/source configuration')
    reports={}
    for name,c in manifest['cases'].items():
        folder=(a.retry_root if name in retries else a.root)/name
        report,bins,energies,centroids=extract(folder,c['bounds_mm'],c['hu']);reports[name]=report
        report['retry_after_oom']=name in retries
        output=a.out/(name.replace('/','_')+'.npz')
        np.savez(output,bins=bins,energy_MeV=energies,centroid_mass_g_cm2=centroids)
        report['response_sha256']=sha(output)
        print(name,{k:report[k] for k in ['roots','loss_MeV','local_MeV','escaped_MeV','energy_closure']},flush=True)
    (a.out/'report.json').write_text(json.dumps(dict(status='OFFLINE_PILOT_ONLY',cases=reports,
        analyzer_sha256=sha(Path(__file__)),manifest_sha256=sha(a.root/'manifest.json')),indent=2)+'\n')
