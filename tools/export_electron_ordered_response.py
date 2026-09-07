"""Export a distinct, hash-pinned diagnostic schema; never replace v2.1 data.

CSV supplies channel fractions/CDFs; companion binary retains the exact ordered
mass vectors for EVERY sampled deposit, aligned one-to-one with CSV rows.
"""
import argparse,csv,json,re,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha
from compare_electron_response_geometries import load_v2_binary

def canonical_ceiling_roundoff(energy, ceiling):
    # Geant4's 6000 MeV / 12 can be 500 + 3 double ULPs. This is
    # metadata endpoint canonicalization, never physical extrapolation.
    energy=np.asarray(energy)
    if np.any(energy>ceiling+8*np.spacing(float(ceiling))):
        raise ValueError('Measured energy exceeds declared response ceiling')
    return np.minimum(energy,ceiling)

def swept_bin_lower(pre_energy, post_energy, bins):
    pre_energy=np.asarray(pre_energy);post_energy=np.asarray(post_energy)
    if np.any(post_energy<0) or np.any(post_energy>pre_energy+1e-10):
        raise ValueError('Invalid parent slowing step')
    # Response weights use energy lost over the whole recorded parent step,
    # including the final step down to zero; pre-step extrema alone truncate
    # that actually recorded support. Keep the histogram's bin assignment.
    return np.maximum(np.asarray(bins)*5.,post_energy)

def domains(path,hu,inset=None,energy_max=185):
    d=load_v2_binary(path);p=d[d[:,3]==0];text=(path.parent/'run.txt').read_text()
    def mm(k):return float(re.search(r'^d:Ge/ZResponse/'+k+r' = ([\d.]+) mm$',text,re.M)[1])
    bounds=np.array([[-mm('HLX'),mm('HLX')],[-mm('HLY'),mm('HLY')],[mm('TransZ')-mm('HLZ'),mm('TransZ')+mm('HLZ')]])
    if inset is None:inset=100 if hu==-1000 else 5
    take=np.all((p[:,10:13]>=bounds[:,0]+inset)&(p[:,10:13]<=bounds[:,1]-inset)&(p[:,13:16]>=bounds[:,0]+inset)&(p[:,13:16]<=bounds[:,1]-inset),axis=1)
    nb=int(np.ceil(energy_max/5));e=canonical_ceiling_roundoff(p[take,17]/12,energy_max);b=np.minimum((e/5).astype(int),nb-1)
    low=np.full(nb,np.inf);high=np.zeros(nb)
    np.minimum.at(low,b,swept_bin_lower(e,p[take,18]/12,b));np.maximum.at(high,b,e)
    return low,high

def export(source,out):
    manifest=json.loads((source/'manifest.json').read_text());out.mkdir(exist_ok=False);groups={}
    energies={s.get('energy_max_MeVu',185) for s in manifest['sources'].values()}
    if len(energies)!=1 or next(iter(energies)) not in (185,500):raise ValueError('Response energy grid mismatch')
    energy_max=next(iter(energies));nb=int(np.ceil(energy_max/5));full=energy_max==500
    for filename,info in manifest['sources'].items():
        file=source/filename;raw=Path(info['raw_path'])
        if sha(file)!=info['payload_sha256'] or sha(raw)!=info['raw_sha256'] or sha(raw.parent/'run.txt')!=info['config_sha256']:raise ValueError('Source pin')
        with np.load(file) as archive:arrays={key:archive[key] for key in archive.files}
        section=info.get('section_id',{-1000:0,100:8}.get(info['hu']))
        if section is None or not 0<=section<25:raise ValueError('Missing Schneider section identity')
        groups.setdefault(section,[]).append((arrays,domains(raw,info['hu'],info.get('parent_inset_mm'),energy_max)))
    if full and set(groups)!=set(range(25)):raise ValueError('Incomplete Schneider section coverage')
    records=[];vectors=[];channels=[];csvfile=out/'joint_response.csv'
    with csvfile.open('x') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['section_id','energy_bin','nonlocal_fraction','cdf','longitudinal_mass_g_cm2','radial_mass_g_cm2'])
        for section,items in sorted(groups.items()):
            loss=sum(a['loss'] for a,d in items);local=sum(a['local'] for a,d in items);escape=sum(a['escape'] for a,d in items);terminal=sum(a['terminal'] for a,d in items)
            low=np.min([d[0] for a,d in items],axis=0);high=np.max([d[1] for a,d in items],axis=0)
            for b in range(nb):
                if loss[b]<=0:
                    if full:raise ValueError(f'Missing measured energy bin: section={section} bin={b}')
                    continue
                selected=[(a,j) for a,d in items for j in np.flatnonzero(a['bins']==b)]
                dep=sum(a['weights'][j] for a,j in selected);fraction=dep/loss[b]
                residual=local[b]+escape[b]+dep-loss[b]
                if abs(residual-terminal[b])>1e-7*loss[b]:raise ValueError('Export closure')
                cumulative=0.
                if not selected:
                    selected=[(None,None)]
                for number,(a,j) in enumerate(selected):
                    if a is None:v=np.zeros((1,3));cdf=1.
                    else:
                        v=a['vectors'][a['offsets'][j]:a['offsets'][j+1]].copy()
                        net=v.sum(axis=0);phi=np.arctan2(net[1],net[0]);co,si=np.cos(phi),np.sin(phi)
                        # Align net transverse direction with CSV's +x, keeping
                        # internal turns; one common GPU azimuth rotates all.
                        x=v[:,0].copy();y=v[:,1].copy();v[:,0]=co*x+si*y;v[:,1]=-si*x+co*y
                        cumulative+=a['weights'][j]/dep;cdf=1. if number==len(selected)-1 else cumulative
                    net=v.sum(axis=0);records.append((len(vectors),len(v)));vectors.extend(v)
                    w.writerow([section,b,fraction,cdf,net[2],float(np.hypot(net[0],net[1]))])
                channels.append(dict(section_id=section,energy_bin=b,energy_min_MeVu=float(low[b]),energy_max_MeVu=float(high[b]),
                    nonlocal_fraction=float(fraction),parent_loss_MeV=float(loss[b]),terminal_local_excess_MeV=float(terminal[b]),
                    raw_local_MeV=float(local[b]),kinetic_local_MeV=float(local[b]-terminal[b]),nonlocal_deposit_MeV=float(dep),neutral_escape_retained_MeV=float(escape[b]),
                    raw_kinetic_closure_residual_MeV=float(residual),redistribution_inactive=bool(dep==0),escape_retained_fraction=float(escape[b]/loss[b])))
    binary=out/'ordered_paths.bin'
    with binary.open('xb') as f:
        f.write(struct.pack('<8sIII',b'ELPATH01',1,len(records),len(vectors)))
        f.write(np.array(records,dtype='<u4').tobytes());f.write(np.array(vectors,dtype='<f8').tobytes())
    meta=dict(schema_version=3 if full else 2,status='UNVALIDATED_INTERFACE_DIAGNOSTIC',projectile=[6,12],energy_bin_width_MeVu=5,energy_max_MeVu=energy_max,
        charged_escape_allowed=False,energy_domain_convention='recorded_parent_step_swept_energy_clipped_to_pre_bin',fixed_parent_inset_mm=({str(s['section_id']):s['parent_inset_mm'] for s in manifest['sources'].values()} if full else dict(air=100,tissue=5)),sources=list(manifest['sources'].values()),channels=channels,
        ordered_path_file=binary.name,ordered_path_sha256=sha(binary),ordered_path_size_bytes=binary.stat().st_size,
        data_filename=csvfile.name,data_size_bytes=csvfile.stat().st_size,data_sha256=sha(csvfile),
        compiler_sha256=sha(Path(__file__)),source_manifest_sha256=sha(source/'manifest.json'),limitations=manifest['limitations'])
    with csvfile.with_suffix('.metadata.json').open('x') as f:json.dump(meta,f,indent=2)
    print('Exported',len(records),'deposit paths,',len(vectors),'vectors; unvalidated',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();export(a.source,a.out)
