"""Export a distinct, hash-pinned diagnostic schema; never replace v2.1 data.

CSV supplies channel fractions/CDFs; companion binary retains the exact ordered
mass vectors for EVERY sampled deposit, aligned one-to-one with CSV rows.
"""
import argparse,csv,json,re,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha
from compare_electron_response_geometries import load_v2_binary

def domains(path,hu):
    d=load_v2_binary(path);p=d[d[:,3]==0];text=(path.parent/'run.txt').read_text()
    def mm(k):return float(re.search(r'^d:Ge/ZResponse/'+k+r' = ([\d.]+) mm$',text,re.M)[1])
    bounds=np.array([[-mm('HLX'),mm('HLX')],[-mm('HLY'),mm('HLY')],[mm('TransZ')-mm('HLZ'),mm('TransZ')+mm('HLZ')]])
    inset=100 if hu==-1000 else 5
    take=np.all((p[:,10:13]>=bounds[:,0]+inset)&(p[:,10:13]<=bounds[:,1]-inset)&(p[:,13:16]>=bounds[:,0]+inset)&(p[:,13:16]<=bounds[:,1]-inset),axis=1)
    e=p[take,17]/12;b=np.minimum((e/5).astype(int),36)
    low=np.full(37,np.inf);high=np.zeros(37);np.minimum.at(low,b,e);np.maximum.at(high,b,e)
    return low,high

def export(source,out):
    manifest=json.loads((source/'manifest.json').read_text());out.mkdir(exist_ok=False);groups={}
    for filename,info in manifest['sources'].items():
        file=source/filename;raw=Path(info['raw_path'])
        if sha(file)!=info['payload_sha256'] or sha(raw)!=info['raw_sha256'] or sha(raw.parent/'run.txt')!=info['config_sha256']:raise ValueError('Source pin')
        with np.load(file) as archive:arrays={key:archive[key] for key in archive.files}
        groups.setdefault(info['hu'],[]).append((arrays,domains(raw,info['hu'])))
    records=[];vectors=[];channels=[];csvfile=out/'joint_response.csv'
    with csvfile.open('x') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['section_id','energy_bin','nonlocal_fraction','cdf','longitudinal_mass_g_cm2','radial_mass_g_cm2'])
        for hu,items in sorted(groups.items()):
            section=0 if hu==-1000 else 8
            loss=sum(a['loss'] for a,d in items);local=sum(a['local'] for a,d in items);escape=sum(a['escape'] for a,d in items);terminal=sum(a['terminal'] for a,d in items)
            low=np.min([d[0] for a,d in items],axis=0);high=np.max([d[1] for a,d in items],axis=0)
            for b in range(37):
                if loss[b]<=0:continue
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
                    raw_kinetic_closure_residual_MeV=float(residual),redistribution_inactive=bool(dep==0),escape_retained_fraction=float(escape[b]/loss[b])))
    binary=out/'ordered_paths.bin'
    with binary.open('xb') as f:
        f.write(struct.pack('<8sIII',b'ELPATH01',1,len(records),len(vectors)))
        f.write(np.array(records,dtype='<u4').tobytes());f.write(np.array(vectors,dtype='<f8').tobytes())
    meta=dict(schema_version=2,status='UNVALIDATED_INTERFACE_DIAGNOSTIC',projectile=[6,12],energy_bin_width_MeVu=5,energy_max_MeVu=185,
        charged_escape_allowed=False,fixed_parent_inset_mm=dict(air=100,tissue=5),sources=list(manifest['sources'].values()),channels=channels,
        ordered_path_file=binary.name,ordered_path_sha256=sha(binary),ordered_path_size_bytes=binary.stat().st_size,
        data_filename=csvfile.name,data_size_bytes=csvfile.stat().st_size,data_sha256=sha(csvfile),
        compiler_sha256=sha(Path(__file__)),source_manifest_sha256=sha(source/'manifest.json'),limitations=manifest['limitations'])
    with csvfile.with_suffix('.metadata.json').open('x') as f:json.dump(meta,f,indent=2)
    print('Exported',len(records),'deposit paths,',len(vectors),'vectors; unvalidated',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();export(a.source,a.out)
