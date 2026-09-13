#!/usr/bin/env python3
"""Compile a complete Geant4 material/density x 18-ion export into one EM package."""
from pathlib import Path
import argparse,csv,json,struct,hashlib,tempfile
import numpy as np

def compile_package(source,output):
    manifest=json.loads((source/'extraction_manifest.json').read_text())
    materials=manifest['materials'];species=manifest['species_za'];records=[];nodes=[];vectors=[]
    node_offset=vector_offset=0;source_hashes={};binary_hashes=set()
    catalogs={}
    for z,a in species:
        root=source/f'ion_{z}_{a}'/'dump_v5';status=json.loads((root/'status.json').read_text())
        if not status['complete'] or status['material_density_count']!=len(materials):raise ValueError('Incomplete export')
        binary_hashes.add(status['binary_sha256'])
        catalogs[z,a]={v['material']:v for v in csv.DictReader((root/'catalog.csv').open())}
    if len(binary_hashes)!=1:raise ValueError('Mixed extraction binaries')
    for material in materials:
      for z,a in species:
        d=source/f'ion_{z}_{a}'/'dump_v5'/material['name'];meta=catalogs[z,a][material['name']]
        n=np.loadtxt(d/'nodes.csv',delimiter=',',skiprows=1,ndmin=2)
        v=np.loadtxt(d/'vectors.csv',delimiter=',',skiprows=1,ndmin=2)
        models={int(row['id']):row for row in csv.DictReader((d/'models.csv').open())}
        codes={'Bragg':0,'BetheBloch':1,'BraggIon':2,'LindhardSorensen':3}
        for idx,row in models.items():
            if row['model'] not in codes or row['fluctuation'] not in ['IonFluc','UrbanFluc','UniversalFluc']:raise ValueError('Unsupported model '+str(row))
            sel=n[:,11]==idx;n[sel,11]=codes[row['model']];n[sel,12]=1 if row['fluctuation']=='IonFluc' else 0
        if n.shape[1]!=13 or not np.isfinite(n).all() or np.any(np.diff(n[:,0])<=0):raise ValueError('Invalid nodes')
        if np.any(n[:,[1,2,3,4,5,6,8,9,10]]<0):raise ValueError('Negative physical node field')
        _, unique = np.unique(n[:,0].astype('<f4'), return_index=True)
        n=n[np.sort(unique)]
        offsets=[];counts=[];parts=[]
        for k in range(4):
            arows=v[v[:,0]==k]
            if not len(arows) or np.any(arows[:,3]<=arows[:,2]) or not np.all(arows[:,1]==np.arange(len(arows))):raise ValueError('Bad native vector intervals')
            offsets.append(vector_offset+sum(counts));counts.append(len(arows));parts.append(arows[:,2:8])
        raw=np.concatenate(parts).astype('<f4');peak_rows=v[v[:,0]==3];peak=float(peak_rows[np.argmax(peak_rows[:,4]),2])/(a*float(meta['mass_ratio']))
        keys=['mass_MeV','mass_ratio','cut_MeV','I_MeV','e0_MeV','spin','step_fraction','final_range_mm','linear_loss_limit','is_ion','form_factor_per_MeV','magnetic_moment_square_minus_one','lowest_kinetic_MeV']
        values=[float(meta[k]) for k in keys]+[peak]
        records.append(struct.pack('<14f12I',*values,node_offset,len(n),*offsets,*counts,z,a))
        nodes.append(n.astype('<f4'));vectors.append(raw);node_offset+=len(n);vector_offset+=len(raw)
        for p in d.glob('*.csv'):source_hashes[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as f:
        tmp=Path(f.name);f.write(struct.pack('<8s6I',b'EMJOINT1',1,len(materials),len(species),len(records),node_offset,vector_offset))
        for m in materials:f.write(struct.pack('<if',m['section'],m['density_g_cm3']))
        for z,a in species:f.write(struct.pack('<2I',z,a))
        for r in records:f.write(r)
        for n in nodes:f.write(n.tobytes())
        for v in vectors:f.write(v.tobytes())
    tmp.replace(output)
    metadata=dict(manifest,validation_status='research_candidate_not_validated',schema='EMJOINT1',version=1,records=len(records),nodes=node_offset,segments=vector_offset,record_bytes=104,node_bytes=52,segment_bytes=24,extraction_binary_sha256=next(iter(binary_hashes)),source_sha256=source_hashes,package_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),package_bytes=output.stat().st_size)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['records','nodes','segments','package_bytes','package_sha256']},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();compile_package(args.source,args.output)
