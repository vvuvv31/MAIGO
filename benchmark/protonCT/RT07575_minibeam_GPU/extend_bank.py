#!/usr/bin/env python3
"""Generate missing homogeneous material/density responses; never extrapolate or clamp CT."""
import concurrent.futures, hashlib, json, math, os, shutil, struct, subprocess, sys
from pathlib import Path
import numpy as np
import yaml
from common import ROOT,REPO,sha,save_json
OLD=Path('/mnt/sda/wuwei/real_ct_electron_diagnostic_20260928')
TOOLS=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/extensions/tools/material_electron')
BANK=ROOT/'response_extension'
BINARY=OLD/'build/topas'

def main():
    BANK.mkdir(exist_ok=True)
    f=(ROOT/'inputs/patient_world_packed.bin').open('rb');h=struct.unpack('<5I6f',f.read(44));n=int(np.prod(h[2:5]));rho=np.fromfile(f,'<f4',n);sec=np.fromfile(f,'u1',n)
    index_path=REPO/'data/ct_electron/proton/index.json';index=json.loads(index_path.read_text());bounds={}
    for t in index['tables']:
        i=t['material_identity'];bounds.setdefault(i['section_id'],[]).append(i['runtime_density_g_cm3'])
        for key in ['path','state_reference_file']:
            t[key]=str((index_path.parent/t[key]).resolve())
        i['section_metadata_path']=str((index_path.parent/i['section_metadata_path']).resolve())
    meta=REPO/'data/schneider/schneider_stopping_v1.metadata.json';sections=json.loads(meta.read_text())['section_manifest']
    elements=['Hydrogen','Carbon','Nitrogen','Oxygen','Magnesium','Phosphorus','Sulfur','Chlorine','Argon','Calcium','Sodium','Potassium','Titanium'];jobs=[];coverage=[]
    for sid in np.unique(sec):
        values=rho[sec==sid];lo=float(values.min());hi=float(values.max());old=bounds.get(int(sid),[]);needed=[]
        if not old:needed=sorted(set([lo,hi]))
        else:
            if lo<min(old):needed.append(lo)
            if hi>max(old):needed.append(hi)
        allvals=sorted(old+needed)
        for a,b in zip(allvals[:-1],allvals[1:]):
            count=math.ceil(math.log(b/a)/math.log(1.2))
            for k in range(1,count):
                val=float(np.float32(a*(b/a)**(k/count)))
                if val not in allvals:needed.append(val)
        coverage.append(dict(section=int(sid),min_density=lo,max_density=hi,previous_range=[min(old),max(old)] if old else None,added=sorted(needed)))
        truth=next(s for s in sections if s['section_id']==int(sid))
        for k,density in enumerate(sorted(needed)):
            tag=f'rt_s{sid}_{k}';d=BANK/tag;d.mkdir(exist_ok=True);name=f'RTResponseSection{sid}Density{k}';half=2000/density;pairs=[(a,float(w)) for a,w in zip(elements,truth['element_weights']) if w>0]
            lines=['i:Ts/NumberOfThreads = 2','i:Ts/Seed = '+str(1002079000+len(jobs)),'b:Ts/PauseBeforeQuit = "False"',
            'sv:Ph/Default/Modules = 3 "g4em-standard_opt4" "AllChargedUrbanPhysics" "PrimaryIonizationResponsePhysics"','d:Ph/Default/CutForAllParticles = 0.05 mm','d:Ph/Default/EMRangeMax = 10 GeV',
            's:Ge/World/Material = "Vacuum"',*[f'd:Ge/World/HL{axis} = {2*half:.17g} mm' for axis in 'XYZ'],
            f'sv:Ma/{name}/Components = {len(pairs)} '+' '.join('"'+a+'"' for a,w in pairs),f'uv:Ma/{name}/Fractions = {len(pairs)} '+' '.join(f'{w:.17g}' for a,w in pairs),
            f'd:Ma/{name}/Density = {density:.17g} g/cm3',f'd:Ma/{name}/MeanExcitationEnergy = {truth["mean_excitation_energy_eV"]:.17g} eV',
            's:Ge/ROI/Parent = "World"','s:Ge/ROI/Type = "TsBox"',f's:Ge/ROI/Material = "{name}"',*[f'd:Ge/ROI/HL{axis} = {half:.17g} mm' for axis in 'XYZ'],
            's:Ge/BeamPosition/Parent = "World"','s:Ge/BeamPosition/Type = "Group"',*[f'd:Ge/BeamPosition/Trans{axis} = 0 mm' for axis in 'XYZ'],*[f'd:Ge/BeamPosition/Rot{axis} = 0 deg' for axis in 'XYZ'],
            's:So/Beam/Type = "Beam"','s:So/Beam/Component = "BeamPosition"','s:So/Beam/BeamParticle = "proton"','d:So/Beam/BeamEnergy = 250 MeV','u:So/Beam/BeamEnergySpread = 0',
            's:So/Beam/BeamPositionDistribution = "None"','s:So/Beam/BeamAngularDistribution = "None"','i:So/Beam/NumberOfHistoriesInRun = 16',
            's:Sc/Steps/Quantity = "PrimaryElectronTransportNtuple"','s:Sc/Steps/Component = "ROI"','s:Sc/Steps/OutputType = "binary"','s:Sc/Steps/OutputFile = "steps"','s:Sc/Steps/IfOutputFileAlreadyExists = "Exit"','b:Sc/Steps/RecordTransportState = "True"']
            text='\n'.join(lines)+'\n';config=d/'topas.txt'
            if config.exists():assert config.read_text()==text
            else:config.write_text(text)
            identity=dict(kind='schneider_section',section_id=int(sid),material_name=name,runtime_density_g_cm3=density,section_metadata_path=str(meta),section_metadata_sha256=sha(meta),composition=truth)
            jobs.append(dict(tag=tag,path=str(d),histories=16,energy_MeVu=250,recording_bounds_mm=[[-half,half]]*3,config_sha256=sha(config),material_identity=identity))
    save_json(BANK/'coverage.json',coverage);save_json(BANK/'jobs.json',jobs)
    env=os.environ.copy();env.update(TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+env.get('LD_LIBRARY_PATH',''),OPENBLAS_NUM_THREADS='1')
    def run(j):
        d=Path(j['path'])
        if not (d/'complete.json').exists():
            with (d/'build.log').open('w') as log:
                if not (d/'steps.phsp').exists():
                    with (d/'topas.log').open('w') as raw_log:
                        subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=raw_log,stderr=subprocess.STDOUT,check=True)
                manifest=dict(projectile=[1,1],material_identity=j['material_identity'],scored_component='ROI',production_cut_mm=.05,pins={str(BINARY):sha(BINARY),str(OLD/'extensions/PrimaryElectronTransportNtuple.cc'):sha(OLD/'extensions/PrimaryElectronTransportNtuple.cc'),str(meta):sha(meta)},jobs=[j])
                save_json(d/'source_manifest.json',manifest)
                subprocess.run([sys.executable,str(TOOLS/'compile.py'),'--manifest',str(d/'source_manifest.json'),'--output',str(d/'response.bin'),'--energy-max','250','--projectile-z','1','--projectile-a','1','--material-response','--transport-state-source'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,str(TOOLS/'export_state_refs.py'),'--metadata',str(d/'response.metadata.json'),'--metadata-sha256',sha(d/'response.metadata.json'),'--output',str(d/'state_refs.bin')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            save_json(d/'complete.json',dict(status='COMPILED'))
        print('compiled',j['tag'],flush=True)
        return dict(status='COMPILED',material_identity=j['material_identity'],path=str(d/'response.bin'),data_sha256=sha(d/'response.bin'),metadata_sha256=sha(d/'response.metadata.json'),state_reference_file=str(d/'state_refs.bin'),state_reference_sha256=sha(d/'state_refs.bin'),state_reference_metadata_sha256=sha(d/'state_refs.metadata.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:tables=list(pool.map(run,jobs))
    index['tables']+=tables;save_json(BANK/'index.json',index)
    manifest=json.loads((ROOT/'manifest.json').read_text());manifest['electron_bank_extension']=dict(path=str(BANK/'index.json'),sha256=sha(BANK/'index.json'),added_tables=len(tables),histories_per_new_table=16,scope='same homogeneous 250 MeV proton recording method as original research bank; independent of patient reference dose')
    for job in manifest['jobs']+manifest['pilots']:
        path=ROOT/(job['name']+'.yaml');c=yaml.safe_load(path.read_text());c.update(material_electron_response_index_file=str(BANK/'index.json'),material_electron_response_index_sha256=sha(BANK/'index.json'));path.write_text(yaml.safe_dump(c,sort_keys=False));job['config_sha256']=sha(path)
    save_json(ROOT/'manifest.json',manifest);print('extended',len(tables),'tables',flush=True)
if __name__=='__main__':main()
