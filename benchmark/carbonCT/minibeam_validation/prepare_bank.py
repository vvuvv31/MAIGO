#!/usr/bin/env python3
"""Independent C12 homogeneous electron extraction for the shipped lung CT.

Uses the proton bank's material/density grid only, never its particle responses.
Original recorded transport states remain runtime dependencies.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,hashlib,json,os,re,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'out/carbon_ct_minibeam_20261002'
OLD=Path('/mnt/sda/wuwei/real_ct_electron_diagnostic_20260928')
BINARY=OLD/'build/topas'
TOOLS_SOURCE=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/extensions/tools/material_electron')
BANK=ROOT/'data/ct_electron/carbon'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--start',type=int,default=0);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);BANK.mkdir(parents=True,exist_ok=True)
    tools=OUT/'compiler';tools.mkdir(exist_ok=True)
    for src in TOOLS_SOURCE.glob('*.py'):
        dst=tools/src.name
        if dst.exists():assert sha(dst)==sha(src)
        else:shutil.copy2(src,dst)
    index=json.loads((ROOT/'data/ct_electron/proton/index.json').read_text())
    jobs=[]
    for i,table in enumerate(index['tables']):
        old=OLD/'bank'/Path(table['path']).parent
        tag=old.name;d=BANK/tag;d.mkdir(exist_ok=True)
        text=(old/'topas.txt').read_text()
        text=text.replace('s:So/Beam/BeamParticle = "proton"','s:So/Beam/BeamParticle = "GenericIon(6,12)"')
        text=text.replace('d:So/Beam/BeamEnergy = 250 MeV','d:So/Beam/BeamEnergy = 3600 MeV')
        text=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {1002108000+i}',text)
        config=d/'topas.txt'
        if config.exists():assert config.read_text()==text
        else:config.write_text(text)
        identity=copy.deepcopy(table['material_identity'])
        identity['section_metadata_path']=str(ROOT/'data/schneider/schneider_stopping_v1.metadata.json')
        assert sha(identity['section_metadata_path'])==identity['section_metadata_sha256']
        half=float(re.search(r'd:Ge/ROI/HLX = ([\d.eE+-]+) mm',text)[1])
        jobs.append(dict(tag=tag,path=str(d),histories=16,energy_MeVu=300,
            recording_bounds_mm=[[-half,half]]*3,config_sha256=sha(config),material_identity=identity))
    save(OUT/'bank_jobs.json',jobs)
    pins={str(BINARY):sha(BINARY),str(OLD/'extensions/PrimaryElectronTransportNtuple.cc'):sha(OLD/'extensions/PrimaryElectronTransportNtuple.cc'),
          str(OLD/'extensions/PrimaryIonizationResponsePhysics.cc'):sha(OLD/'extensions/PrimaryIonizationResponsePhysics.cc')}
    pins.update({str(p):sha(p) for p in tools.glob('*.py')})
    save(OUT/'bank_tools.json',pins)
    env=os.environ.copy();env.update(TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
        LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+env.get('LD_LIBRARY_PATH',''),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(job):
        d=Path(job['path']);start=time.monotonic()
        if not (d/'complete.json').exists():
            if not (d/'extraction_complete.json').exists():
                # Refuse incomplete raw rows from an interrupted/failed extraction.
                if (d/'steps.phsp').exists():raise RuntimeError('Incomplete extraction: '+str(d))
                with (d/'topas.log').open('w') as log:
                    subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                save(d/'extraction_complete.json',dict(config_sha256=sha(d/'topas.txt'),raw_sha256=sha(d/'steps.phsp')))
            m=dict(projectile=[6,12],material_identity=job['material_identity'],scored_component='ROI',production_cut_mm=.05,pins=pins,jobs=[job])
            save(d/'source_manifest.json',m)
            with (d/'compile.log').open('w') as log:
                if not (d/'response.metadata.json').exists():
                    subprocess.run([sys.executable,str(tools/'compile.py'),'--manifest',str(d/'source_manifest.json'),
                        '--output',str(d/'response.bin'),'--energy-max','300','--projectile-z','6','--projectile-a','12',
                        '--material-response','--transport-state-source'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if not (d/'state_refs.metadata.json').exists():
                    subprocess.run([sys.executable,str(tools/'export_state_refs.py'),'--metadata',str(d/'response.metadata.json'),
                        '--metadata-sha256',sha(d/'response.metadata.json'),'--output',str(d/'state_refs.bin')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            save(d/'complete.json',dict(status='COMPILED',seconds=time.monotonic()-start))
        print('compiled',job['tag'],round(time.monotonic()-start,2),flush=True)
        return dict(status='COMPILED',material_identity=job['material_identity'],path=job['tag']+'/response.bin',
            data_sha256=sha(d/'response.bin'),metadata_sha256=sha(d/'response.metadata.json'),
            state_reference_file=job['tag']+'/state_refs.bin',state_reference_sha256=sha(d/'state_refs.bin'),
            state_reference_metadata_sha256=sha(d/'state_refs.metadata.json'))
    chosen=jobs[args.start:args.start+args.limit] if args.limit else jobs[args.start:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:tables=list(pool.map(run,chosen))
    result=dict(schema_version=2,projectile=[6,12],status='UNVALIDATED_MATERIAL_RESPONSE_BANK',tables=tables)
    name=f'index_partial_{args.start}_{len(chosen)}.json' if args.limit or args.start else 'index.json'
    save(BANK/name,result)
    print('finished',len(tables),'tables',flush=True)

if __name__=='__main__':main()
