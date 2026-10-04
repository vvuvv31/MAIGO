#!/usr/bin/env python3
"""C12 birth-only extension at missing Schneider density nodes; no electron transport changes."""
import argparse,concurrent.futures,json,os,re,struct,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/electron_response_model'))
from prototype import sha,save
INDEX=ROOT/'benchmark/protonCT/RT07575_minibeam_GPU/response_extension/index.json'
BINARY=ROOT/'out/broadbeam_shared_electron_20261004/native/topas'
DTYPE=np.dtype([(k,'<i4') for k in ['run','event','kind','bin','step']]+[(k,'<f8') for k in ['parent','birth','mu','loss','local','children','residual','length','density']])
assert DTYPE.itemsize==92

def one(task):
    i,entry,out,histories,verify=task;d=out/f'node{i:02d}'
    if (d/'complete.json').exists():
        done=json.loads((d/'complete.json').read_text())
        assert done['histories']==histories and sha(d/'birth.bin')==done['sha256']
        assert done['material_identity']==entry['material_identity'] and done['binary_sha256']==sha(BINARY)
        assert not verify or done['full_state_audit'] is not None
        return done
    d.mkdir(exist_ok=False)
    p=Path(entry['path']);p=p if p.is_absolute() else INDEX.parent/p
    template=p.parent/'topas.txt'
    if not template.exists():template=ROOT/'data/ct_electron/carbon'/p.parent.name/'topas.txt'
    text=re.sub(r'i:Ts/NumberOfThreads = .*','i:Ts/NumberOfThreads = 1',template.read_text());identity=entry['material_identity'];rho=identity['runtime_density_g_cm3']
    assert float(re.search(r'd:Ma/[^\n]+/Density = (\S+) g/cm3',text)[1])==rho
    assert abs(float(re.search(r'd:Ma/[^\n]+/MeanExcitationEnergy = (\S+) eV',text)[1])-identity['composition']['mean_excitation_energy_eV'])<1e-10
    text=re.sub(r's:So/Beam/BeamParticle = .*','s:So/Beam/BeamParticle = "GenericIon(6,12)"',text)
    text=re.sub(r'd:So/Beam/BeamEnergy = .*','d:So/Beam/BeamEnergy = 3060 MeV',text)
    text=re.sub(r'i:So/Beam/NumberOfHistoriesInRun = .*',f'i:So/Beam/NumberOfHistoriesInRun = {histories}',text)
    text=re.sub(r'i:Ts/Seed = .*',f'i:Ts/Seed = {114030001+i*1009}',text)
    text=re.sub(r's:Sc/Steps/Quantity = .*','s:Sc/Steps/Quantity = "ProtonElectronBirthNtuple"',text)
    text=re.sub(r'b:Sc/Steps/RecordTransportState = .*\n','',text)
    text+='\nd:Ge/ROI/MaxStepSize = 0.25 mm\n'
    if verify:
        text+='\ns:Sc/Full/Quantity = "PrimaryElectronTransportNtuple"\ns:Sc/Full/Component = "ROI"\ns:Sc/Full/OutputType = "binary"\ns:Sc/Full/OutputFile = "full"\ns:Sc/Full/IfOutputFileAlreadyExists = "Exit"\nb:Sc/Full/RecordTransportState = "True"\n'
    (d/'topas.txt').write_text(text)
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    start=time.perf_counter()
    with (d/'topas.log').open('x') as f:r=subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,('native failed',d,r.returncode)
    log=(d/'topas.log').read_text();counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',log)
    assert counts and int(counts[-1])==histories
    assert sum(map(float,re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',log)))==0
    raw=np.fromfile(d/'steps.phsp',DTYPE);assert raw.nbytes==(d/'steps.phsp').stat().st_size and np.isin(raw['kind'],[0,1]).all()
    assert np.all(abs(raw['density']-rho)<=4*np.finfo(float).eps*rho) and np.all(raw['density'].astype(np.float32)==np.float32(rho))
    assert len(np.unique(raw['event']))==histories
    for k in DTYPE.names[5:]:assert np.isfinite(raw[k]).all(),k
    summary=raw[raw['kind']==0];birth=raw[raw['kind']==1]
    assert np.all(summary['length']<=.250001) and np.all(summary['residual']<1e-8)
    assert np.max(abs(np.bincount(summary['event'],weights=summary['loss'])-3060))<1e-7
    assert np.all(birth['birth']>0) and np.all(abs(birth['mu'])<=1)
    assert np.all(np.minimum((birth['parent']/5).astype(int),50)==birth['bin'])
    channels=[];samples=[];offset=0
    for b in range(51):
        rows=summary[summary['bin']==b];bs=birth[birth['bin']==b]
        loss=rows['loss'].sum();transfer=rows['children'].sum();local=rows['local'].sum()
        assert loss>0 and abs(local+transfer-loss)<1e-7*loss and abs(bs['birth'].sum()-transfer)<1e-10*loss
        cdf=np.cumsum(bs['birth'])/transfer if len(bs) else np.empty(0)
        if len(cdf):cdf[-1]=1
        samples.append(np.column_stack([cdf,bs['birth'],bs['mu']]).astype('<f8'))
        channels.append((5*b,5*(b+1),transfer/loss,offset,len(bs)));offset+=len(bs)
    payload=d/'birth.bin'
    with payload.open('xb') as f:
        f.write(struct.pack('<idQI',identity['section_id'],rho,offset,51))
        for c in channels:f.write(struct.pack('<dddQQ',*c))
        for arr in samples:f.write(arr.tobytes())
    audit=None
    if verify:
        from macro_model import DTYPE as FULL
        a=np.fromfile(d/'full.phsp',FULL);primary=a[a['parent']==0];children=a[(a['parent']==1)&(a['pdg']==11)&(a['step']==1)]
        # Compare every individual birth, not only energy-bin totals.
        bsort=np.lexsort((birth['birth'],birth['step'],birth['event'],birth['run']));csort=np.lexsort((children['birth'],children['parent_step'],children['event'],children['run']))
        b=birth[bsort];c=children[csort];assert len(b)==len(c)
        assert np.array_equal(b['event'],c['event']) and np.array_equal(b['step'],c['parent_step'])
        assert np.array_equal(b['birth'],c['birth'])
        parentkeys={(int(x['run']),int(x['event']),int(x['step'])):x for x in primary}
        maxangle=0
        for x,y in zip(b,c):
            parent=parentkeys[(int(x['run']),int(x['event']),int(x['step']))]
            assert x['parent']==parent['energy']/12
            maxangle=max(maxangle,abs(x['mu']-np.dot(parent['direction'],y['direction'])))
        assert maxangle<1e-12
        for b in range(51):
            pr=primary[np.minimum((primary['energy']/60).astype(int),50)==b];s=summary[summary['bin']==b]
            assert abs((pr['energy']-pr['remaining']).sum()-s['loss'].sum())<1e-9
            assert abs(pr['dep'].sum()-s['local'].sum())<1e-9
        audit=dict(full_state_sha256=sha(d/'full.phsp'),birth_records=len(birth),all_birth_energies_exact=True,max_cosine_difference=maxangle,all_primary_bins_match=True)
    done=dict(node=i,material_identity=identity,histories=histories,source_primary_MeV=3060,production_cut_mm=.05,maximum_step_mm=.25,birth_samples=offset,bytes=payload.stat().st_size,sha256=sha(payload),raw_sha256=sha(d/'steps.phsp'),input_sha256=sha(d/'topas.txt'),binary_sha256=sha(BINARY),seconds=time.perf_counter()-start,full_state_audit=audit,status='EXPERIMENTAL_FRESH_BIRTH_DISTRIBUTION_NOT_DOSE_ACCEPTED')
    save(d/'complete.json',done);print('COMPLETE',i,offset,done['bytes'],flush=True);return done

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--histories',type=int,default=64);p.add_argument('--nodes',nargs='*',type=int);p.add_argument('--workers',type=int,default=3);p.add_argument('--verify-full',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    tables=json.loads(INDEX.read_text())['tables'];nodes=a.nodes if a.nodes is not None else list(range(len(tables)))
    tasks=[(i,tables[i],a.out,a.histories,a.verify_full) for i in nodes]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    if nodes!=list(range(len(tables))):return
    output=a.out/'birth_bank.bin'
    with output.open('xb') as f:
        f.write(struct.pack('<8sIIII',b'MAIBIR02',2,len(results),6,12))
        for i,r in enumerate(results):
            file=a.out/f'node{i:02d}/birth.bin';assert sha(file)==r['sha256'];f.write(file.read_bytes())
    save(a.out/'manifest.json',dict(sha256=sha(output),bytes=output.stat().st_size,index_provenance_sha256=sha(INDEX),results=results,generator_sha256=sha(Path(__file__)),scope='C12 0-255MeV/u; fresh birth-only extension at exact density nodes; native electron transport unchanged, only recording reduced'))
if __name__=='__main__':main()
