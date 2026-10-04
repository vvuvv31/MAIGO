#!/usr/bin/env python3
"""Pin the proton package and inspect birth spectra; no dose approximation."""
import argparse
import json
from pathlib import Path
import numpy as np
from macro_model import DTYPE as RAW_DTYPE
from prototype import sha, save, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tables',nargs='+',default=['s1_mid9','s5_0','s19_1'])
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ip=ROOT/'data/ct_electron/proton/index.json'; index=json.loads(ip.read_text())
    assert index['projectile']==[1,1]
    files={}; tables=[]
    for table in index['tables']:
        folder=ip.parent/Path(table['path']).parent
        for file in folder.iterdir():
            if file.is_file(): files[str(file)]=file.stat().st_size
        meta=json.loads((folder/'response.metadata.json').read_text())
        refs=json.loads((folder/'state_refs.metadata.json').read_text())
        assert sha(folder/'response.metadata.json')==table['metadata_sha256']
        assert sha(folder/'state_refs.metadata.json')==table['state_reference_metadata_sha256']
        tables.append(dict(name=folder.name,section=table['material_identity']['section_id'],
                           density=table['material_identity']['runtime_density_g_cm3'],
                           extraction_energy_max_MeVu=meta['nominal_energy_max_MeVu'],
                           raw_rows=sum(s['rows'] for s in refs['sources']),
                           channels=len(meta['channels'])))
    dtype=np.dtype(dict(names=list(RAW_DTYPE.names)+['parent_ke'],
                        formats=[RAW_DTYPE.fields[n][0] for n in RAW_DTYPE.names]+['<f8'],
                        offsets=[RAW_DTYPE.fields[n][1] for n in RAW_DTYPE.names]+[152],itemsize=312))
    spectra=[]
    bins=np.array([0,.01,.025,.05,.1,.2,.3,.4,.45,.5,.65,1.])
    for name in a.tables:
        table=next(t for t in index['tables'] if Path(t['path']).parent.name==name)
        folder=ip.parent/name
        refs=json.loads((folder/'state_refs.metadata.json').read_text()); chunks=[]; parents=set(); identities=[]
        for source in refs['sources']:
            path=Path(source['path']); path=path if path.is_absolute() else folder/path
            assert sha(path)==source['sha256'] and path.stat().st_size==312*source['rows']
            raw=np.memmap(path,dtype,mode='r')
            primary=(raw['step']==1)&(raw['parent']==0)
            assert np.all(raw['pdg'][primary]==2212) and np.all(raw['track'][primary]==1)
            parents.update(zip(raw['run'][primary].tolist(),raw['event'][primary].tolist()))
            selected=(raw['step']==1)&(raw['pdg']==11)&(raw['parent']==1)&(raw['parent_ke']<=170.)
            chunks.append(raw['birth'][selected].copy())
            identities.append(dict(path=str(path),sha256=source['sha256']))
        energy=np.concatenate(chunks); assert len(energy)>0
        order=np.argsort(energy); cumulative=np.cumsum(energy[order])/energy.sum()
        quantiles=energy[order][np.searchsorted(cumulative,[.5,.9,.99])]
        counts=np.histogram(energy,bins)[0]; mass=np.histogram(energy,bins,weights=energy)[0]
        assert int(counts.sum())==len(energy) and abs(mass.sum()-energy.sum())<1e-9
        spectra.append(dict(table=name,primary_histories=len(parents),root_electrons=len(energy),
                     source_condition='Direct proton ionization electron birth; recorded parent pre-step KE <=170 MeV',
                     electron_max_MeV=float(energy.max()),birth_energy_sum_MeV=float(energy.sum()),
                     birth_energy_weighted_q50_q90_q99_MeV=quantiles.tolist(),
                     bin_edges_MeV=bins.tolist(),count=counts.tolist(),birth_energy_fraction=(mass/mass.sum()).tolist(),
                     raw_sources=identities))
    me=.51099895; mp=938.27208816; gamma=1+170/mp
    tmax=2*me*(gamma*gamma-1)/(1+2*gamma*me/mp+(me/mp)**2)
    save(a.out/'inventory.json',dict(status='PROTON_PACKAGE_AND_EXTRACTION_SPECTRA_ONLY',ct_acceptance=False,
        index_sha256=sha(ip),config_sha256=sha(ROOT/'config/proton_ct_minibeam.yaml'),
        tables=tables,table_count=len(tables),all_package_files_bytes=sum(files.values()),
        raw_files_bytes=sum(v for k,v in files.items() if k.endswith('steps.phsp')),
        response_files_bytes=sum(v for k,v in files.items() if k.endswith('response.bin')),
        proton_170MeV_free_electron_Tmax_MeV=tmax,spectra=spectra,
        limitations='Extraction-bank spectra, not actual CT dose-weighted runtime spectrum; parent pre-step KE is not the internal interaction KE. No interface, dose, or speed acceptance.',
        script_sha256=sha(Path(__file__))))
    print('tables',len(tables),'package_GiB',sum(files.values())/2**30,'Tmax',tmax)
    for s in spectra: print(s['table'],s['root_electrons'],s['electron_max_MeV'],s['birth_energy_weighted_q50_q90_q99_MeV'])


if __name__=='__main__':
    main()
