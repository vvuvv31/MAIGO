#!/usr/bin/env python3
"""Compile verified TOPAS electronic stopping into a strict SCHNIOSP v1 bank.
No synthetic production values, material aliases or missing-row fallback.
"""
import argparse, csv, hashlib, json, math, struct
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
SECTIONS,ENERGIES=25,60002
E_MIN,E_MAX,E_STEP=.01,6000.11,.1
SPECIES=[(1,1),(1,2),(1,3),(2,3),(2,4),(2,6),(3,6),(3,7),(4,7),(4,9),(4,10),(5,8),(5,10),(5,11),(6,10),(6,11),(6,12),(4,6)]
QUANTITY='unrestricted electronic dE/dx, linear at unit density'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-csv',required=True,type=Path);ap.add_argument('--raw-json',required=True,type=Path)
    ap.add_argument('--campaign-manifest',required=True,type=Path)
    ap.add_argument('--out-bin',required=True,type=Path)
    a=ap.parse_args(); raw=json.loads(a.raw_json.read_text()); campaign=json.loads(a.campaign_manifest.read_text())
    if raw.get('num_sections')!=25 or raw.get('num_energies')!=60002 or raw.get('num_species')!=18:raise ValueError('Dimensions')
    if raw.get('quantity')!='unrestricted electronic dE/dx' or 'geant4-11-03' not in raw.get('geant4_version','').lower():raise ValueError('Quantity/version')
    for key,val in [('energy_min_mevu',E_MIN),('energy_max_mevu',E_MAX),('energy_step_mevu',E_STEP)]:
        if not math.isfinite(raw[key]) or abs(raw[key]-val)>1e-8:raise ValueError('Energy grid')
    if [tuple(x) for x in raw['species_za']]!=SPECIES:raise ValueError('Species registry')
    if sha(campaign['binary'])!=campaign['binary_sha256']:raise ValueError('Extractor changed')
    sections=raw['sections'];rho=np.array([x['density_g_cm3'] for x in sections])
    if [x['section_id'] for x in sections]!=list(range(25)) or not np.all(np.isfinite(rho)&(rho>0)):raise ValueError('Sections/density')
    values=np.full((25,18,60002),np.nan);seen=np.zeros_like(values,dtype=bool);slots={v:k for k,v in enumerate(SPECIES)}
    with a.raw_csv.open() as f:
        for row in csv.DictReader(l for l in f if not l.startswith('#')):
            s=int(row['section_id']);k=slots[(int(row['species_z']),int(row['species_a']))]
            e=float(row['energy_mevu']);density=float(row['density_g_cm3']);lin=float(row['linear_stopping_power_mev_per_mm'])
            if not (0<=s<25) or not all(map(math.isfinite,[e,density,lin])) or lin<=0:raise ValueError('Invalid row')
            i=round((e-E_MIN)/E_STEP)
            if not 0<=i<60002 or abs(e-(E_MIN+i*E_STEP))>1e-6:raise ValueError('Off-grid row')
            if seen[s,k,i]:raise ValueError('Duplicate row')
            if abs(density-rho[s])>1e-9 or row['material_name']!=sections[s]['material_name']:raise ValueError('Mixed material/density')
            seen[s,k,i]=True;values[s,k,i]=lin/density
    if not seen.all():raise ValueError('Incomplete table')
    refpath=REPO/'data/schneider/schneider_stopping_v1.bin'
    with refpath.open('rb') as f:
        header=f.read(44)
        if header[:8]!=b'SCHNSTOP':raise ValueError('C12 reference header')
        ref=np.fromfile(f,'<f8')[25:25+25*4302].reshape(25,4302)
    # Reference includes nuclear stopping; do not hide low-energy differences.
    energy=E_MIN+np.arange(60002)*E_STEP
    relative=np.abs(values[:,16,:4302]-ref)/ref
    reference_energy=energy[:4302]
    check=relative[:,reference_energy>=5]
    worst=float(check.max())
    if worst>.02:raise ValueError(f'C12 E>=5 cross-check failed: {worst}')
    out=a.out_bin;meta=out.with_suffix('.metadata.json')
    if out.exists() or meta.exists():raise ValueError('Refuse overwrite')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:
        f.write(b'SCHNIOSP'+struct.pack('<IIII',1,25,18,60002)+struct.pack('<ddd',E_MIN,E_MAX,E_STEP))
        rho.astype('<f8').tofile(f);values.astype('<f8').tofile(f)
    metadata={'schema_version':1,'format':'binary','data_filename':out.name,'data_sha256':sha(out),'data_size_bytes':out.stat().st_size,
      'sections_count':25,'species_count':18,'energies_count':60002,'species_za':SPECIES,
      'energy_grid_MeV_per_u':{'minimum':E_MIN,'maximum':E_MAX,'step':E_STEP},'quantity':QUANTITY,
      'provenance':'TOPAS/Geant4 G4EmCalculator ComputeElectronicDEDX','geant4_version':raw['geant4_version'],
      'sections':sections,'campaign':campaign,'raw_csv':str(a.raw_csv),'raw_json':str(a.raw_json),
      'raw_csv_sha256':sha(a.raw_csv),'raw_json_sha256':sha(a.raw_json),'compiler_sha256':sha(__file__),
      'c12_cross_check':{'minimum_energy_MeVu':5,'tolerance':.02,'max_relative_difference':worst,
                         'low_energy_max_relative_difference':float(relative[:,reference_energy<5].max()),'reference_sha256':sha(refpath)}}
    meta.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'binary':str(out),'sha256':sha(out),'metadata_sha256':sha(meta),'c12_max_relative_difference':worst}))
if __name__=='__main__':main()
