#!/usr/bin/env python3
"""Assemble a proton YAML and full runtime bank with explicit reuse provenance."""
import argparse,json,shutil,struct
from pathlib import Path
import numpy as np
from campaign import REPO,sha,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();work=a.work.resolve();shared=out/'shared'
    # Preserve all 18 projectile slots; replace only the freshly extracted proton.
    old=shared/'all_ion_elastic_v1.bin';raw=work/'elastic/raw/z1a1.bin';new=out/'all_ion_elastic_proton_v1.bin'
    pins=json.loads((work/'elastic/provenance.json').read_text())
    for fn,h in pins.items():
        if sha(Path(fn))!=h:raise ValueError('Elastic provenance changed: '+fn)
    with open(old,'rb') as f:
        magic=f.read(8);ver,np_,ns,nt,ne,nq=struct.unpack('<6I',f.read(24))
        if magic!=b'ELBANK01' or (ver,np_,ns,nt)!=(1,18,26,13):raise ValueError('Elastic schema')
        masses=np.fromfile(f,'<f8',18);grid=np.fromfile(f,'<f4',ne)
    dtype=np.dtype([('fraction','<f4'),('mass','<f4'),('a','<u4')])
    with open(raw,'rb') as f:
        if f.read(8)!=b'ELRAW001':raise ValueError('Elastic raw magic')
        if struct.unpack('<6I',f.read(24))!=(1,1,ns,nt,ne,nq):raise ValueError('Elastic raw dimensions')
        mass=struct.unpack('<d',f.read(8))[0];eg=np.fromfile(f,'<f8',ne)
        rates=np.fromfile(f,'<f8',ns*nt*ne).reshape(ns,nt,ne);samples=np.fromfile(f,dtype,nt*ne*nq).reshape(nt,ne,nq)
        if f.read(1) or not np.array_equal(grid,eg.astype('<f4')):raise ValueError('Elastic grid/size')
    if not np.all(np.isfinite(rates)&(rates>=0)) or not np.all(np.isfinite(samples['fraction'])&(samples['fraction']>=0)&(samples['fraction']<=1)):raise ValueError('Elastic nonfinite values')
    for ti in range(nt):
        for ei in range(ne):
            if np.any(rates[:,ti,ei]>0) and np.any(samples['a'][ti,ei]==0):raise ValueError('Elastic rate without target samples')
    shutil.copy2(old,new)
    with open(new,'r+b') as f:
        f.seek(32);f.write(struct.pack('<d',mass));f.seek(32+18*8+ne*4);f.write(rates.astype('<f4').tobytes())
        f.seek(32+18*8+ne*4+18*ns*nt*ne*4);f.write(samples.tobytes())
    meta=json.loads(old.with_suffix('.metadata.json').read_text());meta.update(sha256=sha(new),bytes=new.stat().st_size,
        new_proton_raw={str(raw):sha(raw)},new_proton_input_pins=pins,parent_bank_sha256=sha(old),
        water_reference='Proton row: Water_75eV; other projectile rows retained from parent bank',
        scope='New proton elastic row; 17 unchanged projectile rows from pinned reference extraction')
    dump(new.with_suffix('.metadata.json'),meta)
    recoil=work/'recoil/raw/recoil_stopping.bin';dest=out/'elastic_recoil_stopping_water75.bin'
    rp=json.loads((work/'recoil/provenance.json').read_text())
    for fn,h in rp.items():
        if sha(Path(fn))!=h:raise ValueError('Recoil provenance changed')
    shutil.copy2(recoil,dest)
    with open(dest,'rb') as f:
        if f.read(8)!=b'ELRSP001':raise ValueError('Recoil format')
        ni,nm,nenergy=struct.unpack('<3I',f.read(12));f.read(ni*8+nenergy*4);vals=np.fromfile(f,'<f4')
        if nm!=26 or len(vals)!=ni*nm*nenergy or not np.all(np.isfinite(vals)&(vals>0)):raise ValueError('Recoil payload')
    dump(dest.with_suffix('.metadata.json'),dict(format='ELRSP001',sha256=sha(dest),water_material='Water_75eV',mean_excitation_energy_eV=75,raw_sha256=sha(recoil),input_pins=rp))
    raw_water=work/'tables/proton_water.phsp';values=np.loadtxt(raw_water)
    if values.shape!=(4001,5) or not np.isfinite(values).all() or not np.all(values[:,2]>0):raise ValueError('Water stopping raw')
    np.savetxt(out/'proton_water_stopping.csv',values[:,[0,2]],delimiter=',',fmt='%.12g',header='Water_75eV; proton unrestricted electronic stopping\nenergy_MeVu,stopping_power_MeV_per_mm')
    dump(out/'proton_water_stopping.metadata.json',dict(projectile={'z':1,'a':1},water_material='Water_75eV',mean_excitation_energy_eV=75,raw_sha256=sha(raw_water),data_sha256=sha(out/'proton_water_stopping.csv')))
    for name in ['ion_stopping_power_water_geant4_11_3_2.csv']:
        shutil.copy2(REPO/'data'/name,shared/name)
    shutil.copy2(REPO/'data/water_unified/g4_water_material.json',shared/'g4_water_material.json')
    text=(REPO/'config/proton_source_gpu.yaml.template').read_text()
    paths=dict(PROTON_WATER_STOPPING_CSV=out/'proton_water_stopping.csv',ION_WATER_STOPPING_CSV=shared/'ion_stopping_power_water_geant4_11_3_2.csv',
        PROTON_PRIMARY_RATE=out/'proton_rates.bin',PROTON_PRIMARY_CINEL03=out/'proton_cinel03.bin',
        SECONDARY_RATE=shared/'secondary_inelastic_rates_v2_1.bin',SECONDARY_CINEL03=shared/'cinel03_secondary_targets_v2_1_14p.bin',
        PROTON_SCHNEIDER_STOPPING=out/'proton_schneider_stopping_v1.bin',PROTON_PHYSICS_BUNDLE=out/'physics_bundle.json',
        EM_DELTA_MOMENTS=shared/'unified_em_delta_moments_v2.bin',EM_PACKAGE=shared/'unified_em_v1.bin',ELASTIC_PACKAGE=new,RECOIL_STOPPING=dest)
    hashes=dict(EM_SHA256=sha(paths['EM_PACKAGE']),DELTA_MOMENTS_SHA256=sha(paths['EM_DELTA_MOMENTS']),ELASTIC_SHA256=sha(new),RECOIL_STOPPING_SHA256=sha(dest))
    for k,v in {**paths,**hashes}.items():text=text.replace('@'+k+'@',str(v))
    for key in ['all_ion_elastic_file','all_ion_elastic_sha256','elastic_recoil_stopping_file','elastic_recoil_stopping_sha256']:text=text.replace('# '+key+':',key+':')
    text=text.replace('data/water_unified/g4_water_material.json',str(shared/'g4_water_material.json'))
    text=text.replace('phantom_length_mm: 200','phantom_length_mm: 450').replace('voxel_bins_z: 400','voxel_bins_z: 900').replace('beam_energy_spread: 0.01','beam_energy_spread: 0.0')
    text='\n'.join(line for line in text.splitlines() if not line.startswith('#'))+'\n'
    if '@' in text:raise ValueError('Unresolved template token')
    (out/'proton_fullphysics.yaml').write_text('# Proton 0.1–250 MeV; non-minibeam, Water_75eV. Research package; dose accuracy not yet accepted.\n'+text)
    print(out/'proton_fullphysics.yaml')
if __name__=='__main__':main()
