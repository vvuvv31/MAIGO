"""Minimal 25-material full-slowing C12 response campaign for CT Gamma A/B.

500 MeV/u covers patient energy spread without upper extrapolation. Four
histories/material are exploratory data, NOT a validated production package.
All jobs together: 100 CPUs / 100 GiB; local Slurm only, 3D DoseToMedium.
"""
import argparse,json,re
from pathlib import Path
from audit_schneider_response_scope import schneider_identity
from analyze_longitudinal_holdout import sha

def prepare(template,schneider,out):
    text=schneider.read_text();line=next(l for l in text.splitlines() if 'SchneiderHUToMaterialSections =' in l)
    bounds=list(map(int,line.split('=',1)[1].split()[1:]))
    if len(bounds)!=26:raise ValueError('Require 25 Schneider materials')
    source=template.read_text()
    source=re.sub(r'^.*(?:Ge/ZPhantom/|Ge/ZLayer\d+/|Sc/Dose\d+/).*\n','',source,flags=re.M)
    source=source.replace('NumberOfHistoriesInRun = 20000','NumberOfHistoriesInRun = 4')
    source=source.replace('BeamEnergySpread = 1.0','BeamEnergySpread = 0.0').replace('BeamEnergy = 2100.0 MeV','BeamEnergy = 6000.0 MeV')
    source=source.replace('Ts/NumberOfThreads = 24','Ts/NumberOfThreads = 4')
    for axis in 'XYZ':source=re.sub(r'^d:Ge/World/HL'+axis+r' = .*$',f'd:Ge/World/HL{axis} = 1000 m',source,flags=re.M)
    source=re.sub(r'^d:Ge/Patient/TransX = .*$', 'd:Ge/Patient/TransX = -900 m',source,flags=re.M)
    out.mkdir(exist_ok=False);cases={}
    for section,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        hu=-1000 if section==0 else 100 if section==8 else (lo+hi-1)//2
        identity=schneider_identity(schneider,hu);rho=identity['density_g_cm3']
        if identity['material_section']!=section:raise ValueError('Section identity')
        dest=out/f'section_{section:02d}'/'9182501';dest.mkdir(parents=True)
        inset=5/rho;start=10/rho;length=1210/rho;halfxy=1000/rho
        material=f'PatientTissueFromHU{"Negative"+str(-hu) if hu<0 else str(hu)}'
        t=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {9182501+section}',source)
        t=t.replace('d:Ge/BeamPosition/TransZ = 0.01 mm',f'd:Ge/BeamPosition/TransZ = {start} mm')
        t+='\nd:Ph/Default/CutForAllParticles = 0.05 mm\n'
        t+=f's:Ge/ZResponse/Parent = "World"\ns:Ge/ZResponse/Type = "TsBox"\ns:Ge/ZResponse/Material = "{material}"\n'
        for key,value in [('HLX',halfxy),('HLY',halfxy),('HLZ',length/2),('TransZ',length/2)]:t+=f'd:Ge/ZResponse/{key} = {value} mm\n'
        for prefix in ('Ge/ZResponse','Sc/Dose3D'):
            for axis,n in [('X',20),('Y',20),('Z',2420)]:t+=f'i:{prefix}/{axis}Bins = {n}\n'
        t+=f's:Sc/Dose3D/Quantity = "DoseToMedium"\ns:Sc/Dose3D/Component = "ZResponse"\ns:Sc/Dose3D/OutputType = "csv"\ns:Sc/Dose3D/OutputFile = "{dest}/dose"\n'
        t+=f's:Sc/ElectronDeposit/Quantity = "CarbonElectronDepositNtupleV3"\ns:Sc/ElectronDeposit/Component = "ZResponse"\nb:Sc/ElectronDeposit/PropagateToChildren = "False"\ns:Sc/ElectronDeposit/OutputType = "Binary"\ns:Sc/ElectronDeposit/OutputFile = "{dest}/steps"\n'
        (dest/'run.txt').write_text(t)
        (dest/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=ct_el_s{section:02d}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=4G\n#SBATCH --time=01:00:00\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\nulimit -f 2097152\ncd {dest}\ntest ! -e steps.phsp\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n')
        cases[str(section)]=dict(identity,section_id=section,path=str(dest/'steps.phsp'),parent_inset_mm=inset,
            energy_max_MeVu=500,histories=4,cpus=4,mem_GiB=4,slurm_file=str(dest/'run.slurm'))
    inputs={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    for p in (template,schneider,Path('/home/wuwei/topas/topas-build/topas'),Path('/home/wuwei/topas/extensions/CarbonElectronDepositNtupleV3.cc'),Path(__file__)):inputs[str(p)]=sha(p)
    with (out/'manifest.json').open('x') as f:json.dump(dict(status='UNVALIDATED_CT_RESPONSE_CAMPAIGN',max_cpus=100,max_mem_GiB=100,cases=cases,inputs=inputs),f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('template',type=Path);p.add_argument('schneider',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.template.resolve(),a.schneider.resolve(),a.out.resolve())
