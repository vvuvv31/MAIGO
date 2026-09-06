"""Independent bulk-source response pilot. No interface-dose fitting or package output."""
import argparse,json,re
from pathlib import Path
from analyze_longitudinal_holdout import sha

def prepare(template,out,air_seeds=(9181751,9181752),tissue_seeds=(9181751,9181752),mem_gib=8,beam_mevu=175):
    out.mkdir(parents=True,exist_ok=False)
    source=template.read_text()
    source=re.sub(r'^.*(?:Ge/ZPhantom/|Ge/ZLayer\d+/|Sc/Dose\d+/).*\n','',source,flags=re.M)
    source=source.replace('NumberOfHistoriesInRun = 20000','NumberOfHistoriesInRun = 4')
    source=source.replace('BeamEnergySpread = 1.0','BeamEnergySpread = 0.0')
    if beam_mevu not in (175,185):raise ValueError('Only predeclared pilot energies 175/185 permitted')
    source=source.replace('BeamEnergy = 2100.0 MeV',f'BeamEnergy = {beam_mevu*12}.0 MeV')
    total_cpus=4*len(air_seeds)+2*len(tissue_seeds)
    total_memory=mem_gib*(len(air_seeds)+len(tissue_seeds))
    if total_cpus>192 or total_memory>160:raise ValueError('Campaign resource budget exceeded')
    if len(set(air_seeds))!=len(air_seeds) or len(set(tissue_seeds))!=len(tissue_seeds):raise ValueError('Duplicate seeds')
    manifest=dict(status='INDEPENDENT_RESPONSE_PILOT',parent_energy_window_MeVu=[170,175.01],
        histories_per_job=4,scope='C12 electron families born in bulk air or soft tissue; finite-box escape must be audited',
        beam_MeVu=beam_mevu,max_total_cpus=total_cpus,max_total_mem_GiB=total_memory,cases={},inputs={str(template):sha(template)})
    for name,hu,halfxy,halfz,nxy,nz,start,cpus in [('air',-1000,150,400,60,400,200,4),('tissue',100,50,40,50,160,10,2)]:
        if beam_mevu==185 and name=='air':halfz,nz=1100,1100
        mat='PatientTissueFromHUNegative1000' if hu<0 else 'PatientTissueFromHU100'
        for seed in (air_seeds if name=='air' else tissue_seeds):
            dest=out/name/str(seed);dest.mkdir(parents=True)
            t=source.replace('Ts/NumberOfThreads = 24',f'Ts/NumberOfThreads = {cpus}')
            t=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {seed}',t)
            t=t.replace('d:Ge/BeamPosition/TransZ = 0.01 mm',f'd:Ge/BeamPosition/TransZ = {start} mm')
            t+='\nd:Ph/Default/CutForAllParticles = 0.05 mm\n'
            t+=f's:Ge/ZResponse/Parent = "World"\ns:Ge/ZResponse/Type = "TsBox"\ns:Ge/ZResponse/Material = "{mat}"\nd:Ge/ZResponse/HLX = {halfxy} mm\nd:Ge/ZResponse/HLY = {halfxy} mm\nd:Ge/ZResponse/HLZ = {halfz} mm\nd:Ge/ZResponse/TransZ = {halfz} mm\ni:Ge/ZResponse/XBins = {nxy}\ni:Ge/ZResponse/YBins = {nxy}\ni:Ge/ZResponse/ZBins = {nz}\n'
            t+=f's:Sc/Dose3D/Quantity = "DoseToMedium"\ns:Sc/Dose3D/Component = "ZResponse"\ni:Sc/Dose3D/XBins = {nxy}\ni:Sc/Dose3D/YBins = {nxy}\ni:Sc/Dose3D/ZBins = {nz}\ns:Sc/Dose3D/OutputType = "csv"\ns:Sc/Dose3D/OutputFile = "{dest}/dose"\n'
            # Homogeneous component: both scorers attach to its own voxel
            # structure, no child components. TOPAS requires matching flags.
            t+=f's:Sc/ElectronDeposit/Quantity = "CarbonElectronDepositNtupleV3"\ns:Sc/ElectronDeposit/Component = "ZResponse"\nb:Sc/ElectronDeposit/PropagateToChildren = "False"\ns:Sc/ElectronDeposit/OutputType = "Binary"\ns:Sc/ElectronDeposit/OutputFile = "{dest}/steps"\n'
            (dest/'run.txt').write_text(t)
            (dest/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=bulk_el_{name}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem_gib}G\n#SBATCH --time=00:20:00\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\nulimit -f 1048576\ncd {dest}\ntest ! -e steps.phsp\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n')
            manifest['cases'][name+'/'+str(seed)]=dict(hu=hu,source_z_mm=start,
                bounds_mm=[[-halfxy,halfxy],[-halfxy,halfxy],[0,2*halfz]],dims=[nxy,nxy,nz])
    manifest['inputs'].update({str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    for f in ['/home/wuwei/topas/topas-build/topas','/home/wuwei/topas/extensions/CarbonElectronDepositNtupleV3.cc','/home/wuwei/topas/extensions/CarbonElectronDepositNtupleV3.hh']:
        manifest['inputs'][f]=sha(Path(f))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('template',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--air-seeds',nargs='+',type=int,default=[9181751,9181752]);p.add_argument('--tissue-seeds',nargs='+',type=int,default=[9181751,9181752])
    p.add_argument('--mem-gib',type=int,default=8)
    p.add_argument('--beam-mevu',type=int,default=175)
    a=p.parse_args();prepare(a.template.resolve(),a.out.resolve(),a.air_seeds,a.tissue_seeds,a.mem_gib,a.beam_mevu)
