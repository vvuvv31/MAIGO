"""Frozen double-interface EM-only diagnostic; no patient or production enable.

Reuses validated mass-world 3D scoring templates. No fitted parameters.
Two cases, two seeds, local Slurm: total 96 CPUs / 40 GiB if run together.
"""
import argparse,json,re,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def prepare(source,out):
    m=json.loads((source/'manifest.json').read_text())
    if m['scoring_geometry']!='mass_voxels':raise ValueError('Require mass-world reference')
    base=source/'air_to_tissue';raw=(base/'phantom.cctg').read_bytes()
    magic,version,nx,ny,nz=struct.unpack_from('<5I',raw)
    if (magic,version,nx,ny,nz)!=(0x47544343,3,100,100,130):raise ValueError('Unexpected CCTG')
    out.mkdir(exist_ok=False);cases={}
    for name,segments in [('thin_air',[(100,5),(-1000,2),(100,58)]),
                          ('thin_tissue',[(-1000,40),(100,2),(-1000,23)])]:
        case=out/name;case.mkdir();gpu=case/'gpu';gpu.mkdir()
        densities=[];sections=[]
        for hu,length in segments:
            mat=m['materials'][str(hu)]
            densities.extend([mat['rho']]*(length*2));sections.extend([mat['section']]*(length*2))
        if len(densities)!=nz:raise ValueError('Grid length mismatch')
        (case/'phantom.cctg').write_bytes(raw[:44]+np.repeat(np.array(densities,dtype='<f4'),10000).tobytes()+
            np.repeat(np.array(sections,dtype='u1'),10000).tobytes()+raw[44+nx*ny*nz*5:])
        gt=(base/'gpu/run.yaml').read_text().replace(str(base),str(case))
        (gpu/'run.yaml').write_text(gt)
        for seed in ('s1','s2'):
            old=base/seed;dest=case/seed;dest.mkdir();text=(old/'run.txt').read_text()
            text=re.sub(r'^.*(?:Ge/ZLayer\d+/|Sc/Dose\d+/).*\n','',text,flags=re.M)
            z=0
            for i,(hu,length) in enumerate(segments):
                material='PatientTissueFromHUNegative1000' if hu==-1000 else 'PatientTissueFromHU100'
                component=f'ZLayer{i}'
                text+=f'\ns:Ge/{component}/Parent = "ZPhantom"\ns:Ge/{component}/Type = "TsBox"\ns:Ge/{component}/Material = "{material}"\n'
                for key,value in [('HLX',100),('HLY',100),('HLZ',length/2),('TransZ',z+length/2-32.5)]:
                    text+=f'd:Ge/{component}/{key} = {value} mm\n'
                for prefix in (f'Ge/{component}',f'Sc/Dose{i}'):
                    for axis,bins in [('X',100),('Y',100),('Z',length*2)]:text+=f'i:{prefix}/{axis}Bins = {bins}\n'
                text+=f's:Sc/Dose{i}/Quantity = "DoseToMedium"\ns:Sc/Dose{i}/Component = "{component}"\ns:Sc/Dose{i}/OutputType = "csv"\ns:Sc/Dose{i}/OutputFile = "{dest}/dose{i}"\ns:Sc/Dose{i}/IfOutputFileAlreadyExists = "Exit"\n'
                z+=length
            (dest/'run.txt').write_text(text)
            (dest/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=ordered_{name}_{seed}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=24\n#SBATCH --mem=10G\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\ncd {dest}\ntest ! -e dose0.csv\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n')
        cases[name]=dict(segments=[dict(hu=h,length_mm=l) for h,l in segments],interface_mm=segments[0][1],
                        interfaces_mm=[segments[0][1],segments[0][1]+segments[1][1]])
    report=dict(m,cases=cases,status='DOUBLE_INTERFACE_DIAGNOSTIC_NOT_ACCEPTANCE',
        source_manifest_sha256=sha(source/'manifest.json'),generator_sha256=sha(Path(__file__)),
        topas_binary_sha256=sha(Path('/home/wuwei/topas/topas-build/topas')))
    report['inputs']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    with (out/'manifest.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.source.resolve(),a.out.resolve())
