"""Prepare reference-only air/tissue interfaces; no GPU candidate route enabled."""
import argparse,json,re,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def material(hu,table):
    text=table.read_text()
    def vec(name):
        line=next(x for x in text.splitlines() if re.match(r"^\w+:Ge/Patient/"+name+r"\s*=",x))
        p=line.split("=",1)[1].split()
        return np.array([float(x) for x in p[1:1+int(p[0])]])
    e=vec("SchneiderHounsfieldUnitSections")
    i=int(np.searchsorted(e,hu,side="right")-1)
    rho=(vec("SchneiderDensityOffset")[i]+vec("SchneiderDensityFactor")[i]*(vec("SchneiderDensityFactorOffset")[i]+hu))*vec("DensityCorrection")[hu+1000]
    sec=int(np.searchsorted(vec("SchneiderHUToMaterialSections"),hu,side="right")-1)
    return float(np.float32(rho)),sec

def prepare(template,grid,table,root,buffer_mm=0,mass_scoring=False):
    if buffer_mm < 0 or buffer_mm != int(buffer_mm):
        raise ValueError("buffer_mm must be a nonnegative integer")
    root.mkdir(parents=True,exist_ok=False)
    raw=grid.read_bytes()
    if struct.unpack_from("<5I",raw)!=(0x47544343,3,100,100,440):raise ValueError("Grid template mismatch")
    footer=raw[44+4400000*5:]
    length_mm=45+buffer_mm
    nz=length_mm*2
    header=bytearray(raw[:44]);struct.pack_into("<I",header,16,nz)
    mats={h:material(h,table) for h in (-1000,100)}
    manifest=dict(status="BASELINE_INTERFACE_DIAGNOSTIC_ONLY",histories=20000,energy_MeVu=175,
        scale=1,materials={str(h):dict(rho=r,section=s) for h,(r,s) in mats.items()},
        downstream_material_buffer_mm=buffer_mm,
        scoring_geometry="mass_voxels" if mass_scoring else "parallel_worlds",
        cases={},geometry_template_sha256=sha(grid),schneider_sha256=sha(table))
    for name,segments in [("air_to_tissue",[(-1000,40),(100,5+buffer_mm)]),("tissue_to_air",[(100,5),(-1000,40+buffer_mm)])]:
        case=root/name;case.mkdir()
        density=[];sections=[]
        for hu,length in segments:
            density.extend([mats[hu][0]]*int(length*2))
            sections.extend([mats[hu][1]]*int(length*2))
        (case/"phantom.cctg").write_bytes(bytes(header)+np.repeat(np.array(density,dtype="<f4"),10000).tobytes()+np.repeat(np.array(sections,dtype="u1"),10000).tobytes()+footer)
        gt=(template/"gpu_base/run.yaml").read_text()
        gt=gt.replace(str(template/"gpu_base"),str(case/"gpu")).replace(str(grid),str(case/"phantom.cctg"))
        gt=gt.replace("voxel_bins_z: 440",f"voxel_bins_z: {nz}").replace("phantom_length_mm: 220.0",f"phantom_length_mm: {length_mm}.0")
        (case/"gpu").mkdir();(case/"gpu/run.yaml").write_text(gt)
        for rep in ("s1","s2"):
            dest=case/rep;dest.mkdir()
            t=(template/f"topas_{rep}/run.txt").read_text()
            t=re.sub(r'^.*Ge/Slab/.*\n','',t,flags=re.M)
            t=re.sub(r'^.*Sc/Dose3D/.*\n','',t,flags=re.M)
            t+='\ns:Ge/ZPhantom/Parent = "World"\ns:Ge/ZPhantom/Type = "TsBox"\ns:Ge/ZPhantom/Material = "Vacuum"\nd:Ge/ZPhantom/HLX = 100 mm\nd:Ge/ZPhantom/HLY = 100 mm\nd:Ge/ZPhantom/HLZ = 22.5 mm\nd:Ge/ZPhantom/TransZ = 22.5 mm\n'
            t=t.replace('HLX = 100 mm','HLX = 101 mm').replace('HLY = 100 mm','HLY = 101 mm')
            t=t.replace('HLZ = 22.5 mm',f'HLZ = {length_mm/2+1} mm').replace('TransZ = 22.5 mm',f'TransZ = {length_mm/2} mm')
            z=0
            for index,(hu,length) in enumerate(segments):
                comp=f"ZLayer{index}"
                mat="PatientTissueFromHUNegative1000" if hu==-1000 else "PatientTissueFromHU100"
                t+=f'\ns:Ge/{comp}/Parent = "ZPhantom"\ns:Ge/{comp}/Type = "TsBox"\ns:Ge/{comp}/Material = "{mat}"\nd:Ge/{comp}/HLX = 100 mm\nd:Ge/{comp}/HLY = 100 mm\nd:Ge/{comp}/HLZ = {length/2} mm\nd:Ge/{comp}/TransZ = {z+length/2-length_mm/2} mm\n'
                if mass_scoring:
                    t+=f'i:Ge/{comp}/XBins = 100\ni:Ge/{comp}/YBins = 100\ni:Ge/{comp}/ZBins = {length*2}\n'
                t+=f's:Sc/Dose{index}/Quantity = "DoseToMedium"\ns:Sc/Dose{index}/Component = "{comp}"\ni:Sc/Dose{index}/XBins = 100\ni:Sc/Dose{index}/YBins = 100\ni:Sc/Dose{index}/ZBins = {length*2}\ns:Sc/Dose{index}/OutputFile = "{dest}/dose{index}"\ns:Sc/Dose{index}/OutputType = "csv"\ns:Sc/Dose{index}/IfOutputFileAlreadyExists = "Overwrite"\n'
                z+=length
            (dest/"run.txt").write_text(t)
            job=f'#!/bin/bash\n#SBATCH --job-name=iface_{name}_{rep}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=24\n#SBATCH --mem=10G\n#SBATCH --output={dest}/job_%j.log\n#SBATCH --error={dest}/job_%j.err\nset -euo pipefail\ncd {dest}\ntest ! -e dose0.csv\ntest ! -e dose1.csv\n/home/wuwei/topas/topas-build/topas {dest}/run.txt\n'
            (dest/"run.slurm").write_text(job)
        manifest["cases"][name]=dict(segments=[dict(hu=h,length_mm=l) for h,l in segments],interface_mm=segments[0][1])
    manifest["inputs"]={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("template","grid","table","out"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--buffer-mm",type=int,default=0)
    p.add_argument("--mass-scoring",action="store_true")
    a=p.parse_args()
    prepare(a.template.resolve(),a.grid.resolve(),a.table.resolve(),a.out.resolve(),a.buffer_mm,a.mass_scoring)
