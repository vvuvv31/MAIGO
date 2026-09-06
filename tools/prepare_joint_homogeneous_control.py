"""Matched homogeneous controls for the joint interface diagnostic, not a fit.

Keep reference geometry/source/scorers; change BOTH layers to the same material.
The two-layer boundary remains, isolating composition from numerical geometry.
"""
import argparse,json,re,struct
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def prepare(reference,out):
    m=json.loads((reference/'manifest.json').read_text())
    for name,pin in m['inputs'].items():
        if sha(reference/name)!=pin:raise ValueError('Changed reference input')
    out.mkdir(exist_ok=False)
    for name,c in m['cases'].items():
        hu=-1000 if name=='air_to_tissue' else 100
        rho=m['materials'][str(hu)]['rho'];section=m['materials'][str(hu)]['section']
        source=reference/name;dest=out/name;dest.mkdir()
        raw=(source/'phantom.cctg').read_bytes()
        magic,version,nx,ny,nz=struct.unpack_from('<5I',raw)
        if (magic,version,nx,ny)!=(0x47544343,3,100,100):raise ValueError('Grid schema')
        count=nx*ny*nz
        (dest/'phantom.cctg').write_bytes(raw[:44]+np.full(count,rho,dtype='<f4').tobytes()+np.full(count,section,dtype='u1').tobytes()+raw[44+count*5:])
        (dest/'gpu').mkdir()
        (dest/'gpu/run.yaml').write_text((source/'gpu/run.yaml').read_text().replace(str(source),str(dest)))
        for seed in ('s1','s2'):
            folder=dest/seed;folder.mkdir()
            text=(source/seed/'run.txt').read_text().replace(str(source),str(dest))
            material='PatientTissueFromHUNegative1000' if hu==-1000 else 'PatientTissueFromHU100'
            text=re.sub(r'(s:Ge/ZLayer[01]/Material = )"[^"]+"',lambda x:x[1]+'"'+material+'"',text)
            (folder/'run.txt').write_text(text)
            (folder/'run.slurm').write_text((source/seed/'run.slurm').read_text().replace(str(source),str(dest)).replace('--job-name=iface_','--job-name=joint_control_'))
        for s in c['segments']:s['hu']=hu
    m['status']='HOMOGENEOUS_JOINT_CONTROL_ONLY'
    m['inputs']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    m['parent_manifest_sha256']=sha(reference/'manifest.json')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.reference.resolve(),a.out.resolve())
