"""Pack validated active-step references; FP32 acceptance is a separate C++ gate."""
import argparse
from array import array
import csv
import hashlib
import json
from pathlib import Path
import struct

MATERIALS=['Water_75eV','M0','M1','M8','M20']

def read_table(path):
    meta={}
    f=path.open()
    while True:
        line=f.readline()
        if not line.startswith('#'):break
        key,_,value=line[2:].strip().partition(' ');meta[key]=value
    return meta,csv.DictReader(f,fieldnames=line.strip().split(',')),f

def main():
    ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--edges',type=Path,required=True)
    ap.add_argument('--materials',nargs='+',default=MATERIALS)
    ap.add_argument('--species-count',type=int,default=52)
    args=ap.parse_args();root=args.campaign;out=args.output
    state=json.loads((root/'reference_status.json').read_text())
    assert state['status']=='COMPLETE' and not state['failed']
    dense={(r['z'],r['a']):r for r in state['completed'] if r['phase']=='dense'}
    validation={(r['z'],r['a']):r for r in state['completed'] if r['phase']=='validation'}
    assert dense.keys()==validation.keys() and len(dense)==args.species_count
    edge_state=json.loads((args.edges/'reference_status.json').read_text())
    assert edge_state['status']=='COMPLETE' and not edge_state['failed']
    edges={(r['z'],r['a']):r for r in edge_state['completed']}
    assert edges.keys()==dense.keys()
    ions=sorted(dense);out.mkdir(exist_ok=False)
    nodes=array('f');mfps=array('f');segments=array('f');records=[];materials=[];species=[]
    source_files=[];nprobe=0
    with (out/'probes.bin').open('wb') as probes, (out/'mfp_probes.bin').open('wb') as mp:
      for mi,m in enumerate(args.materials):
       for si,(z,a) in enumerate(ions):
        case=dense[z,a];check=validation[z,a]
        for case_info in [case,check]:
         for key in [m,m+'_vectors',m+'_mfp']:
          info=case_info['files'][key];p=Path(info['path'])
          assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
          source_files.append(info)
        path=Path(case['files'][m]['path']);meta,rows,f=read_table(path)
        assert meta['ORACLE_STATUS']=='VALID_ACTIVE_ION_STEP_CONTEXT'
        assert meta['loss_context']=='native_splines_fixed_step_factor_v2'
        assert int(meta['particle_z'])==z and int(meta['particle_a'])==a
        mat=[int(meta['section'])]+[float(meta[k]) for k in ['density_g_cm3','zeff','radlen_mm','production_cut_mm']]
        if si==0:materials.append(mat)
        else:assert mat==materials[mi]
        ion=[z,a,float(meta['particle_mass_mev'])]
        if mi==0:species.append(ion)
        else:assert ion==species[si]
        no=len(nodes)//2;nf=len(mfps)//2
        for r in rows:nodes.extend([float(r['energy_mev']),float(r['factor'])])
        f.close();nc=len(nodes)//2-no
        edge_info=edges[z,a]['files'][m+'_mfp_edges']
        assert hashlib.sha256(Path(edge_info['path']).read_bytes()).hexdigest()==edge_info['sha256']
        source_files.append(edge_info)
        mfp_rows=[]
        for info in [case['files'][m+'_mfp'],edge_info]:
         with Path(info['path']).open() as f:
          for r in csv.DictReader(f):mfp_rows.append((float(r['energy_mev']),float(r['lambda_mm'])))
        for row in sorted(mfp_rows):mfps.extend(row)
        fc=len(mfps)//2-nf
        offsets=[];counts=[]
        with Path(case['files'][m+'_vectors']['path']).open() as f:
         raw=list(csv.DictReader(f))
        for k in range(3):
         offsets.append(len(segments)//6)
         for r in raw:
          if int(r['kind'])==k:segments.extend(float(r[key]) for key in ['x0','x1','y0','y_third','y_twothirds','y1'])
         counts.append(len(segments)//6-offsets[-1])
        records.append([float(meta['mass_ratio']),float(meta['minimum_scaled_energy_mev']),no,nc,nf,fc]+offsets+counts)
        _,rows,f=read_table(Path(check['files'][m]['path']))
        keys=['energy_mev','range_mm','dedx_mev_mm','lambda_mm','inverse_mev','factor',
              'query_range_mm','query_dedx_mev_mm','query_energy_mev','query_lambda_mm']
        for r in rows:
         if float(r['energy_mev'])>6500:continue
         probes.write(struct.pack('<II10f',mi,si,*(float(r[k]) for k in keys)));nprobe+=1
        f.close()
        with Path(check['files'][m+'_mfp']['path']).open() as f:
         for r in csv.DictReader(f):
          e=float(r['energy_mev'])
          if e<=6500:mp.write(struct.pack('<II2f',mi,si,e,float(r['lambda_mm'])))
        print('packed',m,z,a,flush=True)
    p=out/'urban_mcs_candidate.bin'
    with p.open('wb') as f:
     f.write(b'URBANV22');f.write(struct.pack('<7I',2,len(materials),len(species),len(nodes)//2,len(mfps)//2,len(segments)//6,1))
     for m in materials:f.write(struct.pack('<i4f',*m))
     for s in species:f.write(struct.pack('<II f',*s))
     for r in records:f.write(struct.pack('<2f10I',*r))
     for arr in [nodes,mfps,segments]:
      if __import__('sys').byteorder!='little':arr.byteswap()
      arr.tofile(f)
    manifest=dict(status='AWAITING_FP32_VALIDATION',package=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
       bytes=p.stat().st_size,record_count=len(records),probe_count=nprobe,source_files=source_files,
       physics='G4EmStandardPhysics_option4 with all transported ions using Urban',
       loss_context='native splines with fixed step-start dynamic factor',materials=materials,species=species)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CANDIDATE',manifest['sha256'],manifest['bytes'],flush=True)
if __name__=='__main__':main()
