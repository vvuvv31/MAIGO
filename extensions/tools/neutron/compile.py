#!/usr/bin/env python3
"""Compile completed native collision samples; does NOT certify GPU transport."""
from pathlib import Path
import argparse,csv,hashlib,json,struct
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):
 with p.open() as f:return list(csv.DictReader(f))
def compile_package(raw:Path,out:Path):
 if out.exists():raise ValueError('output already exists')
 material_dtype=np.dtype([('section','<i4'),('density','<f4'),('temperature','<f4'),('excitation','<f4')])
 rate_dtype=np.dtype([(n,'<u4') for n in ['material','node','subtype','event_offset','event_count']]+[('energy','<f4'),('rate','<f4')])
 event_dtype=np.dtype([(n,'<u4') for n in ['rate','product_offset','product_count']]+[(n,'<i4') for n in ['target_z','target_a','status','model']]+[(n,'<f4') for n in ['continuation','dx','dy','dz','deposit','kinetic_gain']])
 product_dtype=np.dtype([(n,'<i4') for n in ['pdg','z','a']]+[(n,'<f4') for n in ['mass','energy','dx','dy','dz','time']])
 materials=[];rates=[];events=[];products=[];models=[];manifest_materials=[];raw_hashes={};audits=[]
 for mi,(name,section) in enumerate([('Water_75eV',-1),('G4_Cu',-2),('G4_AIR',-3)]):
  p=raw/name
  if not (p/'COMPLETE').exists():raise ValueError(f'incomplete native output: {p}')
  rr,ee,pp=[records(p/f'{k}.csv') for k in ['rates','events','products']]
  expected=list(map(int,(p/'COMPLETE').read_text().split()));assert expected==[len(ee),len(pp)]
  provenance=(p/'provenance.txt').read_text();fields=dict(line.split(' ',1) for line in provenance.splitlines() if line.startswith(('density_g_cm3 ','temperature_K ','excitation_eV ')))
  density,temperature,excitation=[float(fields[k]) for k in ['density_g_cm3','temperature_K','excitation_eV']];assert density>0 and temperature>0
  if name=='Water_75eV':assert excitation==75 and density==1
  materials.append((section,density,temperature,excitation));mapping={};eoffset=len(events);poffset=len(products);roffset=len(rates)
  nodes={};seen=set()
  for ri,r in enumerate(rr):
   node=int(r['node']);energy=float(r['energy_MeV']);rate=float(r['rate_per_mm']);subtype=int(r['subtype']);off,count=int(r['event_offset']),int(r['event_count']);assert energy>0 and rate>=0 and off+count<=len(ee) and np.isfinite(energy+rate)
   assert (node,subtype) not in seen;seen.add((node,subtype));assert node not in nodes or nodes[node]==energy;nodes[node]=energy
   assert (rate==0 and count==0) or (rate>0 and count>0)
   for j in range(off,off+count):assert j not in mapping;mapping[j]=roffset+ri;assert int(ee[j]['node'])==node and ee[j]['process']==r['process']
   rates.append((mi,node,subtype,eoffset+off,count,energy,rate))
  assert sorted(nodes)==list(range(len(nodes))) and np.all(np.diff(list(nodes.values()))>0)
  assert len(mapping)==len(ee)
  maxclosure=0.;elastic_gains=[];unexposed=0;next_product=0
  for j,e in enumerate(ee):
   assert int(e['event'])==j
   off,count=int(e['product_offset']),int(e['product_count']);assert off==next_product and off+count<=len(pp);next_product+=count
   model=e['model'];unexposed+=model=='not_exposed_by_process'
   if model not in models:models.append(model)
   continuation,deposit,gain=[float(e[k]) for k in ['continuation_MeV','local_deposit_MeV','kinetic_gain_MeV']];direction=[float(e[k]) for k in ['dx','dy','dz']]
   assert min(continuation,deposit)>=0 and np.isfinite([continuation,deposit,gain,*direction]).all()
   if continuation>0:assert abs(np.linalg.norm(direction)-1)<1e-6
   outgoing=continuation+deposit
   for pi in range(off,off+count):
    a=pp[pi];assert int(a['product'])==pi and int(a['event'])==j
    vals=[float(a[k]) for k in ['mass_MeV','energy_MeV','dx','dy','dz','time_ns']];assert np.isfinite(vals).all() and min(vals[:2])>=0 and abs(np.linalg.norm(vals[2:5])-1)<1e-6
    products.append((int(a['pdg']),int(a['z']),int(a['a']),*vals));outgoing+=vals[1]
   incoming=nodes[int(e['node'])];closure=abs(outgoing-incoming-gain)/max(1,incoming,abs(gain));maxclosure=max(maxclosure,closure);assert closure<1e-10
   if e['process']=='hadElastic':elastic_gains.append(gain)
   events.append((mapping[j],poffset+off,count,int(e['target_z']),int(e['target_a']),int(e['status']),models.index(model),continuation,*direction,deposit,gain))
  assert next_product==len(pp)
  manifest_materials.append(dict(name=name,section=section,density_g_cm3=density,temperature_K=temperature,excitation_eV=excitation,nodes=len(nodes),energy_min_MeV=min(nodes.values()),energy_max_MeV=max(nodes.values()),processes=sorted({r['process'] for r in rr})))
  audits.append(dict(material=name,events=len(ee),products=len(pp),native_kinetic_bookkeeping_relative_max=maxclosure,elastic_kinetic_gain_MeV_min=min(elastic_gains),elastic_kinetic_gain_MeV_max=max(elastic_gains),model_not_exposed_events=unexposed))
  for n in ['rates.csv','events.csv','products.csv','provenance.txt','COMPLETE']:raw_hashes[str(p/n)]=sha(p/n)
 arrays=[np.array(rows,dtype=dtype) for rows,dtype in zip([materials,rates,events,products],[material_dtype,rate_dtype,event_dtype,product_dtype])]
 assert [a.dtype.itemsize for a in arrays]==[16,28,52,36]
 assert all(len(a)<2**32 for a in arrays)
 out.mkdir(parents=True)
 with (out/'collisions.bin').open('wb') as f:
  f.write(struct.pack('<8s7I',b'NEUTV001',1,0,*[len(a) for a in arrays],0))
  for a in arrays:f.write(a.tobytes())
 m=dict(schema='maigo.neutron.native-collisions.v1',status='native_extraction_only',runtime_validated=False,validation_flags=0,reason='Resonance/interpolation, native final-state coverage, boundary transport and photon descendants require independent validation before GPU use.',binary='collisions.bin',binary_sha256=sha(out/'collisions.bin'),record_bytes=dict(material=16,rate=28,event=52,product=36),counts=dict(zip(['materials','rates','events','products'],map(len,arrays))),materials=manifest_materials,models=models,audits=audits,raw_sha256=raw_hashes,kinetic_gain_definition='sum(final kinetic energies)+local deposit-incident kinetic energy; includes Q/thermal terms, not independent Q-value validation')
 (out/'package.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m['counts']))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('raw',type=Path);a.add_argument('output',type=Path);args=a.parse_args();compile_package(args.raw,args.output)
