#!/usr/bin/env python3
"""Derive C12 birth-only law from pinned validated v7 cache, never proton births.

Preserves the original 5 MeV/u channels, energy-weighted CDF and every birth
in channels0..50. The last channel250..255 covers the original exact250 query.
Supported incident primary is <=250 MeV/u; all exported electrons must <=.65MeV.
No thinning or reweighting. Legacy trajectories are offline inputs only.
"""
from pathlib import Path
import argparse,json,hashlib,struct
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
CH=np.dtype([('low','<f8'),('high','<f8'),('fraction','<f8'),('offset','<u8'),('count','<u8')])
BS=np.dtype([('cdf','<f8'),('row','<u8'),('source','<u4'),('padding','<u4')])
ER=np.dtype([('position','<f8',3),('direction','<f8',3),('energy','<f8'),('deposit','<f8'),('length','<f8'),('track','<u4'),('medium','<u4'),('step','<i4'),('reserved','<u4'),('post','<u4'),('padding','<u4')])
TR=np.dtype([('direction','<f8',3),('birth','<f8'),*[(k,'<i4') for k in ['run','event','track','parent','pdg','parent_valid','parent_step','reserved']]])
assert (CH.itemsize,BS.itemsize,ER.itemsize,TR.itemsize)==(40,24,96,64)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 index=ROOT/'data/ct_electron/carbon/index.json';data=json.loads(index.read_text());assert data['projectile']==[6,12] and len(data['tables'])==63
 by_material={}
 for d in (ROOT/'data/ct_electron/carbon_compact').iterdir():
  if not d.is_dir():continue
  manifest=d/'manifest';pin=(d/'manifest.sha256').read_text().split();assert len(pin)==1 and sha(manifest)==pin[0]
  lines=manifest.read_text().splitlines();h=lines[0].split();assert h[0]=='MAIGO_CARBON_EM_7' and h[1]==d.name and int(h[6])==1
  key=(int(h[2]),float(h[3]));assert key not in by_material;by_material[key]=(d,lines,pin[0])
 output=a.out/'birth_bank.bin';results=[]
 with output.open('xb') as out:
  out.write(struct.pack('<8sIIII',b'MAIBIR02',2,len(data['tables']),6,12))
  for i,entry in enumerate(data['tables']):
   identity=entry['material_identity'];key=(identity['section_id'],identity['runtime_density_g_cm3']);d,lines,pin=by_material[key];blocks={line.split()[0]:(n,line.split()) for n,line in enumerate(lines[1:])};verified={}
   def read(name,dtype):
    n,t=blocks[name];count,width=int(t[1]),int(t[2]);f=d/f'{n}.bin';assert width==dtype.itemsize and f.stat().st_size==count*width and sha(f)==t[3],(i,name)
    verified[name]=dict(path=str(f.resolve()),sha256=t[3],count=count);return np.memmap(f,dtype,mode='r',shape=(count,))
   channels=read('birth_channels',CH);samples=read('birth_samples',BS);records=read('endpoint_records',ER);tracks=read('compact_tracks',TR)
   assert len(channels)==60 and np.array_equal(channels['low'],np.arange(60)*5) and np.array_equal(channels['high'],(np.arange(60)+1)*5)
   channels=np.array(channels[:51]);count=int(channels[-1]['offset']+channels[-1]['count']);offset=0
   for c in channels:
    assert c['offset']==offset and 0<=c['fraction']<=1 and ((c['fraction']>0)==(c['count']>0));v=samples[offset:offset+int(c['count'])]['cdf']
    assert not len(v) or (np.all(np.isfinite(v)) and v[0]>0 and np.all(np.diff(v)>0) and v[-1]==1)
    offset+=int(c['count'])
   assert offset==count
   out.write(struct.pack('<idQI',key[0],key[1],count,51));out.write(channels.tobytes());maximum=0;max_norm=0
   for first in range(0,count,131072):
    sample=samples[first:min(count,first+131072)];assert np.all(sample['source']==0) and np.all(sample['row']<len(records))
    r=records[sample['row']];assert np.all(r['track']<len(tracks)) and np.all(r['step']==1) and not r['reserved'].any() and not r['padding'].any()
    t=tracks[r['track']];assert np.all(t['parent']==1) and np.all(t['pdg']==11) and np.all(t['parent_valid']==1) and not t['reserved'].any()
    assert np.all(np.isfinite(r['energy'])) and np.all((r['energy']>0)&(r['energy']<=.65)) and np.all(abs(r['energy']-t['birth'])<1e-8),(i,float(r['energy'].max()))
    norm=np.sqrt(np.sum(t['direction']**2,axis=1));enorm=np.sqrt(np.sum(r['direction']**2,axis=1));assert np.all(abs(norm-1)<1e-10) and np.all(abs(enorm-1)<1e-10)
    cosine=np.sum(r['direction']*(t['direction']/norm[:,None]),axis=1);assert np.all(np.isfinite(cosine)) and np.all(abs(cosine)<=1+1e-12)
    values=np.column_stack([sample['cdf'],r['energy'],np.clip(cosine,-1,1)]).astype('<f8');out.write(values.tobytes());maximum=max(maximum,float(r['energy'].max()));max_norm=max(max_norm,float(abs(enorm-1).max()))
   results.append(dict(node=i,material_identity=identity,cache_manifest_sha256=pin,cache_key=d.name,blocks=verified,samples=count,channels=51,max_electron_MeV=maximum,max_direction_norm_error=max_norm))
   print('node',i,'samples',count,'max_electron_MeV',maximum,flush=True)
 manifest=dict(projectile=[6,12],source_index_sha256=sha(index),bank_sha256=sha(output),bytes=output.stat().st_size,maximum_incident_MeVu=250,channel_upper_edge_MeVu=255,maximum_electron_MeV=max(r['max_electron_MeV'] for r in results),nodes=results,scope='Research C12 birth law from original16-history/material300MeV/u extraction. Original channels0..50 preserved, including exact250 query. Electron propagation uses independently validated shared material banks; carbon CT acceptance pending.',generator_sha256=sha(__file__))
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='nodes'}),flush=True)
if __name__=='__main__':main()
