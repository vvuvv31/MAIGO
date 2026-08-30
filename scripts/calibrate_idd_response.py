#!/usr/bin/env python3
"""Build/apply a homogeneous-water +Z carbon IDD response package from 3D scorers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

NX=400; NY=400; NZ=800; DZ=0.5; ENERGIES=(100,200,300,400)

def energy_tag(energy): return f"{float(energy):g}"

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()

def idd(path,dtype):
 return np.memmap(path,dtype=dtype,mode='r',shape=(NZ,NY,NX)).sum((1,2),dtype=np.float64)

def distal_crossing(y,fraction):
 z=(np.arange(NZ)+.5)*DZ; p=int(np.argmax(y)); q=fraction*y[p]
 for i in range(p,NZ-1):
  if y[i]>=q>y[i+1]: return float(z[i]+(q-y[i])*DZ/(y[i+1]-y[i]))
 raise ValueError('distal crossing not found')

def build(args):
 package={'format':'MAIGO_IDD_RESPONSE_V1','geometry':'homogeneous_water_plus_z',
          'histories':1000000,'depth_bin_width_mm':DZ,'energies':[]}
 for e in ENERGIES:
  tag=energy_tag(e)
  gpu=args.gpu_root/f"gpu_fred_paper_1M_e{tag}"/"voxel_dose.raw"
  topas=args.topas_root/f"e{tag}"/f"topas_emittance_inelastic_e{tag}.bin"
  g=idd(gpu,'<f4'); t=idd(topas,'<f8'); r80=distal_crossing(t,.8)
  factor=np.ones(NZ,dtype=np.float64); valid=g>0
  factor[valid]=t[valid]/g[valid]
  peak=int(np.argmax(t)); factor[peak+1:]=1.0
  if np.any((factor[:peak+1]<args.factor_min)|(factor[:peak+1]>args.factor_max)):
   lo=float(factor[:peak+1].min()); hi=float(factor[:peak+1].max())
   raise ValueError(f'{e} MeV/u response [{lo},{hi}] exceeds configured bounds')
  z=(np.arange(NZ)+.5)*DZ
  package['energies'].append({'energy_MeVu':e,'r80_mm':r80,
   'normalized_depth':(z/r80).tolist(),'factor':factor.tolist(),
   'gpu_sha256':sha256(gpu),'topas_sha256':sha256(topas)})
 args.package.parent.mkdir(parents=True,exist_ok=True)
 args.package.write_text(json.dumps(package,indent=2)+'\n')
 print(args.package)

def response(package,energy):
 entries=package['energies']; es=np.array([x['energy_MeVu'] for x in entries],float)
 if energy<=es[0]: lo=hi=0; w=0.
 elif energy>=es[-1]: lo=hi=len(es)-1; w=0.
 else:
  hi=int(np.searchsorted(es,energy)); lo=hi-1; w=(energy-es[lo])/(es[hi]-es[lo])
 r80=(1-w)*entries[lo]['r80_mm']+w*entries[hi]['r80_mm']
 z=(np.arange(NZ)+.5)*DZ; u=z/r80
 def at(entry): return np.interp(u,entry['normalized_depth'],entry['factor'],left=entry['factor'][0],right=1.)
 return (1-w)*at(entries[lo])+w*at(entries[hi])

def apply_one(src,dst,factor):
 dst.parent.mkdir(parents=True,exist_ok=True)
 vin=np.memmap(src,dtype='<f4',mode='r',shape=(NZ,NY,NX))
 vout=np.memmap(dst,dtype='<f4',mode='w+',shape=(NZ,NY,NX))
 for z in range(NZ): vout[z]=vin[z]*factor[z]
 vout.flush(); del vout

def apply(args):
 package=json.loads(args.package.read_text())
 if package.get('geometry')!='homogeneous_water_plus_z': raise ValueError('unsupported response geometry')
 results=[]
 for e in args.energies:
  tag=energy_tag(e)
  src=args.gpu_root/f"gpu_fred_paper_1M_e{tag}"/"voxel_dose.raw"
  dst_dir=args.output_root/f"e{tag}"; dst=dst_dir/"voxel_dose.raw"
  factor=response(package,e); apply_one(src,dst,factor)
  topas=args.topas_root/f"e{tag}"/f"topas_emittance_inelastic_e{tag}.bin"
  result={'energy_MeVu':e,'validation':'not_available'}
  if topas.exists():
   g=idd(dst,'<f4'); t=idd(topas,'<f8'); peak=int(np.argmax(t)); err=np.abs(g[:peak+1]/t[:peak+1]-1)*100
   result={'energy_MeVu':e,'validation':'topas_3d','bins':peak+1,'max_error_percent':float(err.max()),
           'mae_percent':float(err.mean()),'p95_percent':float(np.percentile(err,95)),
           'within_1_percent':float(np.mean(err<1)*100)}
  results.append(result)
  (dst_dir/'voxel_dose.mhd').write_text('ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nElementSpacing = 0.2 0.2 0.5\nDimSize = 400 400 800\nElementType = MET_FLOAT\nDoseUnits = Gy\nElementDataFile = voxel_dose.raw\n')
 report={'criterion':'raw 0.5 mm bins, entrance through TOPAS peak, absolute Gy','results':results}
 args.output_root.mkdir(parents=True,exist_ok=True)
 (args.output_root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 for x in results: print(x)

def main():
 p=argparse.ArgumentParser(); p.add_argument('action',choices=('build','apply'))
 p.add_argument('--gpu-root',type=Path,default=Path('out')); p.add_argument('--topas-root',type=Path,default=Path('/mnt/sda/wuwei/carbon_emittance_inelastic_1M'))
 p.add_argument('--package',type=Path,default=Path('data/packages/c12_water_idd_response_topas_4_2_p3.json'))
 p.add_argument('--output-root',type=Path,default=Path('out/fred_topas_1M_idd_corrected'))
 p.add_argument('--factor-min',type=float,default=.8); p.add_argument('--factor-max',type=float,default=1.2)
 p.add_argument('--energies',default='100,200,300,400',help='comma-separated MeV/u energies to apply')
 a=p.parse_args(); a.energies=tuple(float(x) for x in a.energies.split(','))
 build(a) if a.action=='build' else apply(a)
if __name__=='__main__': main()
