"""Build 26 composition EM bases with continuous-density integral coefficients.

No target-density EM responses are read. Lightweight material scalars are
independently calculated by Geant4; integrated ranges are fitted from the
analytical model. Training densities are not stored as transport EM states.
"""
from pathlib import Path
import argparse,json,struct,subprocess,time
import numpy as np
from synthesize_analytic_density import rebuild,reference_record,eval_segments,RD

DEG=24; SAMPLES=64; SCALARS=257; RATE_KNOTS=257

def build(bases,section_file,helper,output):
 start=time.monotonic();output.mkdir(parents=True,exist_ok=False)
 with np.load(bases) as f:b={k:f[k] for k in f.files}
 sections=json.loads(section_file.read_text())['section_manifest'];elements=['H','C','N','O','Mg','P','S','Cl','Ar','Ca','Na','K','Ti']
 mats=b['materials'];rho_bounds={s:tuple(map(float,b['composition_density_bounds'][s+1])) if 'composition_density_bounds' in b else (float(mats['d'][mats['s']==s].min()),float(mats['d'][mats['s']==s].max())) for s in range(-1,25)}
 inp=[];queries=[];trainx=np.cos(np.pi*(np.arange(SAMPLES)+.5)/SAMPLES)
 for s in range(-1,25):
  bid,rho=b[f'meta_{s}'];lo,hi=rho_bounds[s];lr0,lr1=np.log(lo),np.log(hi)
  weights=[('H',.111894),('O',.888106)] if s==-1 else [(e,w) for e,w in zip(elements,sections[s]['element_weights']) if w>0]
  excitation=75 if s==-1 else sections[s]['mean_excitation_energy_eV']
  # Uniform scalar parameter grid plus independent quadrature nodes and base.
  densities=np.r_[np.exp(np.linspace(lr0,lr1,SCALARS)),np.exp(.5*(lr0+lr1)+.5*(lr1-lr0)*trainx),rho]
  queries.append(densities)
  for d in densities:
   inp.append(f'{len(inp)} {s} {d:.17g} {excitation:.17g} {len(weights)} '+' '.join(f'{e} {w:.17g}' for e,w in weights))
 (output/'material_inputs.txt').write_text('\n'.join(inp)+'\n')
 subprocess.run([str(helper.resolve()),str(output/'material_inputs.txt'),str(output/'material_metadata.txt')],check=True)
 metadata=np.loadtxt(output/'material_metadata.txt').reshape(26,SCALARS+SAMPLES+1,13)
 materials=[];records=[];nodes=[];vectors=[];descriptors=[];coeffs=[];no=vo=co=0
 for si,s in enumerate(range(-1,25)):
  rho=float(b[f'meta_{s}'][1]);materials.append((s,rho));lo,hi=rho_bounds[s]
  # Scalar grid stores cut, electron density coefficient and Sternheimer parameters.
  scalar=metadata[si,:SCALARS,3:11].astype('<f4')
  mb=metadata[si,-1];ranges=[]
  for ion in range(18):
   tag=f'{s}_{ion}';r=b[f'r_{tag}'].copy();bn=b[f'n_{tag}'];bv=[b[f'v_{tag}_{k}'] for k in range(4)]
   # Same energy knots as the composition's range spline; mass-normalized correction.
   ex=np.r_[bv[1][:,0],bv[1][-1,1]].astype(float)
   base_range=eval_segments(bv[1],ex,1);logs=[];rate_ratios=[]
   rr=reference_record(r);rate_u=np.linspace(0,1,RATE_KNOTS);mass=float(rr['f'][0]);re=.51099891/mass;max_e=float(bv[3][-1,1])
   for mt in metadata[si,SCALARS:SCALARS+SAMPLES]:
    _,_,predv=rebuild(r,bn,bv,mb,mt)
    emin=float(predv[3][0,0]);energy=np.exp(np.log(emin)+rate_u*np.log(max_e/emin))
    tau=energy/mass;bg2=tau*(tau+2);beta=bg2/(tau+1)**2;tm=2*.51099891*bg2/(1+2*(tau+1)*re+re*re)
    floor=.00025*mass/938.272013;cut=np.where(energy*(938.272013/mass)<=2,max(mt[3],floor),mt[3]);cut=np.minimum(cut,tm)
    den=1/cut-1/tm-beta/tm*np.log(tm/cut)
    if float(rr['f'][5])>0:den+=.5*(tm-cut)/(energy+mass)**2
    born=mt[4]*(4 if int(rr['u'][-2])==2 else 1)/beta*np.maximum(den,0)
    native=np.maximum(eval_segments(predv[3],energy,3),0)
    ratios=np.divide(native,born,out=np.ones_like(native),where=born>1e-20)
    ratios[0]=ratios[1];rate_ratios.append(ratios)
    target=eval_segments(predv[1],ex,1)*mt[2]/rho
    logs.append(np.log(target/base_range))
   # Chebyshev density approximation of log(mass-normalized range ratio).
   c=np.polynomial.chebyshev.chebfit(trainx,np.array(logs),DEG-1).T.astype('<f4')
   # Descriptor: domain, base scalars, range polynomial offset/knots.
   descriptors.append((np.log(lo),np.log(hi),rho,*mb[3:11],max_e,co,len(ex),co+len(ex),RATE_KNOTS))
   coeffs.append(np.column_stack([ex,c]).astype('<f4'));co+=len(ex)
   rate_coeff=np.polynomial.chebyshev.chebfit(trainx,np.array(rate_ratios),DEG-1).T.astype('<f4')
   coeffs.append(np.column_stack([rate_u,rate_coeff]).astype('<f4'));co+=RATE_KNOTS
   r['u'][0]=no;r['u'][1]=len(bn);no+=len(bn)
   for k,v in enumerate(bv):r['u'][2+k]=vo;r['u'][6+k]=len(v);vo+=len(v);vectors.append(v)
   records.append(r);nodes.append(bn)
  print(f'composition {s}: {time.monotonic()-start:.1f}s',flush=True)
 nodes=np.concatenate(nodes);vectors=np.concatenate(vectors);records=np.array(records,dtype=RD)
 em=output/'continuous26.bin'
 # EMCNT001 standard base layout followed by continuous-density extension.
 with em.open('wb') as f:
  f.write(struct.pack('<8s6I',b'EMCNT001',1,26,18,468,len(nodes),len(vectors)))
  f.write(np.array(materials,dtype=[('s','<i4'),('d','<f4')]).tobytes());f.write(b['species'].tobytes());f.write(records.tobytes());f.write(nodes.tobytes());f.write(vectors.tobytes())
  f.write(struct.pack('<4I',DEG,SCALARS,len(descriptors),co))
  # Packed descriptor: twelve floats plus four unsigned ints.
  for d in descriptors:f.write(struct.pack('<12f4I',*d))
  f.write(metadata[:,:SCALARS,3:11].astype('<f4').tobytes());f.write(np.concatenate(coeffs).tobytes())
 report=dict(schema='EMCNT001',version=1,composition_states=26,ion_species=18,density_response_states=0,range_density_degree=DEG-1,material_scalar_samples=SCALARS,em_bytes=em.stat().st_size,no_target_density_responses=True,elapsed_seconds=time.monotonic()-start)
 em_meta=dict(schema='EMCNT001',version=1,package_bytes=em.stat().st_size,
  materials=[dict(name='Water_75eV' if sec==-1 else f'Schneider_{sec}',section=sec,density_g_cm3=rho) for sec,rho in materials],
  species_za=b['species'].tolist(),records=468,nodes=len(nodes),segments=len(vectors),
  composition_bases=26,density_response_states=0,density_residual_tables=0,
  encoding='26 composition bases + continuous cut/density correction and integral polynomials',
  reference_water='Water_75eV; mean excitation energy75eV',generator=__file__,
  validation_status='candidate; validate components and transport/dose before installation')
 em.with_suffix('.metadata.json').write_text(json.dumps(em_meta,indent=2)+'\n')
 (output/'continuous_density.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bases',type=Path,default=Path('data/em/ions/composition_bases.npz'));p.add_argument('--sections',type=Path,default=Path('data/stopping/schneider/carbon.metadata.json'));p.add_argument('--helper',type=Path,default=Path('extensions/work/build/native/material_metadata'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.bases,a.sections,a.helper,a.output)
