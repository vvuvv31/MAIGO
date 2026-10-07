#!/usr/bin/env python3
"""Pack independent 26-composition scalar/oscillator and atomic components.

These files alone are NOT a complete GPU electron response package.
"""
import argparse,csv,json,struct
from pathlib import Path
import numpy as np
from formats import OSC
ROOT=Path(__file__).resolve().parents[3]
def rows(p):return list(csv.DictReader(p.open()))
def thresholds(probabilities,complete=False):
    result=[];total=0.
    for p in probabilities:
        if not np.isfinite(p) or p<0:raise ValueError('Transition probability')
        total+=p;result.append(min(2**64-1,int(total*2**64)))
    if total>1+1e-8:raise ValueError('Transition probability exceeds one')
    if result and complete:result[-1]=2**64-1
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--birth',type=Path,required=True);p.add_argument('--scalar',type=Path,required=True);p.add_argument('--laws',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    atoms={};fluo={};auger={};settings=None;compositions=[]
    birth=a.birth
    domains={}
    with birth.open('rb') as f:
        magic,v,n=struct.unpack('<8sII',f.read(16));assert magic==b'MAIBCA26' and n==26
        for _ in range(n):
            s,low,high,count=struct.unpack('<IffI',f.read(16));domains[s]=(low,high);f.seek(16*count,1)
    with (a.out/'penelope26.bin').open('xb') as f:
        f.write(struct.pack('<8sII',b'MAIPEN26',1,26))
        for s in range(26):
            d=a.laws/f'composition{s:02d}';r=rows(d/'penelope_oscillators.csv');table=np.empty(len(r),OSC)
            for j,row in enumerate(r):
                for dst,src in [('strength','strength'),('ionization','ionization_MeV'),('resonance','resonance_MeV'),('recoil_cut','recoil_cut_MeV'),('binding','binding_MeV'),('z','z'),('shell_flag','shell_flag')]:table[dst][j]=float(row[src])
            weights=rows(d/'penelope_weights.csv');delta=np.array([float(r['density_correction']) for r in weights if int(r['oscillator'])==0],'<f4')
            if len(delta)!=34 or not np.isfinite(delta).all():raise ValueError('Penelope grid')
            audit=rows(d/'composition_audit.csv');rho=float(audit[0]['density_g_cm3'])
            f.write(struct.pack('<III3f',s,len(table),len(delta),rho,*domains[s]));f.write(table.tobytes());f.write(delta.tobytes())
            cfg=dict(line.split('=') for line in (d/'atomic_settings.txt').read_text().splitlines())
            if settings is None:settings=cfg
            if cfg!=settings:raise ValueError('Atomic configurations differ')
            for row in rows(d/'atomic_shells.csv'):
                key=int(row['z']),int(row['shell_id']);value=(int(row['index']),float(row['binding_MeV']))
                if key in atoms and atoms[key]!=value:raise ValueError('Density-dependent atomic identity')
                atoms[key]=value
            for field,dst,filename in [('target',fluo,'atomic_radiative.csv'),('ejected',auger,'atomic_auger.csv')]:
                grouped={}
                for row in rows(d/filename):
                    key=int(row['z']),int(row['vacancy']);grouped.setdefault(key,[]).append((int(row['target']),int(row.get('ejected',0)),float(row['energy_MeV']),float(row['probability'])))
                for key,value in grouped.items():
                    if key in dst and dst[key]!=value:raise ValueError('Density-dependent atomic transition')
                    dst[key]=value
            compositions.append(dict(section=s,oscillators=len(table),reference_density=rho))
    for z in sorted({key[0] for key in atoms}):
        entries=sorted((shell,index) for (atomic_z,shell),(index,_) in atoms.items() if atomic_z==z)
        if any(index!=rank for rank,(_,index) in enumerate(entries)):
            raise ValueError('Native shell index is not ordered vacancy ordinal')
    with (a.out/'atomic26.bin').open('xb') as f:
        f.write(struct.pack('<8s5I',b'MAIAT026',1,len(atoms),*(int(settings[k]) for k in ['active','auger','cascade'])))
        for (z,id_),(_,binding) in sorted(atoms.items()):
            rad=fluo.get((z,id_),[]);aug=auger.get((z,id_),[])
            total=sum(x[3] for x in aug)
            if aug and total<=0:raise ValueError('Empty Auger probability')
            f.write(struct.pack('<4I3f',z,id_,len(rad),len(aug),binding,max([x[2] for x in rad],default=0),max([x[2] for x in aug],default=0)))
            for data,complete in [(rad,False),(aug,True)]:
                probabilities=[x[3]/total if complete else x[3] for x in data]
                for t,cdf in zip(data,thresholds(probabilities,complete)):
                    target,ejected,energy,_=t
                    if target<=id_ or (ejected and ejected<=id_):raise ValueError('Cyclic atomic transition')
                    f.write(struct.pack('<QfIII',cdf,energy,target,ejected,0))
    with np.load(a.scalar) as source,(a.out/'scalar26.bin').open('xb') as f:
        f.write(struct.pack('<8sII',b'MAIESC26',1,26))
        for s in range(26):
            prefix=f'c{s:02d}_';e=source[prefix+'energy'];c=source[prefix+'scalar_coeff'];cut=source[prefix+'cut_coeff'];domain=source[prefix+'log_domain']
            f.write(struct.pack('<5I2f',s,len(e),c.shape[0],cut.shape[0],cut.shape[1],*domain))
            for v in [e,c,cut]:f.write(v.astype('<f4').tobytes())
    if (a.laws/'composition00/brems_integrals.csv').exists():
        with (a.out/'brems26.bin').open('xb') as f:
            f.write(struct.pack('<8sII',b'MAISB026',2,26))
            for s in range(26):
                data=np.genfromtxt(a.laws/f'composition{s:02d}/brems_integrals.csv',delimiter=',',names=True)
                e=np.unique(data['energy_MeV']).astype('<f4');x=np.unique(data['fraction']).astype('<f4')
                field=next(k for k in data.dtype.names if k.startswith('mass_'))
                values=data[field].reshape(len(e),len(x)).astype('<f4')
                if not np.isfinite(values).all() or np.any(values<0) or np.any(values[:,-1]!=0):raise ValueError('Brems integral endpoints')
                # Numerical quadrature can fluctuate at the roundoff level.
                if np.max(np.maximum(np.diff(values,axis=1),0)/np.maximum(values[:,:-1],1e-30))>2e-6:raise ValueError('Nonmonotone native brems integral')
                values=np.minimum.accumulate(values,axis=1)
                element_data=np.genfromtxt(a.laws/f'composition{s:02d}/brems_element_integrals.csv',delimiter=',',names=True)
                zs=np.unique(element_data['z']).astype('<u4');element_values=[]
                for z in zs:
                    subset=element_data[element_data['z']==z]
                    if len(subset)!=len(e)*len(x) or not np.array_equal(subset['energy_MeV'].astype('<f4'),data['energy_MeV'].astype('<f4')) or not np.array_equal(subset['fraction'].astype('<f4'),data['fraction'].astype('<f4')):
                        raise ValueError('Element spectrum grid')
                    v=subset[field].reshape(len(e),len(x)).astype('<f4')
                    if not np.isfinite(v).all() or np.any(v<0) or np.any(v[:,-1]!=0):raise ValueError('Element spectrum support')
                    if np.max(np.maximum(np.diff(v,axis=1),0)/np.maximum(v[:,:-1],1e-30))>2e-6:raise ValueError('Nonmonotone element spectrum')
                    element_values.append(np.minimum.accumulate(v,axis=1))
                element_values=np.array(element_values,'<f4')
                if np.max(abs(element_values.sum(0)-values)/np.maximum(values,1e-30))>1e-5:raise ValueError('Element spectrum mass sum')
                f.write(struct.pack('<4I',s,len(e),len(x),len(zs)))
                for v in [e,x,values,zs,element_values]:f.write(v.tobytes())
    # Separate genuinely consumed parent parameters from shared material laws.
    for name,z,mass_number,mass,bragg,floor,form,magnetic,spin in [
        ('proton',1,1,np.float32(938.272013),2.,.00025,1.439482e-6,6.79943,1),
        ('carbon',6,12,np.float32(11174.863388)/np.float32(12),0.,0.,5.51029e-6,0.,0)]:
        (a.out/(name+'26.bin')).write_bytes(struct.pack('<8s4I5f',b'MAIEPR26',1,z,mass_number,spin,mass,bragg,floor,form,magnetic))
    report=dict(status='COMPONENT_PACKS_NOT_COMPLETE_GPU_PACKAGE',compositions=26,density_nodes=0,atomic_vacancies=len(atoms),atomic_settings=settings,components={p.name:p.stat().st_size for p in a.out.glob('*.bin')},rows=compositions)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
