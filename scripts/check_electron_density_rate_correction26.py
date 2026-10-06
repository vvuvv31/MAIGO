#!/usr/bin/env python3
"""Screen oscillator-density correction against independent native lambdas.

Experimental analytical correction; no production package or default switch.
Native-grid spline artifacts near thresholds remain explicitly measured.
"""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebval
from build_electron_response26 import collision_nodes,raw_rate
from penelope_composition_model import hard_shell_xs,GRID
from check_electron_penelope_composition import OSC
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--density',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--offgrid',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fp32',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    coeff=np.load(a.density/'penelope_density26.npz');rows=json.loads((a.baseline/'result.json').read_text())['rows'];actual=np.fromfile(a.baseline/'actual.bin','<f4').astype(float)
    source={(n['section'],n['density']):n['rates'] for n in collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged')};off={}
    for d in a.offgrid.glob('c*_off*'):
        meta=json.loads((d/'report.json').read_text());data=np.loadtxt(d/'hard_rate_cubics.csv',delimiter=',',skiprows=1)
        for proc in [0,1]:off[meta['section'],meta['density'],proc]=data[data[:,0]==proc,1:7]
    references={};out=[]
    if a.fp32:
        binary=a.out/'check'
        subprocess.run(['g++','-O3','-std=c++20','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_penelope_lambda26.cpp'),'-o',str(binary)],check=True)
    for row in rows:
        s=row['section'];rho=row['density'];cut=float(np.float32(row['cut']));proc=row['process']
        energy=np.unique(np.r_[np.geomspace(1e-5,.65,1024),(2 if proc==0 else 1)*cut*np.array([.9999,1,1.0001,1.01,1.1])]).astype('<f4');energy=energy[energy<=np.float32(.65)].astype(float)
        if len(energy)!=row['count']:raise ValueError('Matched query support')
        native=source[s,rho][proc] if row['scope']=='native_source_node' else off[s,rho,proc]
        expected=np.maximum(0.,raw_rate(native,energy));values=actual[row['offset']:row['offset']+row['count']].copy()
        if proc==0:
            if s not in references:
                d=ROOT/f'out/electron_complete26_20261006/composition_laws_v3/composition{s:02d}'
                table=np.atleast_1d(np.genfromtxt(d/'penelope_oscillators.csv',delimiter=',',names=True));w=np.genfromtxt(d/'penelope_weights.csv',delimiter=',',names=True)
                delta=w[w['oscillator']==0]['density_correction'][:34];aud=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
                refs=(table,delta,float(aud[0]['density_g_cm3']));references[s]=refs
            ref,rd,ref_rho=references[s];phase=0
            if s==1:
                conduction=ref[ref['ionization_MeV']==0]
                threshold=ref_rho*(1e-6/float(conduction[0]['resonance_MeV']))**2;phase=int(rho>=threshold)
            prefix=f'c{s:02d}_p{phase}_';domain=coeff[prefix+'log_domain'];x=0. if domain[0]==domain[1] else np.clip(2*(np.log(rho)-domain[0])/(domain[1]-domain[0])-1,-1,1)
            base=coeff[prefix+'base'];target=np.zeros(len(base),ref.dtype)
            for dst,src in [('strength','strength'),('ionization_MeV','ionization'),('binding_MeV','binding'),('z','z'),('shell_flag','shell_flag')]:target[dst]=base[src]
            rc=np.exp(chebval(x,coeff[prefix+'resonance_recoil_coeff']));target['resonance_MeV']=rc[:,0];target['recoil_cut_MeV']=rc[:,1]
            td=chebval(x,coeff[prefix+'delta_coeff'])
            def sigma(table,delta,e):
                # Native Penelope total cross-section grid is log interpolated.
                # The full process lambda uses its separate spline afterward.
                e=np.atleast_1d(e)
                upper=np.clip(np.searchsorted(GRID,e),1,33);lower=upper-1
                weight=np.log(e/GRID[lower])/np.log(GRID[upper]/GRID[lower])
                def at(index):
                    ee=GRID[index];ds=delta[index]
                    return np.maximum(hard_shell_xs(table[None,:],ee[:,None],cut,ds[:,None],raw=True).sum(1),1e-40)
                lo=at(lower);hi=at(upper)
                return np.exp((1-weight)*np.log(lo)+weight*np.log(hi))
            # Reconstruct the native process spline for the density ratio,
            # rather than multiplying by the pointwise model XS ratio.
            # Native forces the first lambda node to zero at the cut onset.
            minimum=2*cut;bins=int(np.floor(20*np.log10(1e4/minimum)+.5));knots=np.geomspace(minimum,1e4,bins+1)
            mol=ref[:1].copy();mol['strength']=ref['strength'].sum();mol['ionization_MeV']=0;mol['resonance_MeV']=1e-10;mol['recoil_cut_MeV']=1e-10
            def moller(e):return hard_shell_xs(mol[None,:],np.atleast_1d(e)[:,None],cut,np.zeros((len(np.atleast_1d(e)),1))).sum(1)
            def spline(table,delta):
                values=np.zeros(len(knots));low=knots<=.1
                values[low]=sigma(table,delta,knots[low])
                if cut<.05:
                    pen=float(sigma(table,delta,[.1])[0]);m=float(moller([.1])[0]);ee=knots[~low]
                    values[~low]=moller(ee)*(1+(pen/m-1)*.1/ee)
                else:values[~low]=moller(knots[~low])
                values[0]=0.
                # Exact Geant4 Kvasov recurrence. SciPy's endpoint solver
                # gives different first-bin curvature for this native table.
                n=len(knots)-1;sec=np.zeros(n+1);u=np.zeros(n)
                u[1]=((values[2]-values[1])/(knots[2]-knots[1])-(values[1]-values[0])/(knots[1]-knots[0]))
                u[1]*=6*(knots[2]-knots[1])/(knots[2]-knots[0])**2
                sec[1]=(2*knots[1]-knots[0]-knots[2])/(2*knots[2]-knots[0]-knots[1])
                for i in range(2,n-1):
                    sig=(knots[i]-knots[i-1])/(knots[i+1]-knots[i-1]);p=sig*sec[i-1]+2
                    sec[i]=(sig-1)/p
                    u[i]=(values[i+1]-values[i])/(knots[i+1]-knots[i])-(values[i]-values[i-1])/(knots[i]-knots[i-1])
                    u[i]=6*u[i]/(knots[i+1]-knots[i-1])-sig*u[i-1]/p
                sig=(knots[n-1]-knots[n-2])/(knots[n]-knots[n-2]);p=sig*sec[n-3]+2
                u[n-1]=(values[n]-values[n-1])/(knots[n]-knots[n-1])-(values[n-1]-values[n-2])/(knots[n-1]-knots[n-2])
                u[n-1]=6*sig*u[n-1]/(knots[n]-knots[n-2])-(2*sig-1)*u[n-2]/p
                p=1+sig+(2*sig-1)*sec[n-2];sec[n-1]=u[n-1]/p
                for k in range(n-2,1,-1):sec[k]*=sec[k+1]-u[k]*(knots[k+1]-knots[k-1])/(knots[k+1]-knots[k])
                sec[n]=(sec[n-1]-(1-sig)*sec[n-2])/sig
                sig=1-(knots[2]-knots[1])/(knots[2]-knots[0])
                sec[1]*=sec[2]-u[1]/(1-sig)
                sec[0]=(sec[1]-sig*sec[2])/(1-sig)
                index=np.clip(np.searchsorted(knots,energy)-1,0,n-1);dl=knots[index+1]-knots[index];b=np.clip((energy-knots[index])/dl,0,1)
                result=values[index]+b*(values[index+1]-values[index])+b*(b-1)*((2-b)*sec[index]+(1+b)*sec[index+1])*dl*dl/6
                return np.maximum(result,0.)
            numerator=spline(target,td);denominator=spline(ref,rd)
            ratio=np.divide(numerator,denominator,out=np.ones_like(numerator),where=denominator>0)
            if a.fp32:
                tables=[]
                for native_table in [ref,target]:
                    table=np.empty(len(native_table),OSC)
                    for dst,src in [('strength','strength'),('ionization','ionization_MeV'),('resonance','resonance_MeV'),('recoil_cut','recoil_cut_MeV'),('binding','binding_MeV'),('z','z'),('shell_flag','shell_flag')]:table[dst]=native_table[src]
                    tables.append(table)
                fixture=a.out/'fixture.bin';result=a.out/'ratio.bin'
                with fixture.open('wb') as f:
                    f.write(struct.pack('<IfIII',1,cut,len(ref),len(target),len(energy)))
                    for array in tables+[rd.astype('<f4'),td.astype('<f4'),energy.astype('<f4')]:f.write(array.tobytes())
                subprocess.run([str(binary),str(fixture),str(result)],check=True)
                ratio=np.fromfile(result,'<f4').astype(float)
            values*=ratio
        difference=abs(values-expected);peak=max(expected.max(),1e-30);active=expected>peak*1e-4
        out.append(dict(row,peak_error=float(difference.max()/peak),active_relative=float(np.max(difference[active]/expected[active])) if active.any() else 0.,zero_support_peak_leak=float(values[expected==0].max()/peak) if np.any(expected==0) else 0.))
    report=dict(status='DENSITY_CORRECTION_SCREEN_NOT_GPU_RATE_ACCEPTANCE',arithmetic='CPP_FP32' if a.fp32 else 'PYTHON_FP64_OFFLINE',cases=len(out),maximum_peak_error=max(r['peak_error'] for r in out),maximum_active_relative=max(r['active_relative'] for r in out),maximum_zero_support_peak_leak=max(r['zero_support_peak_leak'] for r in out),component_pass=all(r['peak_error']<.0025 and r['active_relative']<.01 for r in out),rows=out)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    if not report['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
