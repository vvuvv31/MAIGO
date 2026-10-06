#!/usr/bin/env python3
"""Actual-cut native-grid hard-rate functions with26 composition descriptors.

No interpolation of fixed-energy density responses. Native grid-bin rounding
and model-switch crossings define cut-function partitions; failed pieces are
refined against disjoint native cuts. This generates an experimental component.
"""
import argparse,json,os,struct,subprocess
from pathlib import Path
import numpy as np
from numpy.polynomial import chebyshev as ch
ROOT=Path(__file__).resolve().parents[1]
SCALE=20/np.log(10.)

def native_bins(cut,process):return int(np.rint(SCALE*np.log(1e4/((2 if process==0 else 1)*cut))))
def partitions(low,high,process,oscillators=None):
    if low==high:return [(low,float(np.nextafter(np.float32(high),np.float32(np.inf))))]
    a,b=np.log([low,high]);factor=2 if process==0 else 1;edges=[a,b]
    for n in range(native_bins(high,process)-1,native_bins(low,process)+2):
        for half in [-.5,.5]:
            e=np.log(1e4/factor)-(n+half)/SCALE
            if a<e<b:edges.append(e)
        if process==0:
            for i in range(n):
                e=(np.log(.1)-i*np.log(1e4)/n)/(1-i/n)-np.log(factor)
                if a<e<b and native_bins(np.exp(e),process)==n:edges.append(e)
    if process==0 and oscillators is not None:
        # Native Penelope moves a FINITE distant-resonance contribution from
        # hard to soft exactly at cut==resonance. It cannot be smoothed by a
        # polynomial. Close-channel onset also follows its fixed XS grid.
        energies=np.geomspace(1e-4,1e5,100);energies=energies[energies<=.100001]
        support=np.r_[oscillators['resonance_MeV'],oscillators['recoil_cut_MeV'],
                      (.5*(energies[:,None]+oscillators['ionization_MeV'])).ravel()]
        edges.extend(np.log(support[(support>low)&(support<high)]))
    def ceiling(x):
        cut=np.float32(np.exp(x))
        if float(cut)<np.exp(x):cut=np.nextafter(cut,np.float32(np.inf))
        return float(cut)
    # Physical FP32 cut starts assign every representable cut to exactlyone
    # native support interval. Rounded log bounds cannot do that near N jumps.
    edges=np.unique([ceiling(x) for x in np.unique(edges)])
    edges[-1]=float(np.nextafter(np.float32(high),np.float32(np.inf)))
    return list(zip(edges[:-1],edges[1:]))

class Native:
    def __init__(self,binary,reference,out,reuse=None):
        self.binary=binary.resolve();self.reference=reference;self.out=out;self.cache={};self.round=0
        for campaign in reuse or []:
            for d in sorted(campaign.glob('native_round*')):
                if (d/'topas.log').exists() and 'ELECTRON_COMPOSITION_PROBE=PASS' in (d/'topas.log').read_text():
                    q=np.atleast_2d(np.loadtxt(d/'lambda_cuts.txt'))
                    self.load(d,[(int(r[1]),float(r[2])) for r in q])
    def load(self,d,queries):
        values=np.atleast_2d(np.loadtxt(d/'lambda_segments.csv',delimiter=',',skiprows=1))
        boundaries=np.r_[0,1+np.flatnonzero(np.diff(values[:,0])),len(values)]
        if len(boundaries)!=len(queries)+1:raise ValueError('Native cut export group count')
        density=json.loads((self.reference/'report.json').read_text())['density']
        for i,key in enumerate(queries):
            rows=values[boundaries[i]:boundaries[i+1]];n=int(rows[0,5]);minimum=rows[0,3]
            if len(rows)!=n or n!=native_bins(key[1],key[0]):raise ValueError('Unexpected native grid configuration')
            # Keep sufficient intervals for EVERY cut in a piece, including
            # the upper endpoint of the runtime0.65MeV electron domain.
            self.cache[key]=(n,rows[:,9:13]/density)
    def obtain(self,queries):
        queries=sorted(set((p,float(np.float32(c))) for p,c in queries)-self.cache.keys())
        if not queries:return
        d=self.out/f'native_round{self.round:03d}';d.mkdir();self.round+=1
        (d/'topas.txt').write_text((self.reference/'topas.txt').read_text());(d/'requests.txt').write_text('0 0 0 .04 .00099 .00099 0\n')
        (d/'lambda_cuts.txt').write_text(''.join(f'{i} {p} {c:.17g}\n' for i,(p,c) in enumerate(queries)))
        env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
        with (d/'topas.log').open('x') as f:r=subprocess.run([str(self.binary),'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode or 'ELECTRON_COMPOSITION_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(d)
        self.load(d,queries)
    def data(self,process,cuts,count):
        return np.array([self.cache[process,float(np.float32(c))][1][:count] for c in cuts])

def fit(native,process,initial,tolerance,degree):
    train=np.cos(np.pi*(np.arange(32)+.5)/32);test=np.linspace(-.99999,.99999,49);accepted=[];pending=[(*p,0) for p in initial]
    while pending:
        tests=[]
        for low,high,depth in pending:
            cuts=np.minimum((low+(np.r_[train,test]+1)*(high-low)/2).astype(np.float32),np.nextafter(np.float32(high),np.float32(-np.inf)))
            # Quantize actual cuts exactly as a FP32 transport query.
            tests.extend((process,c) for c in cuts)
        native.obtain(tests);retry=[]
        for low,high,depth in pending:
            cuts=np.minimum((low+(np.r_[train,test]+1)*(high-low)/2).astype(np.float32),np.nextafter(np.float32(high),np.float32(-np.inf)))
            bins=[native.cache[process,float(c)][0] for c in cuts]
            if len(set(bins))!=1:
                # A one-ulp rounded N boundary is explicit, never smoothed.
                unique=np.unique(cuts);changes=np.flatnonzero(np.diff([native.cache[process,float(c)][0] for c in unique]))
                starts=np.r_[low,unique[changes+1],high]
                retry.extend((float(x),float(y),depth+1) for x,y in zip(starts[:-1],starts[1:]));continue
            minimum=(2 if process==0 else 1)*float(cuts.min());n=bins[0]
            count=min(n,max(1,int(np.floor(np.log(.65/minimum)/np.log(1e4/minimum)*n))+2))
            target=native.data(process,cuts,count)
            # FillLambdaVector matches Penelope to Moller with
            # del=(sigma_Pen(.1,cut)/sigma_Moller(.1,cut)-1)*.1.
            # The denominator vanishes as cut approaches.05 from below.
            # Factor that native pole analytically; no polynomial may smooth
            # it across the model boundary or hide it in extra density nodes.
            pole=process==0 and np.all(cuts.astype(float)<.05)
            if pole:target=target*(.05-cuts.astype(float))[:,None,None]
            if len(np.unique(cuts))==1:
                coeff=np.zeros((degree+1,count,4),'<f4');coeff[0]=target[0];error=0.
            else:
                actual_x=(2*(cuts.astype(float)-low)/(high-low)-1)
                effective_degree=min(degree,len(np.unique(cuts[:32]))-1)
                fitted=ch.chebfit(actual_x[:32],target[:32].reshape(32,-1),effective_degree).reshape(effective_degree+1,count,4).astype('<f4')
                coeff=np.zeros((degree+1,count,4),'<f4');coeff[:effective_degree+1]=fitted
                actual=ch.chebval(actual_x[32:],coeff).transpose(2,0,1)
                # Match the independently declared active-rate domain
                # (>=1e-4 of peak). Tiny signed spline coefficients cannot
                # sensibly be fitted at a relative-to-themselves tolerance.
                scale=np.maximum(np.max(abs(target),axis=0),np.max(abs(target))*1e-4)
                error=float(np.max(abs(actual-target[32:])/scale))
            if error>tolerance:
                if depth>=14:raise ValueError(f'Unresolved native cut-function error {error} in {low,high}')
                middle=float(np.float32((low+high)/2))
                if not low<middle<high:raise ValueError('Native function mismatch at one FP32 cut')
                retry.extend([(low,middle,depth+1),(middle,high,depth+1)])
            else:accepted.append((low,high,bins[0],coeff,error,pole))
        pending=retry
        print(json.dumps(dict(process=process,accepted=len(accepted),pending=len(pending),native_queries=len(native.cache))),flush=True)
    return sorted(accepted,key=lambda p:p[0])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sections',type=int,nargs='+',default=list(range(26)));p.add_argument('--degree',type=int,default=12);p.add_argument('--tolerance',type=float,default=1e-5)
    p.add_argument('--reuse-native',type=Path,nargs='+',help='Explicit completed offline native campaigns; never runtime fallback')
    p.add_argument('--penelope-functions',type=Path,help='Independent composition oscillator exports for exact support boundaries');a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);source=ROOT/'data/archive/electron_density_nodes_20261006/shared/charged';domains={s:[[],[]] for s in a.sections}
    for line in (source/'index.txt').read_text().splitlines()[1:]:
        s,rho,name=line.split();s=int(s)
        if s not in domains:continue
        with (source/name).open('rb') as f:h=struct.unpack('<8sIid4IQ4d',f.read(80))
        for process in [0,1]:domains[s][process].append(float(np.float32(h[9+process])))
    arrays={};reports=[]
    for section in a.sections:
        d=a.out/f'composition{section:02d}';d.mkdir();native=Native(a.binary,a.reference/d.name,d,[p/d.name for p in a.reuse_native or []])
        osc=None
        if a.penelope_functions and (a.penelope_functions/d.name/'penelope_oscillators.csv').exists():
            osc=np.atleast_1d(np.genfromtxt(a.penelope_functions/d.name/'penelope_oscillators.csv',delimiter=',',names=True))
        for process in [0,1]:
            low,high=min(domains[section][process]),max(domains[section][process]);pieces=fit(native,process,partitions(low,high,process,osc),a.tolerance,a.degree)
            prefix=f'c{section:02d}_p{process}_';arrays[prefix+'bounds']=np.array([r[0] for r in pieces]+[pieces[-1][1]],'<f4');offsets=[0]
            for i,r in enumerate(pieces):arrays[prefix+f'coeff{i:03d}']=r[3];offsets.append(offsets[-1]+r[3].size)
            arrays[prefix+'bins']=np.array([r[2] for r in pieces],'<u4');arrays[prefix+'intervals']=np.array([r[3].shape[1] for r in pieces],'<u4');arrays[prefix+'offsets']=np.array(offsets,'<u4')
            arrays[prefix+'pole']=np.array([r[5] for r in pieces],'<u4')
            upper=[]
            for r in pieces:
                divisor=.05-float(np.nextafter(np.float32(r[1]),np.float32(-np.inf))) if r[5] else 1.
                if divisor<=0:raise ValueError('Pole support partition crosses model boundary')
                upper.append(np.max(np.sum(abs(r[3].astype(float)),axis=0))/divisor)
            arrays[prefix+'majorant']=np.array([np.nextafter(np.float32(6*max(upper)*1.02),np.float32(np.inf))],'<f4')
            reports.append(dict(section=section,process=process,pieces=len(pieces),cut_domain=[low,high],maximum_interval_fit_error=max(r[4] for r in pieces),native_queries=len(native.cache)))
        np.savez_compressed(d/'functions.npz',**{k:v for k,v in arrays.items() if k.startswith(f'c{section:02d}_')})
    np.savez_compressed(a.out/'rates26.npz',**arrays)
    report=dict(status='ACTUAL_CUT_NATIVE_GRID_FUNCTION_COMPONENT_NOT_GPU_OR_NATIVE_OFFGRID_ACCEPTANCE',compositions=len(a.sections),density_states=0,degree=a.degree,tolerance=a.tolerance,array_bytes=sum(v.nbytes for v in arrays.values()),bytes=(a.out/'rates26.npz').stat().st_size,rows=reports)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
