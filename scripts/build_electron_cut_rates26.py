#!/usr/bin/env python3
"""Fit26 hard-rate functions in physical cut, not mixed density controls.

Offline sources remain explicit. Each runtime composition contains continuous
cut coefficients and one small density correction; no83-node directory. This
does not supply conditional states or establish off-grid native equivalence.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np
from numpy.polynomial import chebyshev as ch
from build_electron_response26 import collision_nodes,raw_rate,TS
def evaluate(coeff,x):
    x=np.asarray(x,dtype=np.float32).reshape((-1,)+(1,)*(coeff.ndim-1));a=np.zeros((len(x),)+coeff.shape[1:],np.float32);b=np.zeros_like(a)
    for c in coeff[:0:-1]:a,b=2*x*a-b+c,a
    return x*a-b+coeff[0]
def interpolate(query,cut,data):
    if len(cut)==1:return np.broadcast_to(data[0],(len(query),)+data.shape[1:]).copy()
    hi=np.clip(np.searchsorted(cut,query),1,len(cut)-1);lo=hi-1;fraction=np.clip((query-cut[lo])/(cut[hi]-cut[lo]),0,1)
    fraction=fraction.reshape((-1,)+(1,)*(data.ndim-1));return data[lo]+fraction*(data[hi]-data[lo])
def fit(cut,data,tolerance):
    low,high=np.log(cut[[0,-1]]);peak=np.max(abs(data),axis=0);scale=np.maximum(peak,np.max(peak)*1e-6)
    if low==high:return data[None,None,0].astype('<f4'),dict(pieces=1,degree=0,max_peak_normalized_error=0.)
    train_x=np.cos(np.pi*(np.arange(128)+.5)/128);chosen=None
    for pieces in [1,2,4,8,16,32,64]:
        coeff=[];worst=0.
        for i in range(pieces):
            lower=low+(high-low)*i/pieces;upper=low+(high-low)*(i+1)/pieces
            query=np.exp(lower+(train_x+1)*(upper-lower)/2);target=interpolate(query,cut,data)
            c=np.tensordot(ch.chebvander(train_x,12).T,target,axes=(1,0))*2/len(train_x);c[0]/=2;c=c.astype('<f4')
            logknots=np.log(cut);logknots=logknots[(logknots>=lower)&(logknots<=upper)]
            x=np.unique(np.r_[np.linspace(-1,1,129),2*(logknots-lower)/(upper-lower)-1]).astype(np.float32)
            expected=interpolate(np.exp(lower+(x.astype(float)+1)*(upper-lower)/2),cut,data)
            actual=evaluate(c,x).astype(float);error=float(np.max(abs(actual-expected)/scale));worst=max(worst,error);coeff.append(c)
        chosen=(np.array(coeff),dict(pieces=pieces,degree=12,max_peak_normalized_error=worst))
        if worst<=tolerance:return chosen
    return chosen
def fit_adaptive(cut,data,tolerance):
    """Resolve physical-cut response kinks locally, with disjoint probes.

    Bounds describe the argument of a rate function, not density identities.
    They are chosen by coefficient error, including the narrowcut source
    changes that the uniform polynomial failed to represent.
    """
    low,high=np.log(cut[[0,-1]]);peak=np.max(abs(data),axis=0);scale=np.maximum(peak,np.max(peak)*1e-6)
    if low==high:return data[None,None,0].astype('<f4'),np.array([low,high]),dict(pieces=1,degree=0,max_peak_normalized_error=0.)
    train=np.cos(np.pi*(np.arange(128)+.5)/128);segments=[]
    def one(lower,upper,depth):
        target=interpolate(np.exp(lower+(train+1)*(upper-lower)/2),cut,data)
        coeff=np.tensordot(ch.chebvander(train,12).T,target,axes=(1,0))*2/len(train);coeff[0]/=2;coeff=coeff.astype('<f4')
        knots=np.log(cut);knots=knots[(knots>lower)&(knots<upper)]
        x=np.unique(np.r_[np.linspace(-1,1,257),2*(knots-lower)/(upper-lower)-1]).astype(np.float32)
        query=np.exp(lower+(x.astype(float)+1)*(upper-lower)/2)
        error=np.max((abs(evaluate(coeff,x).astype(float)-interpolate(query,cut,data))/scale).reshape(len(x),-1),axis=1)
        worst=float(error.max())
        if worst<=tolerance or depth>=24 or len(segments)>256:
            segments.append((lower,upper,coeff,worst));return
        pivot=float(np.clip(x[np.argmax(error)],-.9,.9));middle=lower+(pivot+1)*(upper-lower)/2
        one(lower,middle,depth+1);one(middle,upper,depth+1)
    one(low,high,0)
    return np.array([s[2] for s in segments]),np.array([s[0] for s in segments]+[high]),dict(pieces=len(segments),degree=12,max_peak_normalized_error=max(s[3] for s in segments))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data/archive/electron_density_nodes_20261006/shared/charged'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tolerance',type=float,default=.0025);p.add_argument('--adaptive-cut',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    nodes=collision_nodes(a.source,'charged');cuts={}
    for line in (a.source/'index.txt').read_text().splitlines()[1:]:
        s,rho,file=line.split()
        with (a.source/file).open('rb') as f:h=struct.unpack('<8sIid4IQ4d',f.read(80))
        cuts[int(s),float(rho)]=h[9:11]
    arrays={};rows=[]
    for section in range(26):
        group=sorted((n for n in nodes if n['section']==section),key=lambda n:n['density']);rho=np.array([n['density'] for n in group])
        for proc in range(2):
            edges=np.unique(np.concatenate([n['rates'][proc][:,:2].ravel() for n in group]));probes=edges[:-1,None]+np.diff(edges)[:,None]*TS
            values=np.array([raw_rate(n['rates'][proc],probes)/n['density'] for n in group]);cut=np.array([cuts[section,n['density']][proc] for n in group])
            # At a production-cut floor, density still affects native mass
            # rates weakly. Retain its measured linear correction rather than
            # incorrectly claiming pure density scaling at fixed cut.
            unique=np.unique(cut);slope=np.zeros_like(values[0]);weight=0
            for c in unique:
                selected=cut==c
                if selected.sum()>1:
                    x=rho[selected]-rho[selected].mean();slope+=np.tensordot(x,values[selected],axes=(0,0))/np.sum(x*x);weight+=1
            if weight:slope/=weight
            centered=values-rho[:,None,None]*slope
            base=np.array([centered[cut==c].mean(axis=0) for c in unique])
            if a.adaptive_cut:coeff,bounds,stats=fit_adaptive(unique,base,a.tolerance)
            else:
                coeff,stats=fit(unique,base,a.tolerance);bounds=np.linspace(*np.log(unique[[0,-1]]),stats['pieces']+1)
            prefix=f'c{section:02d}_p{proc}_';arrays[prefix+'edges']=edges.astype('<f4');arrays[prefix+'cut_log_domain']=np.log(unique[[0,-1]]).astype('<f4')
            arrays[prefix+'coeff']=coeff;arrays[prefix+'density_slope']=slope.astype('<f4');arrays[prefix+'density_domain']=rho[[0,-1]].astype('<f4')
            arrays[prefix+'cut_piece_bounds']=bounds.astype('<f4')
            # Triangle bound covers all implicit cut pieces and all densities;
            # the native energy cubic still needs its own Lagrange bound.
            upper=np.max(np.sum(abs(coeff.astype(float)),axis=1)+rho[-1]*abs(slope))
            arrays[prefix+'coefficient_majorant']=np.array([np.nextafter(np.float32(6*upper*1.02),np.float32(np.inf))],'<f4')
            row=dict(section=section,process=proc,**stats,intervals=len(edges)-1,cut_domain=unique[[0,-1]].tolist(),native_fit_pass=stats['max_peak_normalized_error']<=a.tolerance)
            rows.append(row);print(json.dumps(row),flush=True)
    np.savez_compressed(a.out/'rates26.npz',**arrays)
    result=dict(status='CUT_CONDITIONED_RATE_COMPONENT_NOT_CONDITIONAL_STATE_OR_GPU_ACCEPTANCE',compositions=26,density_states=0,adaptive_cut=a.adaptive_cut,tolerance=a.tolerance,component_fit_pass=all(r['native_fit_pass'] for r in rows),bytes=(a.out/'rates26.npz').stat().st_size,array_bytes=sum(v.nbytes for v in arrays.values()),reference='explicit native sources; interpolation in actual physical cut; independent off-grid native checks pending',rows=rows)
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result['component_fit_pass']:raise SystemExit(2)
if __name__=='__main__':main()
