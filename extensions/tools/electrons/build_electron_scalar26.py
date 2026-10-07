#!/usr/bin/env python3
"""Build 26 continuous composition scalar prototypes without a density directory.

This isolates scalar fitting from collision/birth distribution approximations.
Source interpolation is the current 83-node reference, not an independent
Geant4 calculation at intermediate densities. Hard rates/outcomes are excluded.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from numpy.polynomial import chebyshev as ch


def load_charged(path):
    with path.open('rb') as f:
        h=struct.unpack('<8sIid4IQ4d',f.read(80))
        if h[0] not in (b'MAIECF01',b'MAIECH02'):
            raise ValueError('Charged schema')
        f.seek(80)
        common=f.read(9520)
        rows=np.frombuffer(f.read(48*h[4]),'<f8').reshape(-1,6).copy()
    return dict(section=h[2],density=h[3],cut=np.array(h[9:11]),
                common=common,energy=rows[:,0],
                # mass mfp, mass stopping, mass range
                mass=np.column_stack([rows[:,1]*h[3],rows[:,2]/h[3],rows[:,3]*h[3]]))


def load_photon(path):
    with path.open('rb') as f:
        h=struct.unpack('<8sIidIIQ',f.read(40))
        if h[0] not in (b'MAIPHF01',b'MAIPHN01'):
            raise ValueError('Photon schema')
        rows=np.frombuffer(f.read(32*h[4]),'<f8').reshape(-1,4).copy()
    return rows[:,0],rows[:,1:]/h[3]


def interpolate_density(density,grid,values):
    if len(grid)==1:
        return np.broadcast_to(values[0],(len(density),)+values.shape[1:]).copy()
    flat=values.reshape(len(grid),-1)
    return np.column_stack([np.interp(density,grid,flat[:,j]) for j in range(flat.shape[1])]).reshape((len(density),)+values.shape[1:])


def fit(grid,values,tolerance,max_degree,log_domain=None):
    if np.any(values<=0) or not np.isfinite(values).all():
        raise ValueError('Positive scalar fields required')
    lmin,lmax=np.log(grid[[0,-1]]) if log_domain is None else log_domain
    if lmin==lmax:
        coeff=np.log(values[:1]).astype('<f4')
        return coeff,dict(degree=0,max_relative=float(np.max(abs(np.exp(coeff.astype(float))/values[:1]-1))))
    # Disjoint Chebyshev training and uniform+density-knot validation grids.
    ntrain=2048
    x=np.cos(np.pi*(np.arange(ntrain)+.5)/ntrain)
    rho=np.exp(lmin+(x+1)*(lmax-lmin)/2)
    target=np.log(interpolate_density(rho,grid,values))
    knots=np.log(grid)
    knots=knots[(knots>=lmin)&(knots<=lmax)]
    vx=np.unique(np.r_[np.linspace(-1,1,4097),2*(knots-lmin)/(lmax-lmin)-1])
    vrho=np.exp(lmin+(vx+1)*(lmax-lmin)/2)
    reference=interpolate_density(vrho,grid,values)
    chosen=None
    # Coefficients computed by Chebyshev projection, then stored/evaluated FP32.
    for degree in sorted({min(d,max_degree) for d in [1,2,4,8,12,16,24,32,max_degree]}):
        coeff=np.tensordot(ch.chebvander(x,degree).T,target,axes=(1,0))*2/ntrain
        coeff[0]/=2
        coeff=coeff.astype('<f4')
        actual=np.exp(evaluate(coeff,vx.astype(np.float32)).astype(float))
        error=float(np.max(abs(actual/reference-1)))
        chosen=(coeff,dict(degree=degree,max_relative=error))
        if error<=tolerance:
            return chosen
    return chosen


def fit_cut(grid,values,tolerance,max_degree):
    """Uniform log-density pieces resolve the production-cut floor kink.

    Piece coordinates are implicit, not native density-node identities.
    """
    for pieces in [1,2,4,8,16,32]:
        domain=np.linspace(*np.log(grid[[0,-1]]),pieces+1)
        # Prefer short local polynomials to a64-degree cutoff evaluation in
        # the worker. Composition directory remains26 regardless of pieces.
        fits=[fit(grid,values,tolerance,min(max_degree,12),domain[i:i+2]) for i in range(pieces)]
        worst=max(s['max_relative'] for c,s in fits)
        degree=max(c.shape[0] for c,s in fits)
        coeff=np.zeros((pieces,degree)+values.shape[1:],dtype='<f4')
        for i,(c,s) in enumerate(fits):
            coeff[i,:len(c)]=c
        result=(coeff,dict(pieces=pieces,degree=degree-1,max_relative=worst))
        if worst<=tolerance:
            return result
    return result


def evaluate(coeff,x):
    """FP32 Clenshaw, matching the proposed device arithmetic."""
    x=np.asarray(x,dtype=np.float32).reshape((-1,)+(1,)*(coeff.ndim-1))
    a=np.zeros((len(x),)+coeff.shape[1:],np.float32);b=np.zeros_like(a)
    for c in coeff[:0:-1]:
        a,b=2*x*a-b+c,a
    return x*a-b+coeff[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tolerance',type=float,default=.0025)
    p.add_argument('--max-degree',type=int,default=64)
    args=p.parse_args()
    if not 0<args.tolerance<.1 or not 1<=args.max_degree<=128:
        p.error('Invalid tolerance/degree')
    groups={s:[] for s in range(26)};photons={s:[] for s in range(26)}
    for line in (args.source/'charged/index.txt').read_text().splitlines()[1:]:
        s,rho,relative=line.split();node=load_charged(args.source/'charged'/relative)
        if node['section']!=int(s) or node['density']!=float(rho):
            raise ValueError('Charged index identity')
        groups[int(s)].append(node)
    for line in (args.source/'photon/index.txt').read_text().splitlines()[1:]:
        s,rho,relative=line.split()
        photons[int(s)].append(load_photon(args.source/'photon'/relative))
    arrays={};report=[];passed=True
    for section in range(26):
        nodes=sorted(groups[section],key=lambda n:n['density'])
        if not nodes or not photons[section]:
            raise ValueError('Incomplete26 composition coverage')
        energy=nodes[0]['energy'];rho=np.array([n['density'] for n in nodes])
        if any(not np.array_equal(energy,n['energy']) for n in nodes):
            raise ValueError('Charged grids differ')
        # Elastic array includes density-dependent bc/xc2; only Urban constants
        # (last56 bytes) are strictly common across all source nodes.
        if any(n['common'][-56:]!=nodes[0]['common'][-56:] for n in nodes):
            raise ValueError('Urban constants differ')
        scalar,scalar_stats=fit(rho,np.array([n['mass'] for n in nodes]),args.tolerance,args.max_degree)
        cut,cut_stats=fit_cut(rho,np.array([n['cut'] for n in nodes]),args.tolerance,args.max_degree)
        pe,pm=photons[section][0]
        perr=0.
        for e,m in photons[section][1:]:
            if not np.array_equal(e,pe):
                raise ValueError('Photon grids differ')
            perr=max(perr,float(np.max(abs(m-pm)/np.maximum(abs(pm),1e-300))))
        prefix=f'c{section:02d}_'
        arrays.update({prefix+'log_domain':np.log(rho[[0,-1]]).astype('<f4'),
                       prefix+'energy':energy.astype('<f4'),prefix+'scalar_coeff':scalar,
                       prefix+'cut_coeff':cut,prefix+'urban':np.frombuffer(nodes[0]['common'][-56:],'<f8').astype('<f4'),
                       prefix+'photon_energy':pe.astype('<f4'),prefix+'photon_mass_rate':pm.astype('<f4')})
        # Positive and monotone mass range at all density validation points.
        x=np.linspace(-1,1,4097,dtype=np.float32)
        mass_range=np.exp(evaluate(scalar,x))[:,:,2]
        monotone=bool(np.all(np.diff(mass_range,axis=1)>0))
        ok=scalar_stats['max_relative']<=args.tolerance and cut_stats['max_relative']<=args.tolerance and monotone and perr<1e-12
        passed &= ok
        row=dict(section=section,scalar=scalar_stats,cut=cut_stats,photon_mass_relative=perr,
                 range_monotone=monotone,component_pass=ok)
        report.append(row);print(json.dumps(row),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'scalar26.npz',**arrays)
    result=dict(schema='ELECTRON_SCALAR26_PROTOTYPE',compositions=26,density_response_nodes=0,
                storage_bytes=(args.output/'scalar26.npz').stat().st_size,
                component_pass=bool(passed),tolerance=args.tolerance,rows=report,
                status='SCALAR_ONLY_NOT_A_COMPLETE_ELECTRON_PACKAGE',
                excludes=['hard_process_rates','collision_outcomes','birth_yields','birth_outcomes','elastic_angular_laws','joint'],
                reference='current83-node piecewise-linear mass response; off-grid independent native extraction not yet checked')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not passed:
        raise SystemExit(2)


if __name__=='__main__':
    main()
