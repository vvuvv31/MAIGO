#!/usr/bin/env python3
"""Build explicit26-composition birth and collision component candidates.

Density dependence uses positive piecewise functional control distributions,
not native material-density response records. Joint sparse cells retain all positive
branches; each cell's conditional energy/angle means are stored together.
This is a new discretization, not a lossless encoding or dose acceptance.
"""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np
from numpy.polynomial import Polynomial
from assemble_electron_birth_bank import CHANNEL,SAMPLE
from compact_electron_runtime import COMPACT

TS=np.array([.125,.375,.625,.875])
INV=np.linalg.inv(np.vander(TS,4,increasing=True))


def controls(rho,degree):
    degree=0 if len(rho)==1 else 1 if len(rho)==2 else degree
    query=np.exp(np.linspace(np.log(rho[0]),np.log(rho[-1]),degree+1))
    query[0]=rho[0];query[-1]=rho[-1]
    mix=np.zeros((degree+1,len(rho)))
    for j,r in enumerate(query):
        hi=min(int(np.searchsorted(rho,r)),len(rho)-1);lo=max(0,hi-1)
        f=0 if lo==hi else (r-rho[lo])/(rho[hi]-rho[lo])
        mix[j,lo]+=1-f;mix[j,hi]+=f
    return query,mix


def birth_nodes(path):
    result=[]
    with path.open('rb') as f:
        magic,version,count,z,a=struct.unpack('<8s4I',f.read(24))
        if (magic,version)!=(b'MAIBIR02',2) or (z,a) not in [(1,1),(6,12)]:raise ValueError('birth schema/species')
        for _ in range(count):
            section,density,n,bins=struct.unpack('<idQI',f.read(24))
            if not 0<=section<26 or not np.isfinite(density) or density<=0 or bins<1:
                raise ValueError('birth material/domain')
            channels=np.frombuffer(f.read(40*bins),CHANNEL).copy()
            states=np.memmap(path,SAMPLE,mode='r',offset=f.tell(),shape=(n,))
            result.append(dict(section=section,density=density,channels=channels,states=states,z=z,a=a))
            f.seek(f.tell()+24*n)
        if f.read(1):raise ValueError('birth trailing bytes')
    return result


def collision_nodes(root,kind):
    result=[]
    for line in (root/'index.txt').read_text().splitlines()[1:]:
        section,density,relative=line.split();path=root/relative
        with path.open('rb') as f:
            if kind=='charged':
                h=struct.unpack('<8sIid4IQ4d',f.read(80))
                if h[0]!=b'MAIECF01':raise ValueError('compact charged required')
                f.seek(9600+48*h[4])
                rates=[np.frombuffer(f.read(48*n),'<f8').reshape(-1,6).copy() for n in h[6:8]]
                bins,nout=h[5],h[8]
            else:
                h=struct.unpack('<8sIidIIQ',f.read(40))
                if h[0]!=b'MAIPHF01':raise ValueError('compact photon required')
                rate_rows=np.frombuffer(f.read(32*h[4]),'<f8').reshape(-1,4).copy()
                bins,nout=h[5],h[6];rates=None
            edges=np.frombuffer(f.read(8*(bins+1)),'<f8').copy()
            channels=np.frombuffer(f.read(16*bins*(2 if kind=='charged' else 3)),
                dtype=[('offset','<u8'),('count','<u8')]).copy()
            states=np.memmap(path,COMPACT,mode='r',offset=f.tell(),shape=(nout,))
            if h[2]!=int(section) or h[3]!=float(density):raise ValueError('collision index identity')
            if f.tell()+24*nout!=path.stat().st_size:raise ValueError('collision payload size')
            if not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):raise ValueError('collision energy grid')
        result.append(dict(section=int(section),density=float(density),rates=rates,edges=edges,
                           channels=channels,states=states,
                           photon_rates=rate_rows if kind=='photon' else None))
    return result


def raw_rate(rows,energy):
    i=np.clip(np.searchsorted(rows[:,0],energy,side='right')-1,0,len(rows)-1)
    selected=rows[i];t=(energy-selected[...,0])/(selected[...,1]-selected[...,0])
    coefficients=selected[...,2:]@INV.T
    return ((coefficients[...,3]*t+coefficients[...,2])*t+coefficients[...,1])*t+coefficients[...,0]


def hard_model(nodes,mix):
    result={}
    for p in range(2):
        edges=np.unique(np.concatenate([n['rates'][p][:,:2].ravel() for n in nodes]))
        probes=edges[:-1,None]+np.diff(edges)[:,None]*TS
        native=np.array([raw_rate(n['rates'][p],probes)/n['density'] for n in nodes])
        values=np.tensordot(mix,native,axes=(1,0)).astype('<f4')
        # |L_j(t)| max over [0,1], then the density convex hull; 1% FP32 reserve.
        basis=[]
        for j in range(4):
            poly=Polynomial([1.])
            for k in range(4):
                if j!=k:poly*=Polynomial([-TS[k],1])/(TS[j]-TS[k])
            roots=poly.deriv().roots()
            t=[0.,1.]+[float(x.real) for x in roots if abs(x.imag)<1e-12 and 0<x.real<1]
            basis.append(max(abs(poly(x)) for x in t))
        upper=float(np.max(np.sum(abs(values.astype(float))*basis,axis=-1)))
        majorant=np.nextafter(np.float32(max(upper*1.02,1e-30)),np.float32(np.inf))
        result[f'rate{p}_edges']=edges.astype('<f4');result[f'rate{p}_values']=values
        result[f'rate{p}_majorant']=np.array([majorant],'<f4')
    return result


def joint_channel(blocks,mix,scale,energy_bins,angle_bins,birth=False):
    keys=[];prob=[];energies=[];angles=[];owners=[]
    for node,b in enumerate(blocks):
        if not len(b):continue
        if birth:
            if np.any(b['cdf'][1:]<b['cdf'][:-1]) or b['cdf'][-1]!=1 or b['cdf'][0]<0:
                raise ValueError('birth CDF')
            weight=np.diff(np.r_[0.,b['cdf']]);energy=b['energy'];pdg=np.full(len(b),11)
        else:
            if np.any(b['cdf'][1:]<b['cdf'][:-1]) or b['cdf'][-1]!=np.iinfo(np.uint64).max:
                raise ValueError('collision CDF')
            weight=np.diff(np.r_[np.uint64(0),b['cdf']]).astype(np.float64)/float(np.iinfo(np.uint64).max)
            energy=b['fraction'];pdg=b['pdg']
        angle=b['cosine'];valid=weight>0
        if not np.isfinite(energy).all() or not np.isfinite(angle).all() or np.any(abs(angle)>1) or not np.isin(pdg,[0,11,22]).all():
            raise ValueError('joint state fields')
        weight,energy,angle,pdg=[x[valid] for x in (weight,energy,angle,pdg)]
        # Log energy retains small continuations; angular endpoints remain.
        low,high=(1e-7,.8) if birth else (1e-12,1.)
        if np.any(energy<=0) or np.any(energy>high*(1+1e-6)):raise ValueError('joint energy domain')
        ei=np.clip(((np.log(energy)-math.log(low))/math.log(high/low)*energy_bins).astype(int),0,energy_bins-1)
        ai=np.clip(((angle+1)*.5*angle_bins).astype(int),0,angle_bins-1)
        code=np.where(pdg==0,0,np.where(pdg==11,1,2))
        keys.append((code*energy_bins+ei)*angle_bins+ai)
        prob.append(weight);energies.append(energy);angles.append(angle);owners.append(np.full(len(weight),node))
    if not keys:return None
    key,inverse=np.unique(np.concatenate(keys),return_inverse=True)
    weight=np.concatenate(prob);energy=np.concatenate(energies);angle=np.concatenate(angles);owner=np.concatenate(owners)
    arrays=[]
    flat=owner*len(key)+inverse
    for state in (np.ones(len(weight)),energy,angle):
        mass=np.bincount(flat,weights=weight*state,minlength=len(blocks)*len(key)).reshape(len(blocks),-1)
        arrays.append(np.tensordot(mix,mass*scale[:,None],axes=(1,0)))
    probability,emoment,amoment=arrays
    total=probability.sum(axis=1)
    active=probability>0
    conditional=np.divide(probability,total[:,None],out=np.zeros_like(probability),where=total[:,None]>0)
    centroid_e=np.divide(emoment,probability,out=np.ones_like(probability),where=active).astype('<f4')
    centroid_mu=np.divide(amoment,probability,out=np.zeros_like(probability),where=active).astype('<f4')
    cumulative=np.cumsum(conditional.astype(np.longdouble),axis=1)
    threshold=np.minimum(cumulative*np.longdouble(2)**64,np.longdouble(2)**64-1).astype('<u8')
    for i in range(len(total)):
        if total[i]>0:threshold[i,np.flatnonzero(active[i])[-1]:]=np.iinfo(np.uint64).max
    return dict(key=key.astype('<u4'),cdf=threshold,energy=centroid_e,cosine=centroid_mu,
                total=total.astype('<f4'),pdg=np.array([0,11,22],'<i4')[key//(energy_bins*angle_bins)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/archive/electron_density_nodes_20261006'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--components',nargs='+',choices=['proton','carbon','charged','photon'],default=['proton','carbon','charged','photon'])
    p.add_argument('--sections',type=int,nargs='+',default=list(range(26)))
    p.add_argument('--density-degree',type=int,default=32)
    p.add_argument('--energy-bins',type=int,default=128);p.add_argument('--angle-bins',type=int,default=256)
    args=p.parse_args()
    if not 1<=args.density_degree<=64 or not 16<=args.energy_bins<=1024 or not 16<=args.angle_bins<=1024:
        p.error('Invalid representation size')
    if len(set(args.sections))!=len(args.sections) or any(s<0 or s>25 for s in args.sections):p.error('Invalid sections')
    args.output.mkdir(parents=True,exist_ok=False);reports=[]
    for component in args.components:
        nodes=birth_nodes(args.source/component/'birth_bank.bin') if component in ('proton','carbon') else collision_nodes(args.source/'shared'/component,component)
        directory=args.output/component;directory.mkdir()
        for section in args.sections:
            group=sorted([n for n in nodes if n['section']==section],key=lambda n:n['density'])
            if not group:raise ValueError('Incomplete compositions')
            rho=np.array([n['density'] for n in group]);query,mix=controls(rho,args.density_degree)
            if len(rho)>1 and np.any(np.diff(rho)<=0):raise ValueError('duplicate or unordered source density')
            birth=component in ('proton','carbon');nchannels=len(group[0]['channels'])
            if any(len(n['channels'])!=nchannels for n in group):raise ValueError('Channel mismatch')
            if not birth and any(not np.array_equal(n['edges'],group[0]['edges']) for n in group):
                raise ValueError('Conditional energy grids differ')
            arrays=dict(density_domain=rho[[0,-1]].astype('<f4'),
                        density_degree=np.array([len(query)-1],'<u4'))
            if birth:arrays['parent_edges']=np.r_[group[0]['channels']['low'],group[0]['channels']['high'][-1]].astype('<f4')
            else:arrays['energy_edges']=group[0]['edges'].astype('<f4')
            if component=='charged':arrays.update(hard_model(group,mix))
            if component=='photon':
                rates=group[0]['photon_rates'].copy();rates[:,1:]/=rho[0];arrays['mass_rates']=rates.astype('<f4')
            cell_count=0
            for channel in range(nchannels):
                blocks=[]
                for n in group:
                    c=n['channels'][channel];blocks.append(n['states'][int(c['offset']):int(c['offset']+c['count'])])
                scale=np.array([n['channels'][channel]['fraction'] for n in group]) if birth else np.ones(len(group))
                # All control distributions retain positive-support channels,
                # including channels with a zero midpoint rate. The runtime
                # selects charged controls by their actual-energy process rate.
                # Reconstructing an interior control geometrically remains a
                # separate approximation when a composition has >2 sources.
                model=joint_channel(blocks,mix,scale,args.energy_bins,args.angle_bins,birth)
                if model:
                    for key,value in model.items():arrays[f'ch{channel:04d}_{key}']=value
                    cell_count+=len(model['key'])
            path=directory/f'composition{section:02d}.npz';np.savez_compressed(path,**arrays)
            row=dict(component=component,section=section,degree=len(query)-1,channels=nchannels,
                     joint_cells=cell_count,bytes=path.stat().st_size,array_bytes=sum(x.nbytes for x in arrays.values()))
            reports.append(row);print(json.dumps(row),flush=True)
    report=dict(schema='ELECTRON_RESPONSE26_CANDIDATE1',status='COMPONENT_CANDIDATE_NOT_GPU_OR_DOSE_ACCEPTANCE',
                energy_bins=args.energy_bins,angle_bins=args.angle_bins,
                native_density_response_records=0,composition_sections=args.sections,components=args.components,
                density_representation='continuous piecewise affine physical-density response on implicit uniform log-density partitions; no native density knots',
                joint_representation='all positive joint cells, conditional first moments; finite-cell approximation',
                charged_outcome_conditioning='actual-energy rate-weighted control selection; interior control distributions are reconstructed geometrically',
                storage_bytes=sum(x['bytes'] for x in reports),array_bytes=sum(x['array_bytes'] for x in reports),rows=reports)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
