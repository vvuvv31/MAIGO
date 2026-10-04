#!/usr/bin/env python3
"""Independent off-grid continuous-energy check for screened Mott quantiles.

Tests the actual order: sample each endpoint table at p, transform to screened
probability/complement, interpolate, then invert with the target screening.
This is an angular component screen, not transport or dose acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import electron_elastic_moments as m


def interpolate(q0, q1, a0, a1, a, weight, p, fp32=False):
    dtype = np.float32 if fp32 else np.float64
    q0, q1 = q0.astype(dtype), q1.astype(dtype)
    a0, a1, a, weight = map(dtype, (a0,a1,a,weight))
    u0=q0*(1+a0)/(2*a0+q0);u1=q1*(1+a1)/(2*a1+q1)
    u=u0+weight*(u1-u0)
    v0=a0*(2-q0)/(2*a0+q0);v1=a1*(2-q1)/(2*a1+q1)
    v=v0+weight*(v1-v0)
    return np.where(p>.5, 2*a*(1-v)/(a+v), 2*a*u/(1-u+a))


def theory(node, energy, qtest, order=32):
    a, low, weight = m.angular_parameters(node, energy)
    rows=[node['table'][low],node['table'][min(low+1,30)]]
    base=m.integrate(a,rows,m.edges(a),order).sum(axis=1)
    mix=np.array([1-weight,weight])/base[:,0]
    expected=(base*mix[:,None]).sum(axis=0)
    bounds=np.unique(np.r_[m.edges(a),qtest])
    pieces=m.integrate(a,rows,bounds,order)
    cdf=np.r_[0,np.cumsum(np.einsum('rij,r->ij',pieces,mix)[:,0])]
    return expected,np.interp(qtest,bounds,cdf)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tables',type=Path)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--nodes',default='12,32,57')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    src=json.loads((args.tables/'report.json').read_text())
    nodes=m.read_nodes(args.index,src['index_sha256'])
    selected=set(range(82)) if args.nodes=='all' else set(map(int,args.nodes.split(',')))
    tables=np.fromfile(args.tables/'quantiles.bin','<f4').reshape(-1,src['knots'])
    prob=np.fromfile(args.tables/'probabilities.bin','<f8')
    gx,gw=leggauss(8)
    dp=np.diff(prob)
    pq=(prob[:-1,None]+dp[:,None]*(gx+1)/2).ravel()
    pw=(dp[:,None]*gw/2).ravel()
    pc=(prob[:-1]+prob[1:])/2
    rows=[];inputs=[]
    for i in range(len(src['cases'])-1):
        left,right=src['cases'][i:i+2]
        if left['node']!=right['node'] or left['node'] not in selected:continue
        q0=np.interp(pq,prob,tables[i]);q1=np.interp(pq,prob,tables[i+1])
        t0=np.interp(pc,prob,tables[i]);t1=np.interp(pc,prob,tables[i+1])
        for weight in (.2113248654051871,.5,.7886751345948129):
            energy=float(np.exp((1-weight)*np.log(left['energy'])+weight*np.log(right['energy'])))
            a=m.angular_parameters(nodes[left['node']],energy)[0]
            q=interpolate(q0,q1,left['screening'],right['screening'],a,weight,pq)
            t=interpolate(t0,t1,left['screening'],right['screening'],a,weight,pc)
            expected,cdf=theory(nodes[left['node']],energy,t)
            observed=np.array([pw.sum(),pw@q,pw@(3*q-1.5*q*q)])
            tf=interpolate(t0,t1,left['screening'],right['screening'],a,weight,pc,True).astype(float)
            if np.any(np.diff(tf)<0) or np.min(tf)<0 or np.max(tf)>2 or not np.isfinite(tf).all():
                raise ValueError('FP32 map validity/monotonicity')
            _,cf=theory(nodes[left['node']],energy,tf)
            residual=cf-pc;tail=np.minimum(pc,1-pc);resolved=tail>=1e-4
            row=dict(node=left['node'],energy=energy,low=i,high=i+1,weight=weight,
                screening=a,screening_low=left['screening'],screening_high=right['screening'],
                moment_relative_error=(observed/expected-1).tolist(),
                maximum_cdf_residual=float(np.max(abs(cdf-pc))),
                fp32_maximum_cdf_residual=float(np.max(abs(residual))),
                fp32_resolved_tail_relative_error=float(np.max(abs(residual[resolved])/tail[resolved])),
                fp32_vs_double_maximum_absolute_q=float(np.max(abs(tf-t))))
            rows.append(row);inputs.append(f"{left['node']} {energy} {i} {i+1} {left['screening']} {right['screening']} {weight}\n")
    gates=dict(relative_moment=1e-3,absolute_cdf=2e-4,relative_tail_above_1e_4=.01)
    failures=[r for r in rows if max(abs(v) for v in r['moment_relative_error'])>gates['relative_moment'] or r['fp32_maximum_cdf_residual']>gates['absolute_cdf'] or r['fp32_resolved_tail_relative_error']>gates['relative_tail_above_1e_4']]
    result=dict(scope='off-grid angular component only, before GPU interpolation and transport validation',
        index_sha256=src['index_sha256'],table_path=str(args.tables),gates=gates,
        cases=len(rows),failures=[dict(node=r['node'],energy=r['energy']) for r in failures],
        maximum_relative_moment_error=max(max(abs(v) for v in r['moment_relative_error']) for r in rows),
        maximum_cdf_residual=max(r['maximum_cdf_residual'] for r in rows),
        fp32_maximum_cdf_residual=max(r['fp32_maximum_cdf_residual'] for r in rows),
        fp32_resolved_tail_relative_error=max(r['fp32_resolved_tail_relative_error'] for r in rows),
        results=rows)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'cases.txt').write_text(str(len(rows))+'\n'+''.join(inputs))
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
    if failures:raise SystemExit('component representation screen failed')


if __name__=='__main__':main()
