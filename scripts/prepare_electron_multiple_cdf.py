#!/usr/bin/env python3
"""Research fixed-count spherical convolutions of the truncated Mott law.

The l-th Legendre coefficient of N independent rotations is f_l**N. This
retains the full single-scatter angular law, rather than fitting two moments.
Fixed-energy research only, with no transport or energy interpolation hooks.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss,legvander
import electron_elastic_moments as m
from prepare_electron_inverse_cdf import probability_knots

LIMITS=dict(quadrature_max_coefficient_difference=1e-9,
    degree_doubling_max_CDF_difference=1e-6,max_CDF_repair=1e-7,
    max_relative_transport_moment_error=1e-4,max_CDF_interpolation_error=1e-4,
    max_relative_tail_error_above_1e_5=.01,max_direct_CDF_error=1e-4,
    max_direct_relative_tail_error_above_1e_5=.01)


def finalize(report,directory):
    rows=report['results'];cells=[cell for row in rows for cell in row['counts']]
    actual={key:max(row[key] for row in rows) for key in
        ['quadrature_max_coefficient_difference','degree_doubling_max_CDF_difference']}
    actual.update({key:max(row[key] for row in cells) for key in
        ['max_CDF_repair','max_CDF_interpolation_error','max_relative_tail_error_above_1e_5',
         'max_direct_CDF_error','max_direct_relative_tail_error_above_1e_5']})
    actual['max_relative_transport_moment_error']=max(abs(v) for row in cells for v in row['moment_relative_error'])
    screen=dict(passed=all(actual[k]<=LIMITS[k] for k in LIMITS),actual=actual,limits=LIMITS,
        table_bytes=report['bytes'],law_count=len(rows),count_conditions=len(cells))
    (directory/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    if not screen['passed']:raise ValueError('Angular reconstruction failed its component screen; no sampler index exported')
    if m.sha(directory/'quantiles.bin')!=report['quantiles_sha256']:raise ValueError('quantile fingerprint')
    text=f"MAIMULT1 {report['index_sha256']} {len(report['cases'])} {report['max_count']} {report['bins']} {report['knots']} quantiles.bin {report['quantiles_sha256']}\n"
    for c in report['cases']:
        text+=' '.join(str(c[k]) for k in ['node','energy','a','cutoff','lower_probability','mean_q','second_q','lower_row'])+'\n'
    (directory/'index.txt').write_text(text)

def coefficients(node,case,degree,order):
    a=float(np.float32(case['a']));cut=float(np.float32(case['cutoff']))
    probability=float(np.float32(case['lower_probability']));lo=case['lower_row']
    rows=[node['table'][lo],node['table'][min(lo+1,30)]]
    theta=2*np.arcsin(np.sqrt(cut/2))
    # Resolve high-order Legendre oscillations also at the cutoff; logarithmic
    # Rutherford subdivision alone is not an adequate high-l quadrature rule.
    boundaries=np.unique(np.r_[m.edges(a,0,cut),2*np.sin(np.linspace(0,theta,max(256,math.ceil(degree*theta/math.pi))) / 2)**2])
    boundaries=np.clip(boundaries,0,cut);u=boundaries*(1+a)/(2*a+boundaries)
    gx,gw=leggauss(order);half=np.diff(u)/2
    sample=(u[:-1,None]+half[:,None])+half[:,None]*gx
    q=(2*a*sample/(1+a-sample)).ravel();w=(half[:,None]*gw).ravel()
    acceptance=np.array([m.acceptance(row,q) for row in rows]);norm=acceptance@w
    mixed=(np.array([probability,1-probability])/norm)@acceptance
    weights=w*mixed
    values=legvander(1-q,degree)
    result=weights@values
    if abs(result[0]-1)>1e-12 or np.max(abs(result))>1+1e-12:raise ValueError('Legendre normalization')
    return result,dict(quadrature_points=len(q),normalizers=norm.tolist())


def cdf_from_coefficients(coeff,q,counts):
    # Integrated Legendre expansion in the standard P_l basis. This form
    # evaluates all requested integer counts with one dense matrix product.
    degree=len(coeff)-1;weights=np.zeros((degree+2,len(counts)))
    weights[0]=.5;weights[1]=-.5
    powered=coeff[1:,None]**np.asarray(counts)[None,:]
    weights[:-2]+=.5*powered;weights[2:]-=.5*powered
    return legvander(1-q,degree+1)@weights


def prepare(node,case,degree,bins,max_count,grid_points):
    coeff,meta=coefficients(node,case,degree,32)
    check,_=coefficients(node,case,degree,16)
    counts=np.arange(2,max_count+1);cut=float(np.float32(case['cutoff']))
    theta=2*np.arcsin(np.sqrt(cut/2));support=2*np.sin(np.minimum(counts*theta,np.pi)/2)**2
    q=np.unique(np.r_[0,np.geomspace(1e-13,2,grid_points),support,2])
    cdf=cdf_from_coefficients(coeff,q,counts)
    coarse=cdf_from_coefficients(coeff[:degree//2+1],q,counts)
    convergence=float(np.max(abs(cdf-coarse)))
    quadrature=float(np.max(abs(coeff-check)))
    p=probability_knots(bins);tables=[];reports=[]
    for col,n in enumerate(counts):
        select=q<=support[col];x=q[select];v=cdf[select,col].copy()
        raw=v.copy();v[0]=0;v[-1]=1
        # Record and tightly bound every monotonicity repair. This is numerical
        # reconstruction tolerance, not permission to truncate physical tails.
        v=np.maximum.accumulate(np.clip(v,0,1))
        repair=float(np.max(abs(v-raw)))
        outside=float(np.max(abs(cdf[q>support[col],col]-1))) if np.any(q>support[col]) else 0.
        quantile=np.interp(p,v,x).astype(np.float32);quantile[0]=0;quantile[-1]=np.float32(support[col])
        first=float(np.dot(np.diff(p),(quantile[:-1].astype(float)+quantile[1:])/2))
        second=float(np.dot(np.diff(p),(quantile[:-1].astype(float)**2+quantile[:-1].astype(float)*quantile[1:]+quantile[1:].astype(float)**2)/3))
        expected1=-np.expm1(n*np.log(coeff[1]));expected2=-np.expm1(n*np.log(coeff[2]))
        errs=[first/expected1-1,(3*first-1.5*second)/expected2-1]
        # Mid-bin checks evaluate against the independent converged spectral
        # CDF grid, not against the fitted inverse-CDF table itself.
        pp=(p[:-1,None]+np.diff(p)[:,None]*np.array([.125,.375,.625,.875])).ravel()
        qq=(quantile[:-1,None].astype(float)+(quantile[1:]-quantile[:-1])[:,None]*np.array([.125,.375,.625,.875])).ravel()
        residual=np.interp(qq,q,cdf[:,col])-pp
        resolved=np.minimum(pp,1-pp)>=1e-5
        direct_p=np.r_[np.geomspace(1e-5,.1,257),np.linspace(.1,.9,129),1-np.geomspace(1e-5,.1,257)]
        direct_q=np.interp(direct_p,p,quantile).astype(np.float32).astype(float)
        direct_residual=cdf_from_coefficients(coeff,direct_q,[n])[:,0]-direct_p
        report=dict(count=int(n),max_CDF_repair=repair,outside_support_CDF_error=outside,moment_relative_error=errs,max_CDF_interpolation_error=float(abs(residual).max()),max_relative_tail_error_above_1e_5=float(np.max(abs(residual[resolved])/np.minimum(pp[resolved],1-pp[resolved]))),support=float(support[col]))
        report.update(max_direct_CDF_error=float(abs(direct_residual).max()),
            max_direct_relative_tail_error_above_1e_5=float(np.max(abs(direct_residual)/np.minimum(direct_p,1-direct_p))),
            direct_spectral_points=len(direct_p))
        reports.append(report);tables.append(quantile)
    summary=dict(node=case['node'],energy=case['energy'],degree=degree,quadrature_max_coefficient_difference=quadrature,degree_doubling_max_CDF_difference=convergence,**meta,counts=reports)
    return np.array(tables),coeff,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--degree',type=int,default=2048);parser.add_argument('--bins',type=int,default=32)
    parser.add_argument('--max-count',type=int,default=32);parser.add_argument('--grid-points',type=int,default=8193)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    if args.degree<128 or args.degree&(args.degree-1) or not 2<=args.max_count<=128 or args.bins!=32 or args.grid_points<4097:
        parser.error('require power-of-two degree>=128, counts2..128, bins32 and grid-points>=4097')
    nodes=m.read_nodes(args.index,args.sha256);source=json.loads(args.cases.read_text());unique={(r['node'],r['energy']):r for r in source};cases=list(unique.values())
    if args.pilot:cases=[unique[(32,.1)]]
    args.out.mkdir(parents=True,exist_ok=True);tables=[];reports=[];spectra=[]
    for i,case in enumerate(cases):
        table,coeff,report=prepare(nodes[case['node']],case,args.degree,args.bins,args.max_count,args.grid_points)
        tables.append(table);reports.append(report);spectra.append(coeff)
        print(json.dumps(dict(index=i,node=case['node'],energy=case['energy'],quadrature_error=report['quadrature_max_coefficient_difference'],CDF_convergence=report['degree_doubling_max_CDF_difference'],max_repair=max(r['max_CDF_repair'] for r in report['counts']),max_moment_error=max(abs(v) for r in report['counts'] for v in r['moment_relative_error']))),flush=True)
    array=np.asarray(tables,dtype='<f4');array.tofile(args.out/'quantiles.bin');np.asarray(spectra,dtype='<f8').tofile(args.out/'legendre_coefficients.bin')
    probability_knots(args.bins).astype('<f8').tofile(args.out/'probabilities.bin')
    summary=dict(scope='Fixed-count, fixed-energy soft Mott spherical convolution; no CT or runtime energy interpolation acceptance.',index_sha256=args.sha256,cases=cases,bins=args.bins,knots=array.shape[-1],max_count=args.max_count,degree=args.degree,bytes=array.nbytes,quantiles_sha256=m.sha(args.out/'quantiles.bin'),results=reports)
    (args.out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    finalize(summary,args.out)

if __name__=='__main__':main()
