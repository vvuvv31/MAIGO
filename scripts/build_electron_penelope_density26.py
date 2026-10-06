#!/usr/bin/env python3
"""Fit continuous oscillator response, preserving the physical conductor onset.

26 composition entries; no native density-node directory. Held-out native
runs test each smooth regime. This does not qualify hard rates or GPU dose.
"""
import argparse,json
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebfit,chebval
from check_electron_penelope_composition import OSC
FIELDS=[('strength','strength'),('ionization','ionization_MeV'),('resonance','resonance_MeV'),('recoil_cut','recoil_cut_MeV'),('binding','binding_MeV'),('z','z'),('shell_flag','shell_flag')]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);records=json.loads((a.native/'report.json').read_text())['records'];arrays={};rows=[]
    for section in range(26):
        group=sorted([r for r in records if r['section']==section],key=lambda r:r['density']);phases={}
        for r in group:
            d=Path(r['directory']);table=np.atleast_1d(np.genfromtxt(d/'penelope_oscillators.csv',delimiter=',',names=True));w=np.genfromtxt(d/'penelope_weights.csv',delimiter=',',names=True)
            delta=w[w['oscillator']==0]['density_correction'][:34]
            phase=int(np.any(table['ionization_MeV']==0)) if section==1 else 0
            phases.setdefault(phase,[]).append((r,table,delta))
        if len(phases)!=(2 if section==1 else 1):raise ValueError('Unexpected physical regimes')
        for phase,data in sorted(phases.items()):
            reference=data[0][1];fixed=['strength','ionization_MeV','binding_MeV','z','shell_flag']
            if any(any(not np.array_equal(reference[f],t[f]) for f in fixed) for r,t,d in data):raise ValueError(f'Unmodeled oscillator topology section{section} phase{phase}')
            prefix=f'c{section:02d}_p{phase}_';base=np.empty(len(reference),OSC)
            for dst,src in FIELDS:base[dst]=reference[src]
            logrho=np.log([r['density'] for r,t,d in data]);domain=logrho[[0,-1]]
            x=np.zeros(len(data)) if domain[0]==domain[1] else 2*(logrho-domain[0])/(domain[1]-domain[0])-1
            # Nine Lobatto points are training; uniformly spaced midpoints are
            # independent checks. Fixed-water entries are exact constants.
            target=-np.cos(np.linspace(0,np.pi,9));training=np.min(abs(x[:,None]-target),axis=1)<2e-7
            if len(data)==1:training[:]=True
            if training.sum()!=(1 if len(data)==1 else 9):raise ValueError('Missing Lobatto training points')
            scalar=np.array([[np.log(t['resonance_MeV']),np.log(t['recoil_cut_MeV'])] for r,t,d in data]).transpose(0,2,1)
            delta=np.array([d for r,t,d in data]);degree=0 if len(data)==1 else 8
            coeff=chebfit(x[training],scalar[training].reshape(training.sum(),-1),degree).reshape(degree+1,len(reference),2).astype('<f4')
            dc=chebfit(x[training],delta[training],degree).astype('<f4')
            predicted=chebval(x.astype('f4'),coeff).transpose(2,0,1)
            error=np.max(abs(np.expm1(predicted-scalar)),axis=(1,2));dp=chebval(x.astype('f4'),dc).T
            delta_error=np.max(abs(dp-delta),axis=1)
            hold=~training;worst=float(error[hold].max()) if hold.any() else float(error.max());dw=float(delta_error[hold].max()) if hold.any() else float(delta_error.max())
            passed=worst<.0025 and dw<1e-5
            arrays.update({prefix+'base':base,prefix+'log_domain':domain.astype('<f4'),prefix+'resonance_recoil_coeff':coeff,prefix+'delta_coeff':dc})
            rows.append(dict(section=section,phase=phase,training=int(training.sum()),heldout=int(hold.sum()),maximum_heldout_relative=worst,maximum_delta_absolute=dw,component_pass=passed))
    np.savez_compressed(a.out/'penelope_density26.npz',**arrays)
    report=dict(status='OSCILLATOR_DENSITY_COMPONENT_NOT_RATE_OR_GPU_ACCEPTANCE',compositions=26,physical_regimes=len(rows),runtime_density_nodes=0,component_pass=all(r['component_pass'] for r in rows),rows=rows)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    if not report['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
