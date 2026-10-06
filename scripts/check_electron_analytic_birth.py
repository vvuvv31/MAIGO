#!/usr/bin/env python3
"""Test the FP32 rejection sampler against independent double quadrature.

This checks the proposed law's implementation, not agreement with TOPAS dose.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from probe_electron_birth_analytic26 import born_joint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];binary=args.output/'check'
    subprocess.run(['g++','-O3','-std=c++20','-I'+str(root/'include'),
                    str(root/'scripts/check_electron_analytic_birth.cpp'),'-o',str(binary)],check=True)
    run=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    (args.output/'sampling.txt').write_text(run.stdout)
    rows=[]
    for line in run.stdout.splitlines():
        a,parent,cut,n,invalid,se,sm,see,smm,attempts=map(float,line.split())
        e,mu,w=born_joint(parent,parent,cut,int(a),transfer_points=1024)
        errors=[]
        for total,square,predicted in [(se,see,float(w@e)),(sm,smm,float(w@mu))]:
            mean=total/n;variance=(square/n-mean**2)*n/(n-1)
            error=abs(mean-predicted)/np.sqrt(variance/n)
            errors.append(float(error))
        rows.append(dict(projectile_a=int(a),parent_MeVu=parent,cut_MeV=cut,
                         samples=int(n),invalid=int(invalid),mean_energy_z=errors[0],
                         mean_cosine_z=errors[1],attempts_per_sample=attempts/n))
    passed=all(r['invalid']==0 and max(r['mean_energy_z'],r['mean_cosine_z'])<5 for r in rows)
    result=dict(component_pass=passed,samples=sum(r['samples'] for r in rows),rows=rows,
                status='SAMPLER_LAW_CHECK_ONLY_NOT_NATIVE_YIELD_OR_DOSE_ACCEPTANCE')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(component_pass=passed,samples=result['samples'],
                         max_z=max(max(r['mean_energy_z'],r['mean_cosine_z']) for r in rows)),indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
