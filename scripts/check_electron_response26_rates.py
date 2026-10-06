#!/usr/bin/env python3
"""Certify every FP32 control cubic with the existing outward-rounded bound."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    files=sorted((args.candidate/'charged').glob('composition*.npz'))
    if len(files)!=26:raise ValueError('Require all26 compositions')
    fixture=args.output/'rates.bin'
    with fixture.open('xb') as f:
        f.write(struct.pack('<8sI',b'ER26RT01',2*len(files)))
        for path in files:
            with np.load(path,allow_pickle=False) as model:
                for process in range(2):
                    edges=model[f'rate{process}_edges'];values=model[f'rate{process}_values']
                    majorant=float(model[f'rate{process}_majorant'][0])
                    if values.shape[1:]!=(len(edges)-1,4):raise ValueError('Rate tensor dimensions')
                    f.write(struct.pack('<2If',len(values),len(edges)-1,majorant))
                    f.write(edges.astype('<f4').tobytes());f.write(values.astype('<f4').tobytes())
    root=Path(__file__).resolve().parents[1];binary=args.output/'check'
    subprocess.run(['g++','-O3','-std=c++20','-I'+str(root/'include'),
                    str(root/'scripts/check_electron_response26_rates.cpp'),'-o',str(binary)],check=True)
    run=subprocess.run([str(binary),str(fixture)],capture_output=True,text=True)
    (args.output/'check.log').write_text(run.stdout+run.stderr)
    if run.stdout:
        result=json.loads(run.stdout);result['returncode']=run.returncode
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if run.returncode:raise SystemExit(run.returncode)


if __name__=='__main__':main()
