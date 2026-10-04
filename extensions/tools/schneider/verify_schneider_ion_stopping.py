#!/usr/bin/env python3
"""Verify final-combination ion stopping data without a transport run."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from compile_schneider_ion_stopping import SPECIES,QUANTITY
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);a=p.parse_args()
b=a.binary;m=json.loads(b.with_suffix('.metadata.json').read_text())
assert m['data_size_bytes']==b.stat().st_size
assert m['quantity']==QUANTITY and [tuple(x) for x in m['species_za']]==SPECIES
with b.open('rb') as f:
 assert f.read(8)==b'SCHNIOSP'
 assert struct.unpack('<IIII',f.read(16))==(1,25,18,60002)
 assert np.allclose(struct.unpack('<ddd',f.read(24)),[.01,6000.11,.1],rtol=0,atol=1e-10)
 rho=np.fromfile(f,'<f8',25);v=np.fromfile(f,'<f8')
assert len(v)==25*18*60002 and np.all(np.isfinite(v)&(v>0))
assert np.all(np.isfinite(rho)&(rho>0))
assert m['c12_cross_check']['max_relative_difference']<=.02
assert m['c12_cross_check']['minimum_energy_MeVu']==5
print('SCHNIOSP: size, schema, 25 sections, 18 species, 60002 energies and C12 gate PASS')
