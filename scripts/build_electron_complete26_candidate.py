#!/usr/bin/env python3
"""Dose-first independent26 candidate; explicitly approximate photon terminals.

Old banks are offline extraction inputs ONLY. The candidate never opens them.
No83 directory is copied. Photon conditional laws use one calibration per
composition and density-scaled rates; cuts/relaxation remain calibration laws.
"""
import argparse,json,struct,shutil
from pathlib import Path
import numpy as np
from build_electron_scalar26 import load_charged
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 for name in ['scalar26.bin','penelope26.bin','atomic26.bin','brems26.bin','proton26.bin','carbon26.bin']:
  shutil.copy2(ROOT/'out/electron_complete26_20261006/components_v4'/name,a.out/name)
 shutil.copy2(ROOT/'out/electron_composition26_20261006/package/rates26.bin',a.out/'rates26.bin')
 shutil.copy2(ROOT/'out/electron_cut_conditioned26_20261005/package/birth26.bin',a.out/'birth26.bin')
 charged={s:[] for s in range(26)};photon={s:[] for s in range(26)}
 for kind,groups in [('charged',charged),('photon',photon)]:
  base=ROOT/'data/archive/electron_density_nodes_20261006/shared'/kind
  for line in (base/'index.txt').read_text().splitlines()[1:]:
   s,rho,file=line.split();groups[int(s)].append((float(rho),base/file))
 ci=['MAIEC026 26'];pi=['MAIPH026 26'];records=[]
 for s in range(26):
  cr=sorted(charged[s])[len(charged[s])//2];pr=min(photon[s],key=lambda x:abs(x[0]-cr[0]))
  data=load_charged(cr[1]);name=f'elastic{s:02d}.bin'
  # Exact composition Urban constants and canonical angular law. No old
  # stopping/range/rate/charged-terminal payload enters this auxiliary file.
  (a.out/name).write_bytes(struct.pack('<8sIIf',b'MAIEL026',1,s,cr[0])+data['common'])
  pn=f'photon{s:02d}.bin';shutil.copy2(pr[1],a.out/pn)
  ci.append(f'{s} {cr[0]:.17g} {name}');pi.append(f'{s} {pr[0]:.17g} {pn}')
  records.append(dict(section=s,calibration_density=cr[0],photon_calibration_density=pr[0]))
 (a.out/'charged.index').write_text('\n'.join(ci)+'\n');(a.out/'photon.index').write_text('\n'.join(pi)+'\n')
 # The spatial joint operator already contains25 compositions, not83 states.
 # Include it in the one shared package; hardlink avoids an unnecessary copy.
 (a.out/'joint.bin').hardlink_to(ROOT/'data/archive/electron_density_nodes_20261006/shared/joint.bin')
 report=dict(status='DOSE_FIRST_26_CANDIDATE_NOT_DEFAULT',compositions=26,runtime_density_nodes=0,
  approximations=['native-grid hard rates mass-scaled at actual cut; density correction deferred','Penelope oscillators at composition calibration density','photon calibrated conditional terminals with continuous mass-scaled rates'],
  shared_bytes=sum(x.stat().st_size for x in a.out.iterdir() if x.is_file() and x.name not in ['proton26.bin','carbon26.bin']),rows=records)
 (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
