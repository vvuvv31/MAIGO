#!/usr/bin/env python3
"""Formal broadbeam plus matched head screens for collision representation changes."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import validate_em_density_factorization as v
from validate_em_analytic_density import compare_results

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
 p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import yaml
 v.OUT=a.output.resolve();v.OUT.mkdir(parents=True,exist_ok=True);config=v.config
 def cfg(case):
  c=config(case);c['material_electron_charged_index_file']=str(a.candidate.resolve()/'charged/index.txt')
  c['material_photon_response_index_file']=str(a.candidate.resolve()/'photon/index.txt');return c
 for case,n,h in [('proton_broadbeam',10,1253150),('carbon_broadbeam',7,925987)]:
  for tag,seed in [('c1',1001904502),('c2',1001904602)]:
   directory=a.control.resolve()/'runs'/f'{case}_continuous_{tag}'
   old=yaml.safe_load((directory/'case.yaml').read_text());new=cfg(case);new['random_seed']=seed
   ignored={'material_electron_charged_index_file','material_photon_response_index_file'}
   differences={k:(old.get(k),new.get(k)) for k in old.keys()|new.keys() if k not in ignored and old.get(k)!=new.get(k)}
   if differences:raise RuntimeError(f'Reference physics differs: {differences}')
   rec=json.loads((directory/'execution.json').read_text());assert rec['runtime_pass'] and (rec['n'],rec['h'],rec['seed'])==(n,h,seed)
   v.save(v.OUT/f'control_{case}_{tag}.json',dict(directory=str(directory),record=rec,same_config_except_collision_encoding=True))
   v.config=cfg;v.run(case,'merged',tag,n,h,seed,(v.ROOT/'build/carbon_mc'))
  target=v.OUT;v.OUT=a.control.resolve()
  current,ref,mask,spacing=v.analyze(case,'continuous',['c1','c2'],report_dir=target)
  v.OUT=target;candidate,ev,_,_=v.analyze(case,'merged',['c1','c2'])
  compare_results(case,current,candidate,ref,ev,mask,spacing,v.OUT)
 # A complete-head-plan low-statistics pair checks interface behavior without
 # pretending to replace the existing0.1% absolute TOPAS qualification.
 for arm in ['reference','merged']:
  v.config=config if arm=='reference' else cfg
  v.run('proton_head_minibeam',arm,'screen',6,1983920,1051907901,v.ROOT/'build/carbon_mc')
 current,ref,mask,spacing=v.analyze('proton_head_minibeam','reference',['screen'])
 candidate,ev,_,_=v.analyze('proton_head_minibeam','merged',['screen'])
 compare_results('proton_head_minibeam',current,candidate,ref,ev,mask,spacing,v.OUT)
 v.config=config
if __name__=='__main__':main()
