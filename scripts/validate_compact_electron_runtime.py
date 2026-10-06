#!/usr/bin/env python3
"""Matched FP32 screens for direct collision storage and local photon aggregation."""
from pathlib import Path
import argparse,json,re
import numpy as np
import validate_em_density_factorization as v

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
 p.add_argument('--binary',type=Path,default=Path('build/carbon_mc'));a=p.parse_args()
 v.OUT=a.output.resolve();v.OUT.mkdir(parents=True,exist_ok=True);config=v.config;records=[]
 for case,h in [('proton_broadbeam',1253150),('carbon_broadbeam',925987),('proton_minibeam',500000),('carbon_minibeam',250000)]:
  order=['reference','fp32','merged'] if case.startswith('proton') else ['merged','fp32','reference']
  results={}
  for arm in order:
   def cfg(name,arm=arm):
    c=config(name)
    if arm!='reference':
     c['material_electron_charged_index_file']=str(a.candidate_root.resolve()/arm/'charged/index.txt')
     c['material_photon_response_index_file']=str(a.candidate_root.resolve()/arm/'photon/index.txt')
    return c
   v.config=cfg
   directory,record=v.run(case,arm,'screen',3,h,1051907701,(a.baseline if arm=='reference' else a.binary).resolve())
   record['arm']=arm;log=(directory/'gpu.log').read_text()
   record['setup_seconds']=[float(x) for x in re.findall(r'\[runtime-stage\] transport_setup_including_upload seconds=([\d.eE+-]+)',log)]
   record['kernel_seconds']=[(float(x),float(y)) for x,y in re.findall(r'Kernel time: primary=([\d.eE+-]+) s electron=([\d.eE+-]+) s',log)]
   record['bank_memory']=re.findall(r'\[independent-fp32-bank\].*',log)[:1]
   results[arm]=(directory,record)
  reference,_=v.summed(results['reference'][0])
  for arm in ['fp32','merged']:
   candidate,_=v.summed(results[arm][0]);difference=candidate-reference
   record=results[arm][1]
   record['dose_relative_L1']=float(abs(difference).sum()/reference.sum())
   record['dose_sum_change_percent']=float(100*difference.sum()/reference.sum())
   record['resident_ratio_to_reference']=record['resident_primary_s']/results['reference'][1]['resident_primary_s']
   record['transport_audits_exact']=True
   for prefix in ['[unified-em-audit]','[independent-charged-work]','[independent-photon-work]','[electron-joint-pilot-work]']:
    def lines(directory):
     return [x for x in (directory/'gpu.log').read_text().splitlines() if x.startswith(prefix)]
    equal=lines(results[arm][0])==lines(results['reference'][0])
    record.setdefault('audit_equality',{})[prefix]=equal
    record['transport_audits_exact'] &= equal
  records.append({arm:record for arm,(_,record) in results.items()})
  v.save(v.OUT/'summary.json',records);print(json.dumps(records[-1]),flush=True)
 v.config=config
if __name__=='__main__':main()
