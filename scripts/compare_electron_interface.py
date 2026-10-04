#!/usr/bin/env python3
"""History-aware native vs independent electron+photon interface comparison."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmark/electron_response_model'))
from macro_model import DTYPE
from prototype import sha,save
RESULT=np.dtype({'names':['status','steps','crossings','returns','photons','position'],'formats':['<u4']*5+[('<f8',3)],'offsets':[0,4,8,12,16,24],'itemsize':48})
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True);p.add_argument('--gpu',type=Path,required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--native-override',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 cases=json.loads(a.cases.read_text());gpu=np.memmap(a.gpu,RESULT,mode='r');assert len(gpu)==len(cases)*a.samples
 results=[]
 for i,c in enumerate(cases):
  source=Path(c['path']);override=a.native_override/source.name if a.native_override else None
  if override and (override/'report.json').exists():source=override
  report=json.loads((source/'report.json').read_text());audit=report['audit'];assert sha(source/'steps.phsp')==audit['raw_sha256']
  r=np.memmap(source/'steps.phsp',DTYPE,mode='r');n=report['histories'];energy=audit['actual_source_energy_MeV'];assert energy==c['energy']
  left_ids=np.unique(r['pre_material'][(r['pre'][:,2]<-1e-7)&(r['pre'][:,2]>-19.999)&(np.max(abs(r['pre'][:,:2]),axis=1)<19.999)]);assert len(left_ids)==1
  left=np.bincount(r['event'],weights=np.where(r['pre_material']==left_ids[0],r['dep'],0),minlength=n)/energy
  g=gpu[i*a.samples:(i+1)*a.samples];assert np.isin(g['status'],[0,1,2]).all()
  right_ids=np.unique(r['pre_material'][(r['pre'][:,2]>1e-7)&(r['pre'][:,2]<19.999)&(np.max(abs(r['pre'][:,:2]),axis=1)<19.999)]);assert len(right_ids)==1
  right=np.bincount(r['event'],weights=np.where(r['pre_material']==right_ids[0],r['dep'],0),minlength=n)/energy
  gr=float((g['status']==1).mean());nr=float(right.mean())
  candidate=(g['status']==0).astype(float);mean=float(left.mean());gm=float(candidate.mean());se=float(np.sqrt(left.var(ddof=1)/n+candidate.var(ddof=1)/a.samples))
  results.append(dict(case=source.name,native_histories=n,gpu_packets=a.samples,native_left_fraction=mean,gpu_left_fraction=gm,delta_percentage_points=100*(gm-mean),relative_left_percent=100*(gm/mean-1),relative_right_percent=100*(gr/nr-1),native_right_fraction=nr,gpu_right_fraction=gr,difference_standard_error_percentage_points=100*se,z=abs(gm-mean)/se,invalid=0,return_crossings=int(g['returns'].sum()),photon_calls=int(g['photons'].sum()),source_raw_sha256=audit['raw_sha256'],scope='Left-material energy fraction, native per-history variance and GPU energy-packet Bernoulli variance. Not a local-dose or peak/valley acceptance.'))
 save(a.out,dict(status='RESEARCH_INTERFACE_COMPARISON_NOT_CT_ACCEPTANCE',gpu_sha256=sha(a.gpu),cases_sha256=sha(a.cases),results=results));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
