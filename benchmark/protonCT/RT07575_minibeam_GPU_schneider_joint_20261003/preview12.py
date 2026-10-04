"""Read completed shards during the full run; paired preview, no new transport."""
import json,re,time
from pathlib import Path
import numpy as np
from common import ROOT,read_mhd,packed_to_patient,save_json,write_mhd
from metrics import roi_indices,dose_metrics

def main():
 m=json.loads((ROOT/'manifest.json').read_text());jobs=m['jobs'][:12];old=Path(m['previous_campaign']);log=ROOT/'group000.log'
 while not log.exists() or log.read_text().count('[plan-shard-done]')<12:
  if (ROOT/'completion.json').exists():raise RuntimeError('campaign ended before preview')
  time.sleep(5)
 shape=(417,35,505);x=np.zeros(shape);y=np.zeros(shape);q=[]
 for j in jobs:
  a,_=read_mhd(old/'out'/j['name']/'dose.mhd');b,_=read_mhd(ROOT/'out'/j['name']/'dose.mhd');x+=a;y+=b
  quality=json.loads((ROOT/'out'/j['name']/'quality_report.json').read_text());assert quality['accepted'] and quality['queue_overflow_count']==0;q.append(quality)
 histories=sum(j['histories'] for j in jobs);scale=m['protons_4Gy']/histories
 xp=packed_to_patient(x)*scale;yp=packed_to_patient(y)*scale;ref,fields=read_mhd(ROOT/'inputs/physical_dose.mhd')
 rois=roi_indices();indices=np.union1d(rois['BODY'],rois['GTV']);a=xp.transpose(0,2,1).reshape(-1)[indices];b=yp.transpose(0,2,1).reshape(-1)[indices];r=ref.transpose(0,2,1).reshape(-1)[indices];mask=r>=ref.max()*.1
 work={};packets=0;parts=log.read_text().split('[plan-shard] ')[1:13]
 for t in parts:
  w=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',next(l for l in t.splitlines() if l.startswith('[electron-joint-pilot-work]'))))
  for k,v in w.items():work[k]=work.get(k,0)+v
  packets+=int(re.search(r'\[electron-queue-time\].*?tasks=(\d+)',t)[1])
 body=np.zeros((35,417,505),bool);body.reshape(-1)[indices]=True;body=np.flip(body.transpose(1,0,2),axis=0)
 xx=-169.1015+(np.arange(505)+.5)*.5;yy=-33.6383+(np.arange(35)+.5)*2.;zz=(np.arange(417)+.5)*.5;phase=(xx+1.8)%3.6-1.8
 domain=body&(abs(xx)[None,None,:]<=14.4)&(abs(yy)[None,:,None]<=10);pv={}
 for lo,hi in [(20,40),(40,80),(80,120),(120,160)]:
  for name,sel in [('peak',abs(phase)<=.25),('valley',abs(abs(phase)-1.8)<=.25)]:
   mask2=domain&((zz>=lo)&(zz<hi))[:,None,None]&sel[None,None,:]
   if x[mask2].sum()>0:pv[f'{lo}-{hi}mm_{name}']=100*(y[mask2].sum()/x[mask2].sum()-1)
 result=dict(status='PAIRED_12_BATCH_PREVIEW_NOT_FORMAL_ACCEPTANCE',histories=histories,work=work,packets=packets,started_fraction=work['started']/packets,
 quality_batches=len(q),max_energy_residual=max(v['relative_energy_residual'] for v in q),
 whole_grid_relative_L1=float(abs(x-y).sum()/x.sum()),body_dose_change=dict(relative_L1=float(abs(a-b).sum()/a.sum()),max_absolute_Gy=float(abs(a-b).max())),
 local_3pct_0mm_vs_TOPAS=dict(legacy=float(100*np.mean(abs(a[mask]-r[mask])<=.03*r[mask])),new=float(100*np.mean(abs(b[mask]-r[mask])<=.03*r[mask]))),
 GTV=dict(legacy=dose_metrics(xp,rois)['GTV'],new=dose_metrics(yp,rois)['GTV']),peak_valley_change_percent=pv)
 out=ROOT/'preview_12batches';out.mkdir(exist_ok=True);write_mhd(out/'dose_4Gy.mhd',yp,fields);save_json(out/'paired_result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
