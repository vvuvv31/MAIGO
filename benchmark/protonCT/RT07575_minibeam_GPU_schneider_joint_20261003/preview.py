"""Small partial full-plan preview; same history normalization as the final run."""
import json,re
from pathlib import Path
import numpy as np
from common import ROOT,read_mhd,write_mhd,packed_to_patient,save_json
from metrics import roi_indices,dose_metrics
from plots import make_plots
m=json.loads((ROOT/'manifest.json').read_text());old=Path(m['previous_campaign']);jobs=m['jobs'][:12]
log=(ROOT/'group000.log').read_text();assert len(re.findall(r'\[batch-wall\]',log))>=12
histories=sum(j['histories'] for j in jobs);x=np.zeros((417,35,505),float);y=x.copy()
for j in jobs:
 q=json.loads((ROOT/'out'/j['name']/'quality_report.json').read_text());assert q['accepted'] and q['queue_overflow_count']==0
 a,_=read_mhd(old/'out'/j['name']/'dose.mhd');b,_=read_mhd(ROOT/'out'/j['name']/'dose.mhd');x+=a;y+=b
scale=m['protons_4Gy']/histories;x=packed_to_patient(x)*scale;y=packed_to_patient(y)*scale
ref,h=read_mhd(ROOT/'inputs/physical_dose.mhd');d=ROOT/'preview_12batches';d.mkdir(exist_ok=False)
write_mhd(d/'dose_4Gy.mhd',y,h);write_mhd(d/'legacy_4Gy.mhd',x,h)
rois=roi_indices();r=dict(status='PARTIAL_PREVIEW_NOT_FINAL',histories=histories,fraction_of_plan=histories/m['protons_4Gy'],GTV=dict(TOPAS=dose_metrics(ref,rois)['GTV'],legacy_same_batches=dose_metrics(x,rois)['GTV'],joint=dose_metrics(y,rois)['GTV']),same_batch_relative_L1=float(abs(y-x).sum()/x.sum()),normalization='incident primary count only; no fitted dose scaling')
save_json(d/'summary.json',r);make_plots(d)
print(json.dumps(r,indent=2))
