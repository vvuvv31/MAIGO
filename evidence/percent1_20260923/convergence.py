from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/percent1_20260923');j=json.loads((p/'comparison.json').read_text());assert j['status']=='COMPLETE'
fig,ax=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
for width,a in zip([.25,10],ax):
 for row in [r for r in j['rows'] if r['window_width_mm']==width]:
  v=row['valley'];g=np.array(v['gpu']['values']);ref=v['topas']['mean'];n=np.arange(1,len(g)+1);curve=100*(np.cumsum(g)/n/ref-1)
  a.plot(n[2:],curve[2:],'.-',label=f"{row['depth_mm']} mm")
 a.axhspan(-1,1,color='#dcfce7');a.axhline(0,lw=.6,c='#888');a.set(xlabel='Cumulative GPU source histories (millions)',ylabel='GPU / pooled TOPAS - 1 (%)',title=f'Fixed central valleys, {width:g} mm depth window',xticks=[3,6,9,12,15]);a.legend(ncol=2,fontsize=8)
fig.suptitle('Independent GPU batches: cumulative estimates at fixed ROI\nFixed 16M TOPAS denominator; curves illustrate fluctuation, not confidence bounds')
fig.savefig(p/'convergence.png',dpi=170);plt.close(fig)
