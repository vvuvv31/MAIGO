from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
stages=json.loads((r/'disjoint_stages.json').read_text())
cycles={int(k):v[0] for k,v in json.loads((r/'phase_cycles.json').read_text()).items()}
fig,axs=plt.subplots(1,3,figsize=(19,7),constrained_layout=True)
short=['Input/source','Setup incl. upload','Primary kernels','Secondary kernels','Other loop work','Readback/finalize','Quality/output','Other process']
axs[0].barh(short,list(stages.values()),color='#3e6a98');axs[0].invert_yaxis();axs[0].set_xlabel('Seconds: unprofiled full process');axs[0].set_title('RT07575: 1M histories / RTX 2080 Ti')
for a,slots,labels,title in [
 (axs[1],[0,6,7,1,2,3,4,5],['CT/material + initial state','EM table/range/rate preparation','Geometry/nuclear rates/pre-loss','Unified EM loss','Scoring branches','MCS','Advance/face handling','Nuclear resolution/bookkeeping'],'Primary: sampled lane cycles'),
 (axs[2],[8,14,15,9,10,11,12,13],['CT/material/stopping + state','EM table/range/rate preparation','Geometry/nuclear rates/pre-loss','Unified EM loss','Geometry/scoring after EM','MCS','Inelastic replay/scoring','Elastic/bookkeeping'],'Secondary: sampled lane cycles')]:
 total=sum(cycles.get(i,0) for i in slots);v=[100*cycles.get(i,0)/total for i in slots]
 a.barh(labels,v,color='#50876b');a.invert_yaxis();a.set_xlabel('Sampled cycle share (%) — not wall-time share');a.set_title(title)
 for j,x in enumerate(v):a.text(x+.35,j,f'{x:.1f}%',va='center',fontsize=8)
 a.set_xlim(0,max(v)*1.2)
fig.savefig(r/'runtime_breakdown.png',dpi=150)
