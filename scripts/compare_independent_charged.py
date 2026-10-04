#!/usr/bin/env python3
"""RT07575 independent charged result against its matched legacy-charged shard."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'out/electron_standalone_20261003'
sys.path.insert(0,str(ROOT/'benchmark/protonCT/RT07575_minibeam_GPU_schneider_joint_20261003'))
from common import read_mhd,sha,save_json
from metrics import roi_indices
from plot_isocenter_profiles import interpolate
from compare_minibeam import AUDITS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=BASE/'rt07575_independent_smoke');p.add_argument('--shards',type=int,default=1);a=p.parse_args();out=a.out.resolve();assert 1<=a.shards<=10
 references=[BASE/f'rt07575_photon_confirm/out/fresh{i:04d}' for i in range(a.shards)]
 candidates=[out/'out'/('independent' if a.shards==1 else f'fresh{i:04d}') for i in range(a.shards)]
 xd=[read_mhd(d/'dose.mhd') for d in references];yd=[read_mhd(d/'dose.mhd') for d in candidates]
 assert all(f==xd[0][1] and d.shape==(417,35,505) for d,f in xd+yd)
 x=sum(d.astype(float) for d,f in xd);y=sum(d.astype(float) for d,f in yd)
 quality=[json.loads((d/'quality_report.json').read_text()) for d in candidates]
 assert all(q['accepted'] and not q['failures'] and q['queue_overflow_count']==0 for q in quality)
 logs=(out/'gpu.log').read_text();oldlogs=(BASE/'rt07575_photon_confirm/gpu.log').read_text()
 chunks=logs.split('[plan-shard] ')[1:];oldchunks=oldlogs.split('[plan-shard] ')[1:]
 audits=lambda s:[l for l in s.splitlines() if l.startswith(AUDITS)]
 selected=[]
 for i in range(a.shards):
  old=next(c for c in oldchunks if f'fresh{i:04d}.yaml' in c.splitlines()[0])
  new=next(c for c in chunks if ('independent.yaml' if a.shards==1 else f'fresh{i:04d}.yaml') in c.splitlines()[0])
  assert audits(old)==audits(new);selected.append(new)
 assert all('legacy_loaded=0 legacy_fallback=0' in c for c in selected)
 forbidden=['RT07575_minibeam_GPU/response_extension','RT07575_minibeam_GPU_packed_20261002/electron_packed','material_prepared']
 trace_root=out if (out/'files.trace').exists() else BASE/'rt07575_independent_smoke'
 assert sha(out/'carbon_mc')==sha(trace_root/'carbon_mc')
 trace=(trace_root/'files.trace').read_text();bad=[l for l in trace.splitlines() if any(s in l for s in forbidden)];assert not bad,bad
 regression=None
 if a.shards>1:
  regression_dose=read_mhd(out/'out/control0000/dose.mhd')[0].astype(float)
  regression=float(abs(regression_dose-xd[0][0]).sum()/xd[0][0].sum(dtype=float));assert regression<1e-5,regression
 work={}
 for label in ['independent-charged-work','independent-photon-work','electron-joint-pilot-work']:
  work[label]={}
  for chunk in selected:
   line=next(l for l in chunk.splitlines() if l.startswith('['+label+']'));w={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)};assert w.get('invalid',0)==0 and w.get('coverage_missing',0)==0
   for k,v in w.items():work[label][k]=work[label].get(k,0)+v
 rois=roi_indices();body=np.zeros((35,417,505),bool);body.reshape(-1)[np.union1d(rois['BODY'],rois['GTV'])]=True;body=np.flip(body.transpose(1,0,2),axis=0)
 xx=-169.1015+(np.arange(505)+.5)*.5;yy=-33.6383+(np.arange(35)+.5)*2;zz=(np.arange(417)+.5)*.5;phase=(xx+1.8)%3.6-1.8
 domain=body&(abs(xx)[None,None,:]<=14.4)&(abs(yy)[None,:,None]<=10);pv={};pv_se={}
 for lo,hi in [(20,40),(40,80),(80,120),(120,160)]:
  for label,sel in [('peak',abs(phase)<=.25),('valley',abs(abs(phase)-1.8)<=.25)]:
   mask=domain&((zz>=lo)&(zz<hi))[:,None,None]&sel[None,None,:];pv[f'{lo}-{hi}mm_{label}']=float(100*(y[mask].sum()/x[mask].sum()-1))
   paired=[100*(new[0][mask].sum(dtype=float)/old[0][mask].sum(dtype=float)-1) for old,new in zip(xd,yd)]
   pv_se[f'{lo}-{hi}mm_{label}']=float(np.std(paired,ddof=1)/np.sqrt(a.shards)) if a.shards>1 else None
 local={}
 for threshold in [.01,.1]:
  mask=body&(x>=threshold*x[body].max());r=abs(y[mask]-x[mask])/x[mask]
  local[str(threshold)]=dict(voxels=int(mask.sum()),pass_1percent=float(np.mean(r<=.01)),pass_3percent=float(np.mean(r<=.03)),maximum_percent=float(100*r.max()),percentiles_percent=(100*np.quantile(r,[.5,.95,.99])).tolist())
 report=dict(status='NO_LEGACY_RT07575_RESEARCH_SCREEN',histories=1983920*a.shards,same_binary_control_relative_L1=regression,trace_root=str(trace_root),legacy_files_opened=bad,legacy_forbidden_markers=forbidden,primary_audits_equal=True,quality=quality,work=work,BODY_relative_L1=float(abs(y[body]-x[body]).sum()/x[body].sum()),whole_grid_relative_L1=float(abs(y-x).sum()/x.sum()),peak_valley_change_percent=pv,peak_valley_sampling_SE_percent=pv_se,local_change=local,reference_raw_sha256=[sha(d/'dose.raw') for d in references],candidate_raw_sha256=[sha(d/'dose.raw') for d in candidates],scope='Matched full-source RT07575 shards, no legacy electron runtime; peak/valley/local differences include electron MC and finite-bank uncertainty. Not a TOPAS gamma or production acceptance. Traced timing includes strace overhead.')
 save_json(out/'report.json',report)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 iso=117.0136;u=xx[(xx>=-35)&(xx<=35)];d=zz[(zz>=25)&(zz<=170)]
 planes=[interpolate(a,d[:,None],u[None,:],0) for a in [x,y]];delta=planes[1]-planes[0];scale=np.max(planes[0]);fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained')
 extent=[u[0]-.25,u[-1]+.25,d[0]-.25-iso,d[-1]+.25-iso]
 for ax,z,label in zip(axs,planes+[100*delta/scale],['Legacy charged + new joint/photon','Independent charged + joint/photon','Difference (% reference plane maximum)']):
  diff=ax is axs[-1];lim=np.max(abs(z)) if diff else scale;im=ax.imshow(z,origin='lower',extent=extent,cmap='RdBu_r' if diff else 'inferno',vmin=-lim if diff else 0,vmax=lim,aspect='equal');ax.set(title=label,xlabel='Lateral u (mm)',ylabel='Depth relative to isocenter (mm)');ax.plot(0,0,'+',color='cyan');fig.colorbar(im,ax=ax)
 fig.suptitle(f'RT07575: {1983920*a.shards:,} primary, v = 0 plane; same incident exposure, no fitted normalization');fig.savefig(out/'isocenter_plane.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained');profiles=[]
 for ax,depth in zip(axs.flat,[40,80,100,iso,130,150]):
  curves=[interpolate(a,depth,u,0) for a in [x,y]]
  for curve,label in zip(curves,['Legacy charged','Independent charged']):ax.plot(u,curve,label=label,lw=1)
  ax.set(title=f'CT depth {depth:.2f} mm',xlabel='Lateral u (mm)',ylabel='Dose (Gy, simulated exposure)');ax.grid(alpha=.2);profiles.append(dict(depth_mm=depth,u_mm=u.tolist(),reference=curves[0].tolist(),independent=curves[1].tolist()))
 axs.flat[0].legend();fig.savefig(out/'lateral_profiles.png',dpi=160);plt.close(fig);save_json(out/'lateral_profiles.json',profiles)
 print(json.dumps({k:report[k] for k in ['status','BODY_relative_L1','peak_valley_change_percent','local_change']},indent=2))
if __name__=='__main__':main()
