import sys,json,struct,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path('/mnt/sdb/wuwei/MAIGO');sys.path.insert(0,str(repo/'tools'))
from evaluate_topas10x_gpu_gamma import pass_mask
run=Path('/mnt/sda/wuwei/final_stopping_20260911/RT07575');refdir=Path('/mnt/sda/wuwei/ct_previous_full20_20260909/RT07575')
out=repo/'evidence/step-31/rt07575-gamma-spatial-20260911'
m=json.loads((refdir/'manifest.json').read_text());state=json.loads((run/'execution.json').read_text());shape=tuple(m['topas_shape_zyx']);sp=np.array(m['spacing_zyx'])
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(run/'gpu_sum.raw')==state['aggregate_sha256'];assert sha(refdir/'topas_sum.raw')==m['reference_sum_sha256']
def mapped(p):return np.flip(np.fromfile(p,'<f4').reshape(m['gpu_shape_zyx']).transpose(1,2,0),axis=2)
g=mapped(run/'gpu_sum.raw');r=np.fromfile(refdir/'topas_sum.raw','<f4').reshape(shape)
mask=r>=.1*r.max();pts=np.argwhere(mask);rr=r[mask].astype(float);gg=g[mask].astype(float);err=100*(gg-rr)/rr
import yaml
cfg=yaml.safe_load((run/'shard_01/config.yaml').read_text())
with Path(cfg['ct_grid_file']).open('rb') as f:
 h=struct.unpack('<5I6f',f.read(44));nx,ny,nz=h[2:5];n=nx*ny*nz;rho=np.fromfile(f,'<f4',n).reshape(nz,ny,nx);mat=np.fromfile(f,'u1',n).reshape(nz,ny,nx)
rho=np.flip(rho.transpose(1,2,0),2);mat=np.flip(mat.transpose(1,2,0),2)
edge=np.zeros(shape,bool);air_edge=np.zeros(shape,bool)
for ax in range(3):
 a=[slice(None)]*3;b=list(a);a[ax]=slice(1,None);b[ax]=slice(None,-1);a=tuple(a);b=tuple(b);diff=mat[a]!=mat[b];ae=diff&((mat[a]==0)|(mat[b]==0));edge[a]|=diff;edge[b]|=diff;air_edge[a]|=ae;air_edge[b]|=ae
boundary=nd.distance_transform_edt(~edge,sampling=sp)[mask];airdist=nd.distance_transform_edt(~air_edge,sampling=sp)[mask]
grads=np.gradient(r,*sp);grad=np.sqrt(sum(x*x for x in grads))[mask]/rr*100
report={'case':'RT07575','evaluated':len(pts),'gpu_sha256':state['aggregate_sha256'],'reference_sha256':m['reference_sum_sha256'],'method':'>=10% reference maximum; existing 3D dose only; packed_xneg restored; gamma fail rescue 0.5 to 0.25mm lattice; local11 further to 0.0625mm, local33 to 0.125mm; no dose fitting applied','boundary_distance':'Euclidean distance to centres of face-adjacent unequal-section voxels; approximate interface proximity, not an anatomical surface distance','gamma':{}}
fails={}
for dd,dta in [(3,3),(2,2),(1,1),(3,0)]:
 for local in [False,True]:
  key=f"{'local' if local else 'global'}_{dd}{dta}"
  coarse=pass_mask(g,r,pts,sp,dd,dta,local,.5);fine=coarse.copy()
  if dta:
   ind=np.flatnonzero(~fine);fine[ind]=pass_mask(g,r,pts[ind],sp,dd,dta,local,.25)
  fails[key]=~fine
  report['gamma'][key]={'coarse_pass_pct':float(100*coarse.mean()),'refined_pass_pct':float(100*fine.mean()),'rescued':int((fine&~coarse).sum()),'failed':int((~fine).sum())}
  print(key,report['gamma'][key],flush=True)
# Additional lattice refinement, reusing only previous failures.
for key,dd,dta,steps in [('local_11',1,1,[.125,.0625]),('local_33',3,3,[.125])]:
    for step in steps:
        ix=np.flatnonzero(fails[key]);hit=pass_mask(g,r,pts[ix],sp,dd,dta,True,step)
        fails[key][ix[hit]]=False
    report['gamma'][key]['final_search_step_mm']=steps[-1]
    report['gamma'][key]['refined_pass_pct']=float(100*(~fails[key]).mean())
    report['gamma'][key]['failed']=int(fails[key].sum())
    report['gamma'][key]['rescued']=int(round(len(pts)*(1-report['gamma'][key]['coarse_pass_pct']/100)))-int(fails[key].sum())
np.savez_compressed(out/'failure_masks.npz',points_zyx=pts,**fails)
def stats(sel):
 n=int(sel.sum())
 if not n:return {'voxels':0}
 result={'voxels':n,'mask_share_pct':100*n/len(pts),'mean_error_pct':float(err[sel].mean()),'rms_error_pct':float(np.sqrt(np.mean(err[sel]**2)))}
 for key in ['local_33','local_22','local_11','local_30']:
  bad=fails[key]&sel;v=int(bad.sum());result[key]={'failed':v,'fail_rate_pct':100*v/n,'share_all_failures_pct':100*v/int(fails[key].sum()),'positive_error_share_pct':float(100*(err[bad]>0).mean()) if v else None,'mean_error_failed_pct':float(err[bad].mean()) if v else None}
 return result
report['dose_bands']={f'{lo}-{hi}%':stats((rr>=r.max()*lo/100)&(rr<r.max()*hi/100)) for lo,hi in [(10,20),(20,50),(50,80),(80,101)]}
report['sections']={str(int(s)):stats(mat[mask]==s) for s in np.unique(mat[mask])}
report['interface_bands']={f'{lo}-{hi}mm':stats((boundary>=lo)&(boundary<hi)) for lo,hi in [(0,1),(1,3),(3,10),(10,1000)]}
report['air_interface_bands']={f'{lo}-{hi}mm':stats((airdist>=lo)&(airdist<hi)) for lo,hi in [(0,1),(1,3),(3,10),(10,1000)]}
report['gradient_bands']={f'{lo}-{hi}%local/mm':stats((grad>=lo)&(grad<hi)) for lo,hi in [(0,1),(1,3),(3,10),(10,10000)]}
# Independent replication variance at fixed source allocation, approximately equal shard histories.
a=[]
for x in m['replicas']:a.append(np.fromfile(Path(x['path'])/'OSMK_Dtotal_full_plan.bin','<f8').reshape(shape)[mask])
a=np.array(a);assert np.allclose(a.sum(0),rr,rtol=1e-6,atol=1e-10)
vt=len(a)*np.var(a,axis=0,ddof=1);a=[]
for x in state['completed']:a.append(mapped(Path(x['directory'])/'dose.raw')[mask].astype(float)/x['histories'])
a=np.array(a);vg=np.var(a,axis=0,ddof=1)/len(a)*m['histories']**2;noise=100*np.sqrt(vt+vg)/rr
report['noise']={'caveat':'5 TOPAS repetitions, 20 nearly equal GPU source allocations; finite-replicate estimates, not per-voxel certainty; event-bank systematic errors excluded','expected_combined_rms_local_pct':float(np.sqrt(np.mean(noise**2))),'observed_rms_local_pct':float(np.sqrt(np.mean(err**2))),'median_combined_sigma_pct':float(np.median(noise)),'groups':{}}
for key in ['local_33','local_11']:
 sel=fails[key];report['noise']['groups'][key]={'observed_rms_pct':float(np.sqrt(np.mean(err[sel]**2))),'expected_noise_rms_pct':float(np.sqrt(np.mean(noise[sel]**2))),'beyond3sigma_pct':float(100*(np.abs(err[sel])>3*noise[sel]).mean())}
# Cluster locations in scorer-centred coordinates; do not label anatomy without contours.
report['clusters']={}
for key in ['local_33','local_11']:
 volume=np.zeros(shape,bool);volume[tuple(pts[fails[key]].T)]=True;labels,num=nd.label(volume);counts=np.bincount(labels.ravel());counts[0]=0;ids=np.argsort(counts)[-5:][::-1];rows=[]
 for ident in ids:
  if not counts[ident]:continue
  q=np.argwhere(labels==ident);sel=labels[mask]==ident;rows.append({'voxels':int(counts[ident]),'bbox_zyx_indices':[q.min(0).tolist(),q.max(0).tolist()],'centroid_scorer_relative_zyx_mm':((q.mean(0)-(np.array(shape)-1)/2)*sp).tolist(),'mean_error_pct':float(err[sel].mean()),'mean_ref_pct_max':float(100*rr[sel].mean()/r.max()),'sections':{str(int(k)):int(v) for k,v in zip(*np.unique(mat[mask][sel],return_counts=True))}})
 report['clusters'][key]=rows
# Small rigid shift+scale diagnostic, not a fitted reported gamma.
X=np.column_stack([np.ones(len(rr))]+[v[mask]/rr for v in grads]);coef=np.linalg.lstsq(X,(gg-rr)/rr,rcond=None)[0];pred=X@coef
report['linear_alignment_diagnostic']={'scale_offset_pct':float(100*coef[0]),'equivalent_gradient_coefficients_zyx_mm':coef[1:].tolist(),'variance_fraction_explained':float(1-np.var((gg-rr)/rr-pred)/np.var((gg-rr)/rr)),'caveat':'first-order regression only; coefficients are not a validated registration or applied correction'}
# Plots on scorer-centred axes.
fig,axs=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
for row,key in enumerate(['local_33','local_11']):
 vol=np.zeros(shape,bool);vol[tuple(pts[fails[key]].T)]=True;z=int(np.argmax(vol.sum((1,2))));extent=[-(shape[2]*sp[2])/2,shape[2]*sp[2]/2,-shape[1]*sp[1]/2,shape[1]*sp[1]/2]
 im=axs[row,0].imshow(r[z]/r.max()*100,origin='lower',extent=extent,vmin=0,vmax=100,cmap='viridis');fig.colorbar(im,ax=axs[row,0],label='TOPAS dose / max (%)')
 e=np.full(shape[1:],np.nan);sel=mask[z];e[sel]=100*(g[z][sel]-r[z][sel])/r[z][sel];im=axs[row,1].imshow(e,origin='lower',extent=extent,cmap='coolwarm',vmin=-5,vmax=5);fig.colorbar(im,ax=axs[row,1],label='(GPU-TOPAS)/TOPAS (%)')
 axs[row,2].imshow(rho[z],origin='lower',extent=extent,cmap='gray',vmin=0,vmax=2);yy,xx=np.where(vol[z]);axs[row,2].scatter((xx+.5-shape[2]/2)*sp[2],(yy+.5-shape[1]/2)*sp[1],s=2,c='red');axs[row,2].set_title(f'{key}: refined failures (red)')
 for ax in axs[row]:ax.set_xlabel('Scorer-centred X (mm)');ax.set_ylabel('Scorer-centred Y (mm)')
 axs[row,0].set_title(f'{key}: Z index {z}, relative Z={(z-(shape[0]-1)/2)*sp[0]:.1f} mm')
fig.savefig(out/'failure_slices.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
for ax in range(3):
 axes=tuple(i for i in range(3) if i!=ax);x=(np.arange(shape[ax])-(shape[ax]-1)/2)*sp[ax];rs=r.sum(axes);gs=g.sum(axes);axs[0,ax].plot(x,rs,label='TOPAS');axs[0,ax].plot(x,gs,label='GPU');axs[0,ax].legend();axs[0,ax].set_title('ZYX'[ax]+' dose summed over other axes')
 for key in ['local_33','local_11']:
  bad=np.bincount(pts[fails[key],ax],minlength=shape[ax]);den=np.bincount(pts[:,ax],minlength=shape[ax]);axs[1,ax].plot(x,np.divide(100*bad,den,out=np.full(len(den),np.nan),where=den>0),label=key)
 axs[1,ax].legend();axs[1,ax].set_ylabel('Failure rate in mask (%)');axs[1,ax].set_xlabel('Scorer-centred coordinate (mm)')
fig.savefig(out/'profiles.png',dpi=160);plt.close(fig)
(out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('DONE',out,flush=True)
