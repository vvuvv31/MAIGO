from pathlib import Path
import sys,json,struct,hashlib
import numpy as np
import pydicom
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path as PolygonPath

def polygon(rows, cols, shape):
    r0=max(0,int(np.floor(rows.min())));r1=min(shape[0]-1,int(np.ceil(rows.max())))
    c0=max(0,int(np.floor(cols.min())));c1=min(shape[1]-1,int(np.ceil(cols.max())))
    rr,cc=np.mgrid[r0:r1+1,c0:c1+1];points=np.column_stack([cc.ravel(),rr.ravel()])
    vertices=np.column_stack([cols,rows]);vertices=np.vstack([vertices,vertices[0]])
    path=PolygonPath(vertices,closed=True)
    inside=path.contains_points(points,radius=1e-7)|path.contains_points(points,radius=-1e-7)
    return rr.ravel()[inside],cc.ravel()[inside]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path('/mnt/sdb/wuwei/MAIGO');sys.path.insert(0,str(repo/'tools'))
from evaluate_topas10x_gpu_gamma import pass_mask
base=repo/'evidence/step-31/rt07575-gamma-spatial-20260911';out=base/'body';out.mkdir(exist_ok=True)
dicom=repo/'benchmark/topas10x/RT07575_pbs_s1/dicom';rsfile=dicom/'RS.RT07575.dcm';rs=pydicom.dcmread(rsfile)
roi=next(x for x in rs.StructureSetROISequence if x.ROIName.strip().upper()=='BODY');rc=next(x for x in rs.ROIContourSequence if x.ReferencedROINumber==roi.ROINumber)
ctfiles=sorted(dicom.glob('IMG*.dcm'),key=lambda f:float(pydicom.dcmread(f,stop_before_pixels=True).ImagePositionPatient[2]));ct=[pydicom.dcmread(f) for f in ctfiles];first=ct[0];z=np.array([float(d.ImagePositionPatient[2]) for d in ct]);origin=np.array(first.ImagePositionPatient,float);dy,dx=map(float,first.PixelSpacing);ny,nx=int(first.Rows),int(first.Columns)
assert str(roi.ReferencedFrameOfReferenceUID)==str(first.FrameOfReferenceUID)
assert np.allclose(first.ImageOrientationPatient,[1,0,0,0,1,0]);assert np.allclose(np.diff(z),2)
assert all(np.allclose(d.ImageOrientationPatient,first.ImageOrientationPatient) and np.allclose(d.ImagePositionPatient[:2],origin[:2]) and np.allclose(d.PixelSpacing,[dy,dx]) and d.Rows==ny and d.Columns==nx for d in ct)
planes={}
for c in rc.ContourSequence:
 assert c.ContourGeometricType=='CLOSED_PLANAR'
 v=np.array(c.ContourData,float).reshape(-1,3);assert np.ptp(v[:,2])<1e-4
 zz=float(v[0,2]);assert zz not in planes,'Multiple loops need explicit topology handling'
 a=np.zeros((ny,nx),bool);rr,cc=polygon((v[:,1]-origin[1])/dy,(v[:,0]-origin[0])/dx,shape=a.shape);a[rr,cc]=True;planes[zz]=a
cz=np.array(sorted(planes));assert z.min()>=cz.min()-1e-4 and z.max()<=cz.max()+1e-4
sdf=np.array([distance_transform_edt(planes[q],sampling=[dy,dx])-distance_transform_edt(~planes[q],sampling=[dy,dx]) for q in cz])
body=[];nearest=[]
for zz in z:
 hi=int(np.searchsorted(cz,zz));
 if hi==0:b=sdf[0]>0
 elif hi==len(cz):b=sdf[-1]>0
 else:
  lo=hi-1;w=(zz-cz[lo])/(cz[hi]-cz[lo]);b=((1-w)*sdf[lo]+w*sdf[hi])>0
 body.append(b);nearest.append(planes[cz[np.argmin(abs(cz-zz))]])
body=np.array(body);nearest=np.array(nearest);np.save(out/'body_mask_zyx.npy',body)
refdir=Path('/mnt/sda/wuwei/ct_previous_full20_20260909/RT07575');run=Path('/mnt/sda/wuwei/final_stopping_20260911/RT07575');m=json.loads((refdir/'manifest.json').read_text());shape=tuple(m['topas_shape_zyx']);assert body.shape==shape
r=np.fromfile(refdir/'topas_sum.raw','<f4').reshape(shape);g=np.flip(np.fromfile(run/'gpu_sum.raw','<f4').reshape(m['gpu_shape_zyx']).transpose(1,2,0),2)
# Validate density/material alignment independently against original DICOM voxels.
with open('/mnt/sda/wuwei/topas10x_threecase_20260905_r2/rt07575_packed.bin','rb') as f:
 h=struct.unpack('<5I6f',f.read(44));n=h[2]*h[3]*h[4];rho=np.fromfile(f,'<f4',n).reshape(h[4],h[3],h[2]);mat=np.fromfile(f,'u1',n).reshape(rho.shape)
rho=np.flip(rho.transpose(1,2,0),2);mat=np.flip(mat.transpose(1,2,0),2)
hu=np.array([d.pixel_array.astype(float)*float(d.RescaleSlope)+float(d.RescaleIntercept) for d in ct]);edges=np.array([-1000,-950,-120,-83,-53,-23,7,18,80,120,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2995,2996]);expected=np.clip(np.searchsorted(edges,hu,side='right')-1,0,24)
match=float((mat==expected).mean());assert match>.999,'CT material alignment mismatch'
old=r>=.1*r.max();mask=old&body;pts=np.argwhere(mask);rr=r[mask].astype(float);err=100*(g[mask]-rr)/rr
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'case':'RT07575','roi_name':str(roi.ROIName),'roi_number':int(roi.ROINumber),'rtstruct_sha256':sha(rsfile),'ct_file_sha256':{p.name:sha(p) for p in ctfiles},'body_voxels':int(body.sum()),'evaluated_voxels':len(pts),'original_evaluated_voxels':int(old.sum()),'excluded_outside_body':int((old&~body).sum()),'contour_planes':len(cz),'contour_z_mm':cz.tolist(),'ct_z_mm':z.tolist(),'mask_method':'Rasterize CLOSED_PLANAR BODY at DICOM voxel centres; interpolate signed in-plane distance fields between 3mm contour planes to 2mm CT slices; no HU threshold, no z extrapolation','gamma_scope':'BODY limits reference query voxels only. Search original full GPU dose field, without zeroing/cropping outside BODY. Threshold and global normalization retain original TOPAS full-volume maximum.','ct_material_alignment_fraction':match,'nearest_contour_mask_disagreement_in_evaluation':int(((nearest!=body)&old).sum()),'gamma':{},'same_voxel_mean_local_error_pct':float(err.mean()),'same_voxel_rms_local_error_pct':float(np.sqrt(np.mean(err**2)))}
fail={}
for dd,dta in [(3,3),(2,2),(1,1),(3,0)]:
 for local in [False,True]:
  key=f"{'local' if local else 'global'}_{dd}{dta}";passed=pass_mask(g,r,pts,np.array(m['spacing_zyx']),dd,dta,local,.5);stages=[{'search_step_mm':.5,'pass_pct':float(100*passed.mean()),'failed':int((~passed).sum())}]
  if dta:
   for step in [.25,.125,.0625]:
    ix=np.flatnonzero(~passed)
    if len(ix):passed[ix]=pass_mask(g,r,pts[ix],np.array(m['spacing_zyx']),dd,dta,local,step)
    stages.append({'search_step_mm':step,'pass_pct':float(100*passed.mean()),'failed':int((~passed).sum())})
  report['gamma'][key]=stages;fail[key]=~passed;print(key,stages,flush=True)
def summarize(sel):
 if not sel.any():return {'voxels':0}
 return {'voxels':int(sel.sum()),'mean_error_pct':float(err[sel].mean()),'rms_error_pct':float(np.sqrt(np.mean(err[sel]**2))),'local11_failed':int((sel&fail['local_11']).sum()),'local33_failed':int((sel&fail['local_33']).sum())}
report['sections']={str(int(s)):summarize(mat[mask]==s) for s in np.unique(mat[mask])}
report['dose_bands']={f'{lo}-{hi}':summarize((rr>=r.max()*lo/100)&(rr<r.max()*hi/100)) for lo,hi in [(10,20),(20,50),(50,80),(80,101)]}
np.savez_compressed(out/'failure_masks.npz',points_zyx=pts,**fail)
# Audit how much of the previous fine-grid failure population was outside BODY.
a=np.load(base/'failure_masks.npz');qp=a['points_zyx'];b=body[tuple(qp.T)];report['previous_failure_inside_body']={k:{'all_failed':int(a[k].sum()),'inside_body':int((a[k]&b).sum()),'outside_body':int((a[k]&~b).sum())} for k in ['local_33','local_11','local_30']}
(out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True);zi=16;extent=[origin[0]-.5*dx,origin[0]+(nx-.5)*dx,origin[1]-.5*dy,origin[1]+(ny-.5)*dy]
axs[0].imshow(hu[zi],origin='lower',extent=extent,cmap='gray',vmin=-1000,vmax=1000);axs[0].contour(origin[0]+np.arange(nx)*dx,origin[1]+np.arange(ny)*dy,body[zi],levels=[.5],colors='lime');axs[0].set_title('BODY from RTSTRUCT (green)')
im=axs[1].imshow(np.where(mask[zi],100*(g[zi]-r[zi])/np.maximum(r[zi],1e-30),np.nan),origin='lower',extent=extent,vmin=-3,vmax=3,cmap='coolwarm');fig.colorbar(im,ax=axs[1],label='GPU-TOPAS / TOPAS (%)');axs[1].set_title('Relative error inside BODY, dose >=10%')
vol=np.zeros(shape,bool);vol[tuple(pts[fail['local_11']].T)]=True;axs[2].imshow(hu[zi],origin='lower',extent=extent,cmap='gray',vmin=-1000,vmax=1000);yy,xx=np.where(vol[zi]);axs[2].scatter(origin[0]+xx*dx,origin[1]+yy*dy,c='red',s=8);axs[2].set_title('Local 1%/1mm failures inside BODY')
for ax in axs:ax.set_xlabel('DICOM X (mm)');ax.set_ylabel('DICOM Y (mm)')
fig.savefig(out/'body_analysis.png',dpi=150);print('DONE',report['evaluated_voxels'],match,flush=True)
