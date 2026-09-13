"""Re-evaluate existing complete doses on nested gamma lattices inside RTSTRUCT BODY."""
from pathlib import Path
import sys,json,hashlib,struct,time
import numpy as np
import pydicom,yaml
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path as Polygon
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'tools'))
from evaluate_topas10x_gpu_gamma import pass_mask
OUT=Path(__file__).resolve().parent
RUNS={'20022516':'final_stopping_20260911/20022516_extended','RT06423':'exactfaces_full20_20260910/RT06423','RT07575':'final_stopping_20260911/RT07575'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(case):
 dest=OUT/(case+'.json');assert not dest.exists()
 run=Path('/mnt/sda/wuwei')/RUNS[case];refdir=Path('/mnt/sda/wuwei/ct_previous_full20_20260909')/case
 m=json.loads((refdir/'manifest.json').read_text());state=json.loads((run/'execution.json').read_text())
 assert state['status']=='complete';assert sha(run/'gpu_sum.raw')==state['aggregate_sha256'];assert sha(refdir/'topas_sum.raw')==m['reference_sum_sha256']
 assert sum(x['histories'] for x in state['completed'])==m['histories']
 for item in state['completed']:
  q=json.loads((Path(item['directory'])/'out/config/quality_report.json').read_text());assert q['queue_overflow_count']==0 and q.get('queue_overflow_energy_MeV',0)==0
 def mapped(a):return np.flip(a.transpose(1,2,0),2) if m['mapping']=='packed_xneg' else a
 assert m['mapping'] in ['native','packed_xneg']
 g=mapped(np.fromfile(run/'gpu_sum.raw','<f4').reshape(m['gpu_shape_zyx']));r=np.fromfile(refdir/'topas_sum.raw','<f4').reshape(m['topas_shape_zyx'])
 assert g.shape==r.shape and np.all(np.isfinite(g)&(g>=0)) and np.all(np.isfinite(r)&(r>=0))
 dicom=REPO/'benchmark/topas10x'/('RT07575_pbs_s1' if case=='RT07575' else case)/'dicom'
 rsfile=next(dicom.glob('RS*.dcm'));rs=pydicom.dcmread(rsfile);roi=next(x for x in rs.StructureSetROISequence if x.ROIName.strip().upper()=='BODY');rc=next(x for x in rs.ROIContourSequence if x.ReferencedROINumber==roi.ROINumber)
 ct=[]
 for p in dicom.glob('*.dcm'):
  d=pydicom.dcmread(p)
  if d.Modality=='CT':ct.append((p,d))
 ct.sort(key=lambda v:float(v[1].ImagePositionPatient[2]));first=ct[0][1];origin=np.array(first.ImagePositionPatient,float);z=np.array([float(d.ImagePositionPatient[2]) for p,d in ct]);dy,dx=map(float,first.PixelSpacing);ny,nx=int(first.Rows),int(first.Columns)
 assert str(roi.ReferencedFrameOfReferenceUID)==str(first.FrameOfReferenceUID)
 assert (len(ct),ny,nx)==r.shape and np.allclose([z[1]-z[0],dy,dx],m['spacing_zyx'])
 for p,d in ct:
  assert np.allclose(d.ImageOrientationPatient,[1,0,0,0,1,0]) and np.allclose(d.ImagePositionPatient[:2],origin[:2]) and np.allclose(d.PixelSpacing,[dy,dx])
 assert np.allclose(np.diff(z),m['spacing_zyx'][0])
 planes={}
 for c in rc.ContourSequence:
  assert c.ContourGeometricType=='CLOSED_PLANAR';v=np.array(c.ContourData,float).reshape(-1,3);assert np.ptp(v[:,2])<1e-4
  zz=float(v[0,2]);assert zz not in planes,'Multiple contour loops require explicit topology handling'
  rows=(v[:,1]-origin[1])/dy;cols=(v[:,0]-origin[0])/dx
  rr,cc=np.mgrid[max(0,int(np.floor(rows.min()))):min(ny-1,int(np.ceil(rows.max())))+1,max(0,int(np.floor(cols.min()))):min(nx-1,int(np.ceil(cols.max())))+1]
  vertices=np.column_stack([cols,rows]);path=Polygon(np.vstack([vertices,vertices[0]]),closed=True);points=np.column_stack([cc.ravel(),rr.ravel()]);inside=path.contains_points(points,radius=1e-7)|path.contains_points(points,radius=-1e-7)
  a=np.zeros((ny,nx),bool);a[rr.ravel()[inside],cc.ravel()[inside]]=True;planes[zz]=a
 cz=np.array(sorted(planes));assert z.min()>=cz.min()-1e-4 and z.max()<=cz.max()+1e-4
 sdf=np.array([distance_transform_edt(planes[q],sampling=[dy,dx])-distance_transform_edt(~planes[q],sampling=[dy,dx]) for q in cz]);body=[]
 for zz in z:
  hi=int(np.searchsorted(cz,zz))
  if hi==0:b=sdf[0]>0
  elif hi==len(cz):b=sdf[-1]>0
  else:
   lo=hi-1;w=(zz-cz[lo])/(cz[hi]-cz[lo]);b=((1-w)*sdf[lo]+w*sdf[hi])>0
  body.append(b)
 body=np.array(body)
 cfg=yaml.safe_load((run/'shard_01/config.yaml').read_text())
 with Path(cfg['ct_grid_file']).open('rb') as f:
  h=struct.unpack('<5I6f',f.read(44));n=h[2]*h[3]*h[4];f.seek(44+4*n);mat=mapped(np.fromfile(f,'u1',n).reshape(h[4],h[3],h[2]))
 hu=np.array([d.pixel_array.astype(float)*float(d.RescaleSlope)+float(d.RescaleIntercept) for p,d in ct]);edges=np.array([-1000,-950,-120,-83,-53,-23,7,18,80,120,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2995,2996]);expected=np.clip(np.searchsorted(edges,hu,side='right')-1,0,24);match=float((mat==expected).mean());assert match>.999
 if case=='RT07575':assert np.array_equal(body,np.load(REPO/'evidence/step-31/rt07575-gamma-spatial-20260911/body/body_mask_zyx.npy'))
 mask=body&(r>=.1*r.max());pts=np.argwhere(mask);assert len(pts)>0
 report={'case':case,'run':str(run),'reference':str(refdir),'gpu_sha256':sha(run/'gpu_sum.raw'),'reference_sha256':sha(refdir/'topas_sum.raw'),'histories':m['histories'],'rtstruct_sha256':sha(rsfile),'ct_file_sha256':{p.name:sha(p) for p,d in ct},'roi':str(roi.ROIName),'roi_number':int(roi.ROINumber),'evaluated_voxels':len(pts),'body_voxels':int(body.sum()),'material_alignment':match,'body_mask_sha256':hashlib.sha256(body.tobytes()).hexdigest(),'physics_config':{k:cfg.get(k) for k in ['ct_primary_midpoint_stopping','ct_primary_midpoint_stopping_diagnostic','ct_secondary_exact_faces','ct_secondary_exact_faces_diagnostic','ct_secondary_ion_section_stopping_file']},'method':'All BODY reference voxels with TOPAS dose >=10% full-volume maximum; global normalization is full-volume TOPAS maximum; local normalization is query dose. Search full GPU dose without clipping it to BODY; trilinear interpolation; no extrapolation or dose normalization. Nested spherical lattices: previous passes remain valid. Zero DTA is same-voxel dose difference. BODY uses voxel-centre polygon rasterization and signed-distance interpolation between contour planes.','gamma':{}}
 print(case,'BODY evaluation voxels',len(pts),'material alignment',match,flush=True)
 for dd,dta in [(3,3),(2,2),(1,1),(3,0)]:
  for local in [False,True]:
   key=f"{'local' if local else 'global'}_{dd}{dta}";passed=np.zeros(len(pts),bool);stages=[]
   for step in [1.,.5,.25,.125]:
    start=time.monotonic();ix=np.flatnonzero(~passed)
    if len(ix) and (dta or not stages):passed[ix]=pass_mask(g,r,pts[ix],np.array(m['spacing_zyx']),dd,dta,local,step)
    stages.append({'step_mm':step,'pass_percent':100*float(passed.mean()),'passed':int(passed.sum()),'failed':int((~passed).sum()),'seconds':time.monotonic()-start})
    print(case,key,stages[-1],flush=True)
   report['gamma'][key]=stages
   (OUT/(case+'.partial.json')).write_text(json.dumps(report,indent=2)+'\n')
 dest.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
