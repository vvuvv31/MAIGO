#!/usr/bin/env python3
"""Isocenter planes and beam-coordinate lateral profiles; no dose fitting."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,read_mhd,sha,save_json
from metrics import roi_indices

OUT=ROOT/'results/isocenter_profiles'
LABELS=['TOPAS','Legacy GPU','Schneider joint GPU']
COLORS=['#202020','#2673b8','#d24732']

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def packed(a):return np.flip(a,axis=2).transpose(2,0,1)

def interpolate(a,depth,u,v):
    depth,u,v=np.broadcast_arrays(depth,u,v)
    return map_coordinates(a,[(depth-.25)/.5,(v+32.6383)/2,(u+168.8515)/.5],order=1,mode='constant',cval=np.nan,prefilter=False)

def stats(a,b):
    return dict(mean_signed_Gy=float(np.mean(b-a)),max_absolute_Gy=float(np.max(abs(b-a))),relative_L1=float(np.sum(abs(b-a))/np.sum(abs(a))))

def planes(data,body,gtv,name,x,y,axis_labels,fixed_label):
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    extent=[x[0]-.25,x[-1]+.25,y[0]-(y[1]-y[0])/2,y[-1]+(y[1]-y[0])/2]
    vmax=max(float(np.nanmax(a)) for a in data)
    differences=[data[2]-data[0],data[1]-data[0],data[2]-data[1]]
    limits=[max(np.max(abs(differences[0])),np.max(abs(differences[1])))]*2+[np.max(abs(differences[2]))]
    for i,(a,label) in enumerate(zip(data,LABELS)):
        ax=axs[0,i];im=ax.imshow(a,origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='inferno',aspect='equal');ax.set_title(label);fig.colorbar(im,ax=ax,label='Dose (Gy)',shrink=.85)
    for i,(a,label,lim) in enumerate(zip(differences,['New GPU - TOPAS','Legacy GPU - TOPAS','New GPU - legacy GPU'],limits)):
        ax=axs[1,i];im=ax.imshow(a,origin='lower',extent=extent,vmin=-lim,vmax=lim,cmap='RdBu_r',aspect='equal');ax.set_title(label);fig.colorbar(im,ax=ax,label='Dose difference (Gy)',shrink=.85)
    for ax in axs.flat:
        for mask,col in [(body,'#57bbaf'),(gtv,'#70ddff')]:
            if mask.min()<.5<mask.max():ax.contour(x,y,mask,levels=[.5],colors=[col],linewidths=.7)
        ax.axvline(0,color='gray',lw=.5,ls=':');ax.axhline(0,color='gray',lw=.5,ls=':');ax.plot(0,0,'+',color='#00e7ff',ms=9,mew=1)
        ax.set_xlabel(axis_labels[0]);ax.set_ylabel(axis_labels[1])
    fig.suptitle(f'RT07575 full plan, 4 Gy prescription | {fixed_label}\nBODY: green; GTV: cyan; +: isocenter | native grid, linear interpolation only at the requested plane',fontsize=12)
    save(fig,name)
    np.savez_compressed(OUT/(name+'.npz'),x_mm=x,y_mm=y,topas_Gy=data[0],legacy_Gy=data[1],joint_Gy=data[2],body=body,gtv=gtv)
    roi=(body>.5)&(data[0]>=.1*np.max(data[0]))
    return dict(new_vs_TOPAS=stats(data[0],data[2]),legacy_vs_TOPAS=stats(data[0],data[1]),new_vs_legacy=stats(data[1],data[2]),
                BODY_above_10pct_plane_TOPAS_peak=dict(pixels=int(roi.sum()),new_vs_TOPAS=stats(data[0][roi],data[2][roi]),new_vs_legacy=stats(data[1][roi],data[2][roi])))

def main():
    OUT.mkdir(exist_ok=True)
    m=json.loads((ROOT/'manifest.json').read_text());old=Path(m['previous_campaign'])
    paths=[ROOT/'inputs/physical_dose.mhd',old/'results/dose_4Gy.mhd',ROOT/'results/dose_4Gy.mhd']
    dose=[];geometry_fields=None
    for path in paths:
        a,f=read_mhd(path)
        if geometry_fields is None:geometry_fields=dict(f)
        else:
            for key in ['Offset','ElementSpacing','DimSize','TransformMatrix']:assert f[key]==geometry_fields[key]
        assert f['TransformMatrix']=='1 0 0 0 1 0 0 0 1'
        assert list(map(float,f['ElementSpacing'].split()))==[.5,.5,2.]
        dose.append(packed(a))
    assert m['ct_check']['canonical_origin_xyz_mm']==[-169.1015,-33.6383,0.]
    rois=roi_indices();masks=[]
    for name in ['BODY','GTV']:
        a=np.zeros((35,417,505),np.float32);a.reshape(-1)[rois[name]]=1;masks.append(packed(a.transpose(0,2,1)))
    u=-168.8515+np.arange(505)*.5;v=-32.6383+np.arange(35)*2;depth=.25+np.arange(417)*.5;iso=117.0136
    ux=u[(u>=-35)&(u<=35)];dd=depth[(depth>=25)&(depth<=170)]
    planar=[interpolate(a,dd[:,None],ux[None,:],0) for a in dose]
    pb,pg=[interpolate(a,dd[:,None],ux[None,:],0) for a in masks]
    results={}
    results['longitudinal']=planes(planar,pb,pg,'isocenter_longitudinal',ux,dd-iso,['Lateral u (mm)','Depth relative to isocenter (mm)'],'plane through beam axis: v = 0')
    vv=v[(v>=-25)&(v<=25)]
    planar=[interpolate(a,iso,ux[None,:],vv[:,None]) for a in dose]
    pb,pg=[interpolate(a,iso,ux[None,:],vv[:,None]) for a in masks]
    results['transverse']=planes(planar,pb,pg,'isocenter_transverse',ux,vv,['Lateral u (mm)','Second transverse axis v (mm)'],'plane normal to beam: depth at isocenter')
    depths=[40.,60.,80.,100.,iso,125.,130.,140.,150.]
    xx=u[(u>=-30)&(u<=30)]
    profiles=np.array([[interpolate(a,d,xx,0) for a in dose] for d in depths])
    fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    difffig,daxs=plt.subplots(3,3,figsize=(15,10),layout='constrained')
    local=[]
    with (OUT/'lateral_profiles.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['depth_from_CT_entrance_mm','depth_from_isocenter_mm','u_mm','v_mm','TOPAS_Gy','legacy_GPU_Gy','joint_GPU_Gy','joint_minus_TOPAS_Gy','joint_minus_legacy_Gy'])
        for i,(d,curves,ax,dax) in enumerate(zip(depths,profiles,axs.flat,daxs.flat)):
            for a,label,col,style in zip(curves,LABELS,COLORS,['-','--','-']):ax.plot(xx,a,label=label,color=col,lw=1.1,ls=style,alpha=.9)
            title=f'CT depth {d:.2f} mm | iso {d-iso:+.2f} mm'
            ax.set(title=title,xlabel='Lateral u (mm)',ylabel='Dose (Gy)',xlim=(-30,30),ylim=(0,None));ax.grid(alpha=.18)
            if i==0:ax.legend(fontsize=8)
            peak=float(curves[0].max());localmask=curves[0]>=.1*peak
            delta=[curves[2]-curves[0],curves[1]-curves[0],curves[2]-curves[1]]
            for diff,label,col,style in zip(delta,['New - TOPAS','Legacy - TOPAS','New - legacy'],[COLORS[2],COLORS[1],'#168b5f'],['-','--','-']):dax.plot(xx,100*diff/peak,label=label,color=col,lw=1,ls=style)
            dax.axhline(0,color='gray',lw=.5);dax.grid(alpha=.18);dax.set(title=title,xlabel='Lateral u (mm)',ylabel='% of TOPAS peak at this depth',xlim=(-30,30))
            if i==0:dax.legend(fontsize=8)
            local.append(dict(depth_from_CT_entrance_mm=d,depth_from_isocenter_mm=d-iso,TOPAS_peak_Gy=peak,new_vs_TOPAS=stats(curves[0],curves[2]),new_vs_legacy=stats(curves[1],curves[2]),new_vs_TOPAS_max_local_pct_above_10pct_profile_peak=float(np.max(100*abs(delta[0][localmask])/curves[0][localmask])),new_vs_legacy_max_pct_of_TOPAS_profile_peak=float(np.max(100*abs(delta[2])/peak))))
            for j,x in enumerate(xx):writer.writerow([d,d-iso,x,0,*curves[:,j],delta[0][j],delta[2][j]])
    fig.suptitle('RT07575 lateral dose along beam incidence | v = 0, 0.5 mm lateral samples\nLinear interpolation to requested depth and v = 0; no slab averaging or smoothing',fontsize=13)
    difffig.suptitle('Lateral dose differences | normalization is the TOPAS profile peak at each depth\nThis is not voxel-local percentage error; no dose fitting',fontsize=13)
    save(fig,'lateral_dose_depths');save(difffig,'lateral_difference_depths')
    # Magnify three entrance-to-isocenter profiles without creating extra spatial samples.
    fig,axs=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for col,i in enumerate([0,2,4]):
        for a,label,color,style in zip(profiles[i],LABELS,COLORS,['-','--','-']):axs[0,col].plot(xx,a,label=label,color=color,ls=style,lw=1.2,marker='.',ms=2.5)
        axs[0,col].set(title=f'CT depth {depths[i]:.2f} mm',ylabel='Dose (Gy)',xlim=(-12,12),ylim=(0,None));axs[0,col].grid(alpha=.2)
        axs[1,col].plot(xx,profiles[i,2]-profiles[i,1],color='#168b5f',marker='.',ms=3,lw=1);axs[1,col].axhline(0,color='gray',lw=.5);axs[1,col].set(xlim=(-12,12),xlabel='Lateral u (mm)',ylabel='New - legacy (Gy)');axs[1,col].grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle('Minibeam lateral detail | dots are the original 0.5 mm lateral sample positions\nBottom row isolates the new electron model change');save(fig,'lateral_minibeam_detail')
    # Dose along the central beam; interpolate laterally only for this diagnostic.
    rays=[interpolate(a,depth,0,0) for a in dose];bodyray=interpolate(masks[0],depth,0,0);occupied=depth[bodyray>.5]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for a,label,col,style in zip(rays,LABELS,COLORS,['-','--','-']):ax.plot(depth-iso,a,label=label,color=col,ls=style,lw=1)
    ax.axvline(0,color='gray',ls=':',label='Isocenter');ax.axvspan(depth[0]-iso,occupied.min()-iso,color='gray',alpha=.12,label='Before BODY on central ray')
    ax.set(xlabel='Depth relative to isocenter (mm)',ylabel='Central-axis dose (Gy)',title='Central-axis depth dose (u = v = 0)',xlim=(depth[0]-iso,depth[-1]-iso));ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'central_axis_depth_dose')
    origin=np.array(list(map(float,geometry_fields['Offset'].split())))
    # Isocenter in patient coordinates, derived independently from centered
    # DICOM geometry and the verified passive Rz(90deg) patient placement.
    patient_center=origin+np.array([416*.5,504*.5,34*2])/2
    patient_iso=patient_center+np.array([-12.7636,42.8515,-1.3617])
    assert np.allclose(patient_iso,[-12.2636,124.5515,-781.5517],atol=1e-8)
    save_json(OUT/'analysis.json',dict(status='COMPLETED_POSTPROCESSING_NO_NEW_TRANSPORT',histories=m['histories'],source_dose_sha256={label:sha(path.with_suffix('.raw')) for label,path in zip(LABELS,paths)},geometry=dict(isocenter_world_xyz_mm=[0,0,0],isocenter_patient_xyz_mm=patient_iso.tolist(),isocenter_depth_from_CT_entrance_mm=iso,nominal_beam_direction_patient_xyz=[-1,0,0],u_patient_direction=[0,1,0],v_patient_direction=[0,0,1],central_ray_BODY_entry_depth_mm=float(occupied.min()),central_ray_BODY_exit_depth_mm=float(occupied.max()),grid_mm=[.5,.5,2.],plane_interpolation='Trilinear interpolation, preserving native varying-axis positions; no additional lateral samples, smoothing, depth averaging, fitted normalization or registration',depth_definition='Geometric distance from CT entrance plane, not radiological/water-equivalent or per-ray skin depth'),planes=results,profiles=local,limitations=['TOPAS/GPU voxel size0.5x0.5x2mm cannot resolve subvoxel minibeam structure.','Exact isocenter plane lies between slices; linear interpolation mixes adjacent native slices.','Single-line profile differences include primary and response-bank noise; deep low-dose local percentages can be large.','All doses retain existing4Gy-plan incident-primary normalization; no renormalization by profile peak.']))
    print(json.dumps(dict(patient_iso=patient_iso.tolist(),planes=results,profiles=local),indent=2))

if __name__=='__main__':main()
