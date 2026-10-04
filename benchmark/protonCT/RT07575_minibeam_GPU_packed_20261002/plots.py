"""Patient-grid dose comparisons, no registration or fitted rescaling."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,read_mhd
from metrics import roi_indices

def make_plots(results):
    ref,f=read_mhd(ROOT/'inputs/physical_dose.mhd');gpu,_=read_mhd(results/'dose_4Gy.mhd')
    rois=roi_indices();mask=np.zeros((35,417,505),bool);mask.reshape(-1)[rois['GTV']]=True;mask=mask.transpose(0,2,1)
    iz=int(np.argmax(mask.sum(axis=(1,2))))
    x0,y0,z0=map(float,f['Offset'].split());sx,sy,sz=map(float,f['ElementSpacing'].split())
    ext=[x0-.5*sx,x0+(417-.5)*sx,y0-.5*sy,y0+(505-.5)*sy]
    fig,ax=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
    for a,arr,title in zip(ax,[ref[iz],gpu[iz],gpu[iz]-ref[iz]],['TOPAS weighted DIJ','GPU group32','GPU - TOPAS']):
        diff=title=='GPU - TOPAS';im=a.imshow(arr,origin='lower',extent=ext,cmap='RdBu_r' if diff else 'inferno',vmin=-1 if diff else 0,vmax=1 if diff else ref.max(),aspect='equal');a.contour(mask[iz],levels=[.5],colors=['cyan'],extent=ext,origin='lower',linewidths=.6);a.set_title(title);a.set_xlabel('Patient X (mm)');a.set_ylabel('Patient Y (mm)');fig.colorbar(im,ax=a,label='Gy')
    fig.suptitle(f'4 Gy plan, patient Z = {z0+iz*sz:.2f} mm');fig.savefig(results/'dose_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for dose,label in [(ref,'TOPAS'),(gpu,'GPU')]:
        flat=dose.transpose(0,2,1).reshape(-1)
        for name,style in [('GTV','-'),('BODY','--')]:
            v=np.sort(flat[rois[name]]);ax[0].plot(v,100*(1-np.arange(len(v))/len(v)),style,label=label+' '+name)
    ax[0].set(xlabel='Dose (Gy)',ylabel='Volume ≥ dose (%)',xlim=(0,8));ax[0].legend()
    # Beam travels -patient X; pooled lateral profile retains slit modulation.
    ix=int(np.median(np.argwhere(mask)[:,2]));columns=np.arange(max(0,ix-5),min(417,ix+6))
    for dose,label in [(ref,'TOPAS'),(gpu,'GPU')]:ax[1].plot(y0+np.arange(505)*sy,dose[iz][:,columns].mean(axis=1),label=label)
    ax[1].set(xlabel='Patient Y (mm), transverse slit axis',ylabel='Mean dose in 5.5 mm depth slab (Gy)');ax[1].legend();fig.savefig(results/'dvh_profile.png',dpi=160);plt.close(fig)
