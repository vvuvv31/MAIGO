"""Depth-resolved descriptive double-Gaussian fits; differences are NOT MC error bars."""
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sigma_model import mixture_pixel_mass

X=np.arange(160)-79.5
R2=(X[:,None]**2+X[None,:]**2).ravel()
INDEX=np.floor(np.sqrt(R2)).astype(int)
USE=INDEX<80
INDEX=INDEX[USE]; RR=R2[USE]
AREA=np.bincount(INDEX,minlength=80)

def fit_plane(plane):
    if plane.shape!=(160,160) or not np.isfinite(plane).all() or np.any(plane<0):
        raise ValueError('Invalid plane')
    total=float(plane.sum())
    if total<=0:return None
    obs=np.bincount(INDEX,weights=plane.ravel()[USE],minlength=80)/AREA/total
    def model(p):
        c,h,f=p
        pixels=mixture_pixel_mass(c,h,f).ravel()[USE]
        return np.bincount(INDEX,weights=pixels,minlength=80)/AREA
    fits=[least_squares(lambda p:1000*(model(p)-obs),s,bounds=([.5,6,0],[6,80,1]),max_nfev=500)
          for s in ([3,12,.1],[4,30,.3])]
    best=min(fits,key=lambda f:f.cost)
    residual=float(np.linalg.norm(model(best.x)-obs)/np.linalg.norm(obs))
    cond=float(np.linalg.cond(best.jac))
    c,h,f=map(float,best.x)
    reasons=[]
    if not best.success:reasons.append('optimizer')
    if np.any(best.active_mask) or c<=.5001 or c>=5.9999 or h<=6.0001 or h>=79.9999:reasons.append('bound')
    if f<.001 or f>.999:reasons.append('vanishing_component')
    if residual>.1:reasons.append('residual_over_10pct')
    if not np.isfinite(cond) or cond>1e6:reasons.append('ill_conditioned')
    # A near-equal alternative minimum with different widths is not identified.
    for other in fits:
        if other.cost<=best.cost*1.01+1e-12 and np.max(np.abs(other.x[:2]-best.x[:2])/best.x[:2])>.1:
            reasons.append('multistart_disagreement');break
    return dict(core=c,halo=h,fraction=f,residual=residual,condition=cond if np.isfinite(cond) else None,
                valid=not reasons,reasons=reasons)

def curves(data,xmax):
    out=[]
    # Fixed nonoverlapping 2mm windows; 4 original 0.5mm slices. No smoothing.
    for k in range(0,data.shape[0]-3,4):
        center=(k+2)*.5
        if center>xmax:break
        out.append(dict(depth_mm=center,fit=fit_plane(np.asarray(data[k:k+4],dtype=float).sum(0))))
    return out
