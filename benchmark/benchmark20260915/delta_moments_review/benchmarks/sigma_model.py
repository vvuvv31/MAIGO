"""Pixel-integrated Gaussian mixture conditioned on the scored square ROI."""
import numpy as np
from scipy.special import erf

CENTERS=np.arange(160)-79.5

def gaussian_pixel_mass(sigma):
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Invalid sigma')
    p=.5*(erf((CENTERS+.5)/(np.sqrt(2)*sigma))-erf((CENTERS-.5)/(np.sqrt(2)*sigma)))
    return p[:,None]*p[None,:]

def mixture_pixel_mass(core,halo,fraction):
    if not 0<=fraction<=1:raise ValueError('Invalid mixture fraction')
    p=(1-fraction)*gaussian_pixel_mass(core)+fraction*gaussian_pixel_mass(halo)
    # Observed dose is normalized by the same finite square, not by infinite-plane dose.
    return p/p.sum()
