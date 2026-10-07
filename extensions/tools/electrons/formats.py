import numpy as np

OSC=np.dtype([(s,'<f4') for s in ['strength','ionization','resonance','recoil_cut','binding']]+[('z','<u4'),('shell_flag','<u4')])

SHELL=np.dtype([('cdf','<u8'),('binding','<f4'),('z','<u4')])

def shells(path):
    r=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True));w=r['electron_fraction']*r['shell_fraction'];w/=w.sum()
    out=np.empty(len(r),SHELL);cdf=np.cumsum(w.astype(np.longdouble));out['cdf']=np.minimum(cdf*np.longdouble(2)**64,np.longdouble(2)**64-1).astype('<u8');out['cdf'][-1]=np.iinfo(np.uint64).max
    out['binding']=r['binding_MeV'];out['z']=r['Z'];return out
