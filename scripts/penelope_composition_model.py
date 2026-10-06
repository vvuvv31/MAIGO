"""Offline analytic actual-cut Penelope oscillator selection reference.

Reconstructs the original100-point shell-probability energy grid from one
composition's oscillators. This is not a GPU runtime or an atomic relaxation
model. In particular, native log interpolation is NOT renormalized: its
residual probability belongs to the last oscillator, as in SampleFinalState.
"""
import numpy as np
MASS=.51099895
GRID=np.geomspace(1e-4,1e5,100)

def hard_shell_xs(oscillators,energy,cut,delta,raw=False):
    ion=oscillators['ionization_MeV'];res=oscillators['resonance_MeV'];recoil=oscillators['recoil_cut_MeV']
    gamma=1+energy/MASS;beta=(gamma*gamma-1)/(gamma*gamma);amol=(energy/(energy+MASS))**2
    p=np.sqrt(energy*(energy+2*MASS));remaining=np.maximum(0,energy-res)
    pp=np.sqrt(remaining*(remaining+2*MASS))
    # Rationalize sqrt((p-pp)^2+m^2)-m; preserves the small recoil limit.
    difference=(res*(2*energy+2*MASS-res))/(p+pp)
    qm=difference*difference/(np.sqrt(difference*difference+MASS*MASS)+MASS)
    small=res<=1e-6*energy
    approx=res*res/(beta*2*MASS);qm=np.where(small,approx*(1-approx*.5/MASS),qm)
    distant=(energy>res)&(qm<recoil)&(cut<=res)
    h0=np.where(distant,(np.log(recoil*(qm+2*MASS)/(qm*(recoil+2*MASS)))+np.maximum(np.log(gamma*gamma)-beta-delta,0))/res,0.)
    wl=np.maximum(cut,recoil);ee=energy+ion;wu=.5*ee;active=wl<wu-1e-11
    # Equal to EINaT1 H0; the dimensionful prefactor matters only at the
    # native log floor. Units here are mm^2 and MeV.
    w=np.minimum(wl,wu)
    h0+=np.where(active,1/(ee-wu)-1/(ee-w)-1/wu+1/w+(1-amol)*np.log(((ee-wu)*w)/((ee-w)*wu))/ee+amol*(wu-w)/(ee*ee),0.)
    constant=np.pi*(2.8179403262e-12)**2*2*MASS/beta
    values=np.where(energy>=ion,oscillators['strength']*constant*h0,0.)
    return values if raw else np.maximum(values,1e-40)

def interpolated_shell_weights(oscillators,energy,cut,delta_at_energy):
    upper=np.clip(np.searchsorted(GRID,energy),1,len(GRID)-1);lower=upper-1
    t=np.log(energy/GRID[lower])/np.log(GRID[upper]/GRID[lower])
    values=[]
    for e in GRID[[lower,upper]]:
        xs=hard_shell_xs(oscillators,e,cut,delta_at_energy(float(e)))
        values.append(np.log(xs)-np.log(xs.sum()))
    return np.exp((1-t)*values[0]+t*values[1])
