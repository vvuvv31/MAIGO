"""Native cubic rate intervals and analytical positive-support witnesses."""
import numpy as np
from numpy.polynomial import Polynomial
TS=np.array([.125,.375,.625,.875])
VANDER=np.vander(TS,4,increasing=True)

def coefficients(row):return np.linalg.solve(VANDER,row[3:7])

def rate(rows,process,energy):
    r=rows[rows[:,0]==process];i=min(len(r)-1,np.searchsorted(r[:,1],energy,side='right')-1)
    if i<0 or energy>r[-1,2]:raise ValueError('rate energy outside coverage')
    a=r[i];return max(0.,float(Polynomial(coefficients(a))((energy-a[1])/(a[2]-a[1]))))

def witnesses(rows,edges):
    """One interior witness per bin/process having strictly positive rate.

    Derivative roots locate every cubic extremum. Bisect a positive endpoint
    towards the interval interior to avoid assigning a bin-edge value to the
    wrong conditional bin. No positive value is thresholded away.
    """
    result={}
    for row in rows:
        p=int(row[0]);lo,hi=row[1:3];poly=Polynomial(coefficients(row));extrema=poly.deriv().roots()
        for cell in range(len(edges)-1):
            a=max(lo,edges[cell]);b=min(hi,edges[cell+1])
            if a>=b:continue
            ta=(a-lo)/(hi-lo);tb=(b-lo)/(hi-lo)
            candidates=[ta,tb,(ta+tb)/2]+[float(z.real) for z in extrema if abs(z.imag)<1e-13 and ta<z.real<tb]
            best=max(candidates,key=poly)
            if poly(best)<=0:continue
            if best==ta or best==tb:
                t=(best+(ta+tb)/2)/2
                for _ in range(60):
                    if poly(t)>0:break
                    t=(t+best)/2
                best=t
            e=lo+(hi-lo)*best
            assert edges[cell]<=e<edges[cell+1] and rate(rows,p,e)>0
            if (cell,p) not in result:result[cell,p]=e
    return result
