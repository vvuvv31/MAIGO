"""Double-precision port of the active Urban angular sampler
(SampleCosineTheta + SimpleScattering), validated against the TOPAS
per-step MSC ntuple for C12 in Copper."""
import math, numpy as np
Z,Zp,A=29.0,6.0,12.0; NUC=931.49410242; ME=0.51099895
R_E=2.8179403262e-13;BOHR=5.29177210903e-9;HBARC=197.3269804e-13;BARN=1e-24
NA=6.02214076e23;RHO=8.96;ACU=63.546
w=Z**(1/6);Z13=w*w
facz=0.990395+w*(-0.168386+w*0.093286)
CT1=facz*(1-8.7780e-2/Z);CT2=facz*(4.0780e-2+1.7315e-4*Z)
CC1=2.3785-Z13*(4.1981e-1-Z13*6.3100e-2)
CC2=4.7526e-1+Z13*(1.7694-Z13*3.3885e-1)
CC3=2.3683e-1-Z13*(1.8111-Z13*3.2774e-1)
CC4=1.7888e-2+Z13*(1.9659e-2-Z13*2.6664e-3)
SIG0=13.24*BARN;HEC=74.510
EPS=2*ME*ME*BOHR*BOHR/(HBARC*HBARC);SIGF=2*math.pi*R_E*R_E
X0=10*12.8628/8.96
def sigma(kin):
    mass=A*NUC;tau=kin/mass;c=mass*tau*(tau+2)/(ME*(tau+1));ww=c-2
    t=0.5*(ww+math.sqrt(ww*ww+4*c));ek=ME*t;et=ek+ME
    b2=ek*(et+ME)/(et*et);bg2=ek*(et+ME)/(ME*ME)
    eps=EPS*bg2/(Z**(2/3))
    if eps<1e-4:s=2*eps*eps
    elif eps<1e10:s=math.log(1+2*eps)-2*eps/(1+2*eps)
    else:s=math.log(2*eps)-1+1/eps
    s*=Zp*Zp*Z*Z/(b2*bg2)
    tl=10.0;b2l=tl*(tl+2*ME)/((tl+ME)**2);bg2l=tl*(tl+2*ME)/(ME*ME)
    s=bg2l*SIG0*(1+HEC*(b2-b2l))/bg2 if ek>tl else s*SIGF
    return s*(1+0.30/(1+math.sqrt(1000*ek)))
def mfp(kin): return 10.0/(NA*RHO/ACU*sigma(kin))
def theta0(t,kin):
    mass=A*NUC;ib=(kin+mass)/(kin*(kin+2*mass));y=t/X0
    return 13.6*Zp*math.sqrt(y)*ib*(CT1+CT2*math.log(y))
TBIG,TSMALL=8.0,1e-16; NUM=0.01
def sample_cos(t,kin,lam,rng):
    tau=t/lam
    if tau>=TBIG: return -1+2*rng.random()
    if tau<TSMALL: return 1.0
    if tau<NUM: xm=1-tau*(1-0.5*tau);x2m=1-tau*(5-6.25*tau)/3
    else: xm=math.exp(-tau);x2m=(1+2*math.exp(-2.5*tau))/3
    def simple():
        a=(2*xm+9*x2m-3)/(2*xm-3*x2m+1);pr=(a+2)*xm/a
        u0,u1=rng.random(),rng.random()
        return -1+2*math.exp(math.log(u0)/(a+1)) if u1<pr else -1+2*u0
    th0=theta0(t,kin);th2=th0*th0
    if th2<TSMALL: return 1.0
    if th0>math.pi/6: return simple()
    x=th2*(1-th2/12)
    if th2>NUM: x=(2*math.sin(0.5*th0))**2
    ltau=math.log(tau);u=math.exp(ltau/6);xx=math.log(lam/X0)
    xsi=max(CC1+u*(CC2+CC3*u)+CC4*xx,1.9)
    c=xsi
    if abs(c-3)<0.001:c=3.001
    elif abs(c-2)<0.001:c=2.001
    c1=c-1;ea=math.exp(-xsi);eaa=1-ea
    xm1=1-(1-(1+xsi)*ea)*x/eaa;x0=1-xsi*x
    if xm1<=0.999*xm: return simple()
    b=1+(c-xsi)*x;b1=b+1;bx=c*x
    eb1=math.exp(math.log(b1)*c1);ebx=math.exp(math.log(bx)*c1);d=ebx/eb1
    xm2=(x0+d-(bx-b1*d)/(c-2))/(1-d)
    f1=ea/eaa;f2=c1/(c*(1-d));pr=f2/(f1+f2)
    qp=xm/(pr*xm1+(1-pr)*xm2)
    u0,u1=rng.random(),rng.random()
    if u0<qp:
        if u1<pr: return 1+math.log(ea+rng.random()*eaa)*x
        var=(1-d)*rng.random()
        if var<NUM*d:
            var/=d*c1;return -1+var*(1-0.5*var*c)*(2+(c-xsi)*x)
        return 1+x*(c-xsi-c*math.exp(-math.log(var+d)/c1))
    return -1+2*u1
def sample_theta(t,kin,lam,n,rng):
    ct=np.array([sample_cos(t,kin,lam,rng) for _ in range(n)])
    return np.arccos(np.clip(ct,-1,1))
