//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// -------------------------------------------------------------------
//
// Adapted sampling algorithms from Geant4 11.3.2 G4IonFluctuations and
// G4UniversalFluctuation. Research-only C12 building block, not production.
#pragma once
#include <cmath>
#include <algorithm>

namespace carbon {
template<class Real> struct RestrictedFluctuationInput {
    Real kinetic_MeV, mass_MeV, mean_MeV;
    // Raw G4IonFluctuations::Dispersion, BEFORE finite-loss correction.
    Real ion_dispersion, universal_dispersion;
    Real cut_MeV, tmax_MeV, excitation_MeV, e0_MeV;
};
template<class Real> struct RestrictedFluctuationDraw { Real loss{}; bool valid{false}; };

template<class Real, class Uniform> class RestrictedFluctuationSampler {
    Uniform& uniform;
    int draws=0;
    bool good=true;
    Real u() {
        if(++draws>8192){good=false;return Real(.5);}
        Real x=uniform();
        if(!(x>=0 && x<1)){good=false;return Real(.5);}
        return std::max(x,Real(1e-12));
    }
    Real normal() {Real a=u(),b=u();return std::sqrt(-2*std::log(a))*std::cos(Real(6.283185307179586)*b);}
    Real gamma(Real shape) {
        bool small=shape<1;Real k=small?shape+1:shape;
        Real d=k-Real(1.0/3),c=1/std::sqrt(9*d);
        while(good) {
            Real x=normal(),v=1+c*x;
            if(v<=0)continue;
            v=v*v*v;Real w=u();
            if(w<1-Real(.0331)*x*x*x*x || std::log(w)<Real(.5)*x*x+d*(1-v+std::log(v)))
                return d*v*(small?std::pow(u(),1/shape):Real(1));
        }
        return 0;
    }
    int poisson(Real mean) {
        // Glandz leaves only small explicit Poisson components (<=8).
        if(!(mean>=0 && mean<=Real(8.001))){good=false;return 0;}
        Real threshold=std::exp(-mean),product=1;int n=0;
        do {++n;product*=u();}while(product>threshold && good);
        return n-1;
    }
    Real truncated_gaussian(Real mean, Real sigma) {
        while(good) {Real x=mean+sigma*normal();if(x>=0 && x<=2*mean)return x;}
        return 0;
    }
    Real glandz_gaussian(Real mean,Real variance) {
        Real sigma=std::sqrt(variance);
        if(mean<Real(.25)*sigma)return 2*u()*mean;
        return truncated_gaussian(mean,sigma);
    }
    Real universal(const RestrictedFluctuationInput<Real>& p) {
        Real mean=p.mean_MeV,cut=p.cut_MeV;
        if(mean<Real(1e-5))return mean;
        if(mean>=10*cut && p.tmax_MeV<=2*cut) {
            Real sigma=std::sqrt(p.universal_dispersion),sn=mean/sigma;
            return sn>=2 ? truncated_gaussian(mean,sigma) : mean*gamma(sn*sn)/(sn*sn);
        }
        if(cut<=p.e0_MeV)return mean;
        Real scaling=std::min(1+Real(.0005)/cut,Real(1.5));mean/=scaling;
        Real a1=0,e1=p.excitation_MeV;
        if(cut>e1) {
            a1=mean*Real(.44)/e1;
            Real fw=a1<42 ? Real(.1)+Real(3.9)*std::sqrt(a1/42) : Real(4);
            a1/=fw;e1*=fw;
        }
        Real w1=cut/p.e0_MeV;
        Real a3=Real(.56)*mean*(cut-p.e0_MeV)/(p.e0_MeV*cut*std::log(w1));
        if(a1<=0)a3/=Real(.56);
        Real loss=0;
        if(a1>8)loss+=glandz_gaussian(a1*e1,a1*e1*e1);
        else if(a1>0){int n=poisson(a1);if(n>0)loss+=(n+1-2*u())*e1;}
        if(a3>0) {
            Real emean=0,var=0,p3=a3,alpha=1;
            if(a3>8) {
                alpha=w1*(8+a3)/(w1*8+a3);
                Real alpha1=alpha*std::log(alpha)/(alpha-1);
                // Algebraically identical split: retain the bounded explicit
                // Poisson mean directly instead of subtracting two large counts.
                Real count=a3*(a3/(8+a3));
                emean=count*p.e0_MeV*alpha1;
                var=p.e0_MeV*p.e0_MeV*count*(alpha-alpha1*alpha1);
                p3=8*(a3/(8+a3));
            }
            Real w3=alpha*p.e0_MeV;
            if(cut>w3){Real w=(cut-w3)/cut;int n=poisson(p3);for(int i=0;i<n && good;++i)loss+=w3/(1-w*u());}
            if(var>0)loss+=glandz_gaussian(emean,var);
        }
        return loss*scaling;
    }
public:
    explicit RestrictedFluctuationSampler(Uniform& rng):uniform(rng){}
    RestrictedFluctuationDraw<Real> sample(const RestrictedFluctuationInput<Real>& p, bool ion_model=true, Real ion_charge=Real(6)) {
        draws=0;good=true;
        if(!(p.kinetic_MeV>0 && p.mass_MeV>0 && p.mean_MeV>=0 &&
             p.ion_dispersion>=0 && p.universal_dispersion>=0 && p.cut_MeV>0 &&
             p.tmax_MeV>=p.cut_MeV && p.excitation_MeV>0 && p.e0_MeV>0))return {};
        if(p.mean_MeV<=Real(1e-9))return {p.mean_MeV,true};
        Real loss;
        if(!ion_model || p.kinetic_MeV>Real(10.0/938.272013)*ion_charge*p.mass_MeV)loss=universal(p);
        else {
            Real var=p.ion_dispersion;
            if(p.mean_MeV>Real(.2)*p.kinetic_MeV) {
                Real g=1+p.kinetic_MeV/p.mass_MeV,beta2=1-1/(g*g);
                Real gam=1+(p.kinetic_MeV-p.mean_MeV)/p.mass_MeV;
                Real b2=std::max(1-1/(gam*gam),Real(.2)*beta2);
                Real x=b2/beta2;
                var*=Real(.25)*(1+x)*(1/(x*x*x)+(1/b2-Real(.5))/(1/beta2-Real(.5)));
            }
            if(var<=0)return {p.mean_MeV,true};
            Real sigma=std::sqrt(var),sn=p.mean_MeV/sigma;
            if(sn>=2)loss=truncated_gaussian(p.mean_MeV,sigma);
            else if(sn>Real(.1))loss=p.mean_MeV*gamma(sn*sn)/(sn*sn);
            else loss=2*p.mean_MeV*u();
        }
        return {loss,good && std::isfinite(loss) && loss>=0};
    }
};
} // namespace carbon
