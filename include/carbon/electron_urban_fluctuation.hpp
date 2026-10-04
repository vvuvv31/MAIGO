#pragma once
#include <cmath>
#include <algorithm>
namespace carbon {
// Electron-only G4UrbanFluctuation/G4UniversalFluctuation (Geant4 11.3.2).
// See LICENSES/Geant4.txt. The heavy-particle Gaussian/Gamma regime does not
// apply to electrons. Normal/Poisson variates use portable algorithms with
// the same probability laws, not the same CLHEP random-number consumption.
struct ElectronUrbanMaterial {double mass{},excitation{},e0{},e1{},e2{},f1{},f2{};};
struct ElectronUrbanLoss {bool valid{};double loss{};unsigned random_draws{};};
template<class Uniform>inline ElectronUrbanLoss sample_electron_urban_loss(
    ElectronUrbanMaterial p,double energy,double mean,double cut,Uniform& uniform) {
    ElectronUrbanLoss out;
    if(!std::isfinite(energy)||energy<=0||!std::isfinite(mean)||mean<0||!std::isfinite(cut)||cut<=0||
       !std::isfinite(p.mass)||p.mass<=0||!std::isfinite(p.excitation)||p.excitation<=0||
       !std::isfinite(p.e0)||p.e0<=0||!std::isfinite(p.e1)||p.e1<=0||!std::isfinite(p.e2)||p.e2<=0||
       !std::isfinite(p.f1)||p.f1<0||!std::isfinite(p.f2)||p.f2<0)return out;
    if(mean<1e-5||cut<=p.e0)return {true,mean,0};
    bool good=true;
    auto u=[&](){if(++out.random_draws>16384){good=false;return .5;}
        const double x=uniform();if(!std::isfinite(x)||x<=0||x>=1){good=false;return .5;}return x;};
    auto poisson=[&](double mu){if(!std::isfinite(mu)||mu<0||mu>8.00001){good=false;return 0;}
        const double stop=std::exp(-mu);double product=1;int n=0;
        do{++n;product*=u();}while(product>stop&&good);return n-1;};
    auto normal=[&](){const double a=u(),b=u();return std::sqrt(-2*std::log(a))*std::cos(6.2831853071795864769*b);};
    auto gauss=[&](double m,double variance){const double sigma=std::sqrt(variance);if(m<.25*sigma)return 2*u()*m;
        double x;do{x=m+sigma*normal();}while((x<0||x>2*m)&&good);return x;};
    const double gamma=1+energy/p.mass,beta2=energy*(energy+2*p.mass)/((energy+p.mass)*(energy+p.mass));
    const double scaling=std::min(1+.0005/cut,1.5),adjusted=mean/scaling;
    const double w2=cut>p.excitation?std::log(2*p.mass*beta2*gamma*gamma)-beta2:0;
    double a1=0,a2=0,e1=p.e1,e2=p.e2;
    if(cut>p.excitation&&w2>std::log(p.excitation)){
        if(w2>std::log(e2)){
            const double c=adjusted*.44/(w2-std::log(p.excitation));
            a1=c*p.f1*(w2-std::log(e1))/e1;a2=c*p.f2*(w2-std::log(e2))/e2;
        }else a1=adjusted*.44/e1;
        const double width=a1<42?.5+3.5*std::sqrt(a1/42):4;a1/=width;e1*=width;
    }
    const double w1=cut/p.e0;
    double a3=.56*adjusted*(cut-p.e0)/(p.e0*cut*std::log(w1));if(a1+a2<=0)a3/=.56;
    double loss=0,m=0,variance=0;
    auto excitation=[&](double a,double e){if(a>8){m+=a*e;variance+=a*e*e;}
        else if(a>0){const int n=poisson(a);if(n>0)loss+=(n+1-2*u())*e;}};
    excitation(a1,e1);excitation(a2,e2);if(variance>0)loss+=gauss(m,variance);
    if(a3>0){
        double alpha=1,p3=a3;m=variance=0;
        if(a3>8){
            alpha=w1*(8+a3)/(8*w1+a3);const double alpha1=alpha*std::log(alpha)/(alpha-1);
            // Stable algebraic form of native namean and a3-namean.
            const double count=a3*(a3/(8+a3));p3=8*(a3/(8+a3));
            m=count*p.e0*alpha1;variance=p.e0*p.e0*count*(alpha-alpha1*alpha1);
        }
        const double low=alpha*p.e0;
        if(cut>low){const double fraction=(cut-low)/cut;const int n=poisson(p3);for(int i=0;i<n&&good;++i)loss+=low/(1-fraction*u());}
        if(variance>0)loss+=gauss(m,variance);
    }
    out.loss=loss*scaling;out.valid=good&&std::isfinite(out.loss)&&out.loss>=0;return out;
}
} // namespace carbon
