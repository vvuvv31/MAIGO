#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
namespace carbon {
// Native Geant4 11.3.2 GS single-elastic law, including its Mott energy
// mixture and angular interpolation. See LICENSES/Geant4.txt and
// G4GoudsmitSaunderson{Table,MscModel}/G4GSMottCorrection.cc.
// This is a collision primitive, not a complete charged transport model.
// Each energy row: 3 correction factors, 31 linear-angle knots, 4 equally
// spaced values representing the final cubic interval exactly.
struct ElectronElasticView {const double* rows{};double mass{},bc{},xc2{};};
struct ElasticEnergyBracket {unsigned low{};double high_weight{};};
inline ElasticEnergyBracket elastic_energy_bracket(double energy,double mass) {
    const double p2=energy*(energy+2*mass),beta=p2/(p2+mass*mass);
    const double midp=.1*(.1+2*mass),midbeta=midp/(midp+mass*mass);
    if(beta>=.9999)return {30,0};
    double x=0;
    if(beta>=midbeta)x=15+15*(beta-midbeta)/(.9999-midbeta);
    else if(energy>.001)x=15*(std::log(energy)-std::log(.001))/std::log(100.);
    const auto lo=static_cast<unsigned>(x);return {lo,x-lo};
}
inline double elastic_screening(ElectronElasticView v,double energy) {
    if(!v.rows||!std::isfinite(energy)||energy<1e-5||energy>.65||v.mass<=0||v.bc<=0||v.xc2<=0)return -1;
    const auto b=elastic_energy_bracket(energy,v.mass);const double a=v.rows[38*b.low];
    const double factor=a+(b.high_weight>0?b.high_weight*(v.rows[38*(b.low+1)]-a):0);
    return v.xc2/(4*energy*(energy+2*v.mass)*v.bc)*factor;
}
inline double elastic_mott_rejection(ElectronElasticView v,unsigned energy_index,double cosine) {
    if(!v.rows||energy_index>=31||!std::isfinite(cosine)||cosine< -1||cosine>1)return -1;
    const double y=std::sqrt(.5*(1-cosine)),x=y*31;
    const unsigned i=static_cast<unsigned>(x);const auto* row=v.rows+38*energy_index+3;
    if(i<30)return row[i]+(x-i)*(row[i+1]-row[i]);
    const double t=(y-30./31)*31;
    const auto* c=row+31;
    // Cubic Lagrange interpolation at t=0,1/3,2/3,1. It reconstructs
    // the native final interval polynomial without a fitted angular tail.
    return -4.5*(t-1./3)*(t-2./3)*(t-1)*c[0]
        +13.5*t*(t-2./3)*(t-1)*c[1]
        -13.5*t*(t-1./3)*(t-1)*c[2]+4.5*t*(t-1./3)*(t-2./3)*c[3];
}
struct ElectronElasticDraw {bool valid{};double cosine{};unsigned attempts{},energy_index{};};
template<class Uniform>inline ElectronElasticDraw sample_electron_elastic(ElectronElasticView view,double energy,Uniform& uniform) {
    ElectronElasticDraw out;const double a=elastic_screening(view,energy);if(!(a>0))return out;
    auto proposal=[&](double u){return 1-2*a*u/(1-u+a);};
    const double u=uniform();if(!std::isfinite(u)||u<0||u>=1)return out;
    out.cosine=proposal(u);
    const auto b=elastic_energy_bracket(energy,view.mass);
    const double choose=uniform();if(!std::isfinite(choose)||choose<0||choose>=1)return out;
    out.energy_index=b.low+(choose<b.high_weight);
    for(unsigned i=0;i<1000;++i){
        ++out.attempts;const double acceptance=elastic_mott_rejection(view,out.energy_index,out.cosine);
        const double coin=uniform();if(!std::isfinite(acceptance)||!std::isfinite(coin)||coin<0||coin>=1)return out;
        if(coin<=acceptance){out.valid=true;return out;}
        if(i+1<1000){const double next=uniform();if(!std::isfinite(next)||next<0||next>=1)return out;out.cosine=proposal(next);}
    }
    return out; // reject exhaustion rather than accepting a rejected angle
}
} // namespace carbon
