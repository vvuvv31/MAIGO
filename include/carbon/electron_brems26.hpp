#pragma once
#include <algorithm>
#include <cmath>
#include "carbon/electron_composition_kinematics.hpp"
namespace carbon::electron_composition26 {
struct BremsSpectrum26 {const float* energies{};const float* fractions{};const float* integral{};unsigned energy_count{},fraction_count{};const unsigned* element_z{};const float* element_integral{};unsigned element_count{};};
inline float brems_integral26(BremsSpectrum26 v,float energy,float fraction) {
    if(!v.energies||!v.fractions||!v.integral||v.energy_count<2||v.fraction_count<2||energy<v.energies[0]||energy>v.energies[v.energy_count-1]||fraction<v.fractions[0]||fraction>1.f)return NAN;
    unsigned a=0,b=v.energy_count-1;
    while(a+1<b){const auto m=a+(b-a)/2;if(v.energies[m]>energy)b=m;else a=m;}
    const float f=std::clamp(std::log(energy/v.energies[a])/std::log(v.energies[b]/v.energies[a]),0.f,1.f);
    unsigned x=0,y=v.fraction_count-1;
    while(x+1<y){const auto m=x+(y-x)/2;if(v.fractions[m]>fraction)y=m;else x=m;}
    const float t=std::clamp(std::log(fraction/v.fractions[x])/std::log(v.fractions[y]/v.fractions[x]),0.f,1.f);
    auto value=[&](unsigned j){const float u=v.integral[a*v.fraction_count+j],w=v.integral[b*v.fraction_count+j];return u+f*(w-u);};
    const float u=value(x),w=value(y);return std::max(0.f,u+t*(w-u));
}
// TOPAS opt4 uses G4Generator2BS, not the Seltzer-Berger constructor's
// ModifiedTsai default. The native algorithm conditions the angle on both
// emitted energy and the sampled target element (PIRS-0203; Koch & Motz).
template<class Uniform> inline float sample_2bs_cosine26(float energy,float photon,unsigned z,Uniform& uniform) {
    if(!(energy>0.f&&photon>0.f&&photon<=energy)||!z||z>104)return NAN;
    constexpr float mass=.51099895f;
    const float ratio=1.f-photon/(energy+mass),ratio1=(1.f+ratio)*(1.f+ratio),ratio2=1.f+ratio*ratio;
    const float gamma=1.f+energy/mass,beta=std::sqrt(energy*(energy+2.f*mass))/(energy+mass);
    const float fz=.00008116224f*std::cbrt(float(z))*std::cbrt(float(z+1));
    const float ymax=2.f*beta*(1.f+beta)*gamma*gamma;
    auto rejection=[&](float y){const float y2=(1.f+y)*(1.f+y),x=4.f*y*ratio/y2;return 4.f*x-ratio1-(ratio2-x)*std::log(fz/y2);};
    const float maximum=std::max(rejection(0.f),rejection(ymax));
    if(!(maximum>0.f)||!std::isfinite(maximum))return NAN;
    for(unsigned retry=0;retry<16384;++retry){const float q=uniform(),y=q*ymax/(1.f+ymax*(1.f-q));
        if(float(uniform())*maximum<=rejection(y)&&y<=ymax)return std::clamp(1.f-2.f*y/ymax,-1.f,1.f);}
    return NAN;
}
struct BremsPair26 {bool valid{};float primary_energy{},primary_cosine{},photon_energy{},photon_cosine{};};
template<class Uniform> inline BremsPair26 sample_brems_pair26(BremsSpectrum26 v,float energy,float cut,Uniform& uniform) {
    BremsPair26 out;if(!(energy>cut&&cut>0.f))return out;
    const float low=cut/energy;
    // Native target selection uses its integrated rate at the actual cut.
    if(!v.element_z||!v.element_integral||!v.element_count)return out;
    const auto stride=v.energy_count*v.fraction_count;float total=0.f;
    for(unsigned i=0;i<v.element_count;++i){auto element=v;element.integral=v.element_integral+i*stride;const float rate=brems_integral26(element,energy,low);if(!std::isfinite(rate))return out;total+=rate;}
    if(!(total>0.f))return out;
    const float choice=total*float(uniform());float cumulative=0.f;unsigned selected=v.element_count-1;
    for(unsigned i=0;i+1<v.element_count;++i){auto element=v;element.integral=v.element_integral+i*stride;cumulative+=brems_integral26(element,energy,low);if(choice<cumulative){selected=i;break;}}
    const unsigned z=v.element_z[selected];v.integral=v.element_integral+selected*stride;
    const float maximum=brems_integral26(v,energy,low);
    if(!(maximum>0.f)||!std::isfinite(maximum))return out;
    const float target=maximum*(1.f-float(uniform()));
    // Closed inversion of the same monotone piecewise log-fraction response.
    unsigned a=0,b=v.energy_count-1;
    while(a+1<b){const auto m=a+(b-a)/2;if(v.energies[m]>energy)b=m;else a=m;}
    const float f=std::clamp(std::log(energy/v.energies[a])/std::log(v.energies[b]/v.energies[a]),0.f,1.f);
    auto value=[&](unsigned j){const float u=v.integral[a*v.fraction_count+j],w=v.integral[b*v.fraction_count+j];return u+f*(w-u);};
    unsigned x=0,y=v.fraction_count-1;
    while(x+1<y){const auto m=x+(y-x)/2;if(value(m)>target)x=m;else y=m;}
    const float u=value(x),w=value(y);if(!(u>w))return {};
    const float t=std::clamp((target-u)/(w-u),0.f,1.f);
    const float fraction=std::clamp(v.fractions[x]*std::exp(t*std::log(v.fractions[y]/v.fractions[x])),low,1.f);
    out.photon_energy=std::max(cut,energy*fraction);out.primary_energy=energy-out.photon_energy;
    out.photon_cosine=sample_2bs_cosine26(energy,out.photon_energy,z,uniform);
    out.primary_cosine=brems_primary_cosine(energy,out.photon_energy,out.photon_cosine);
    out.valid=out.primary_energy>=0.f&&out.photon_energy>=cut&&std::isfinite(out.primary_cosine)&&std::isfinite(out.photon_cosine);return out;
}
}
