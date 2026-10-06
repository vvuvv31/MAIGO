#pragma once
#include <algorithm>
#include <cmath>
#include "carbon/electron_cut_birth.hpp"

namespace carbon::electron_composition26 {
// Geant4 11.3.2 G4MollerBhabhaModel, electron branch. All runtime
// arithmetic is FP32. No density node, energy-bin centroid or independent
// angle draw is involved. The caller supplies the actual production cut.
struct MollerPair {
    bool valid{};
    float primary_energy{},primary_cosine{},secondary_energy{},secondary_cosine{};
};
inline float moller_cosine(float incoming,float outgoing) {
    constexpr float twice_mass=1.0219979f;
    return std::sqrt(std::clamp(outgoing*(incoming+twice_mass)/
        (incoming*(outgoing+twice_mass)),0.f,1.f));
}
template<class Uniform> inline MollerPair sample_moller_pair(float energy,float cut,Uniform& uniform) {
    if(!std::isfinite(energy)||!std::isfinite(cut)||!(cut>0.f)||!(energy>2.f*cut))return {};
    constexpr float mass=.51099895f;
    const float xmin=cut/energy,xmax=.5f,gamma=1.f+energy/mass;
    const float gg=(2.f*gamma-1.f)/(gamma*gamma);
    auto rejection=[&](float x){const float y=1.f-x;
        return 1.f-gg*x+x*x*(1.f-gg+(1.f-gg*y)/(y*y));};
    const float majorant=rejection(xmax);
    // A malformed RNG or arithmetic failure is an explicit error, not a
    // silently truncated hard event. Normal rejection efficiency is high.
    for(unsigned retry=0;retry<16384;++retry){
        const float u=uniform(),v=uniform();
        if(!(u>0.f&&u<1.f&&v>0.f&&v<1.f))return {};
        const float x=xmin*xmax/(xmin*(1.f-u)+xmax*u);
        if(majorant*v>rejection(x))continue;
        const float secondary=x*energy,primary=energy-secondary;
        return {true,primary,moller_cosine(energy,primary),secondary,moller_cosine(energy,secondary)};
    }
    return {};
}
// TOPAS opt4 enables the G4DeltaAngle bound-electron generator. The free
// electron cosine above is therefore NOT the production angular law.
template<class Uniform> inline float sample_electron_delta_cosine(
    ElectronCutBirthComposition composition,float energy,float transfer,Uniform& uniform) {
    if(!composition.shells||!composition.count||!(energy>0.f&&transfer>0.f&&transfer<=energy*.5f))return NAN;
    const auto draw=uniform();unsigned low=0,high=composition.count-1;
    while(low<high){const auto middle=low+(high-low)/2;
        if(draw.bits<composition.shells[middle].cdf)high=middle;else low=middle+1;}
    float binding=composition.shells[low].binding;
    constexpr float mass=.51099895f;
    for(unsigned attempt=1;attempt<=101;++attempt){
        const float x=-std::log(float(uniform()));
        const float kinetic=binding*x,potential=binding*(1.f+x);
        const float outgoing=transfer+potential;
        const float p=std::sqrt(outgoing*(outgoing+2.f*mass));
        const float incoming=energy+potential,total=mass+incoming;
        const float momentum=std::sqrt(incoming*(incoming+2.f*mass));
        const float atomic_momentum=std::sqrt(kinetic*(kinetic+2.f*mass));
        const float c=2.f*float(uniform())-1.f,s=std::sqrt(std::max(0.f,(1.f-c)*(1.f+c)));
        bool accepted=false;float cosine=1.f;
        if(attempt>=100){if(binding==0.f)accepted=true;binding=0.f;}
        const float x0=p*(momentum+atomic_momentum*c);
        if(x0>0.f){
            const float x1=p*atomic_momentum*s;
            // Algebraic cancellation of rest-energy products avoids subtracting
            // O(m^2) terms to recover a small O(m*T) FP32 result.
            const float x2=total*(kinetic-outgoing)-mass*(outgoing+kinetic)
                -outgoing*kinetic-momentum*atomic_momentum*c;
            const float y=-x2/x0;
            if(std::abs(y)<=1.f){
                cosine=-(x2+x1*std::sqrt(std::max(0.f,(1.f-y)*(1.f+y))))/x0;
                if(std::abs(cosine)<=1.f)accepted=true;else cosine=1.f;
            }
        }
        if(accepted)return cosine;
    }
    return NAN;
}
template<class Uniform> inline MollerPair sample_moller_pair(
    ElectronCutBirthComposition composition,float energy,float cut,Uniform& uniform) {
    // Keep the integer shell draw intact. Only the energy proposal needs a
    // floating variate; the shared26 shell law uses all64 random bits.
    auto floating=[&](){return float(uniform());};
    auto pair=sample_moller_pair(energy,cut,floating);if(!pair.valid)return {};
    pair.secondary_cosine=sample_electron_delta_cosine(composition,energy,pair.secondary_energy,uniform);
    const float momentum=std::sqrt(energy*(energy+1.0219979f));
    const float child_momentum=std::sqrt(pair.secondary_energy*(pair.secondary_energy+1.0219979f));
    const float z=momentum-child_momentum*pair.secondary_cosine;
    const float transverse2=child_momentum*child_momentum*(1.f-pair.secondary_cosine)*(1.f+pair.secondary_cosine);
    pair.primary_cosine=z/std::sqrt(z*z+transverse2);
    pair.valid=std::isfinite(pair.secondary_cosine)&&std::isfinite(pair.primary_cosine);
    return pair;
}
// Modified Tsai universal photon angle in G4ModifiedTsai. The recoil
// electron direction must be computed from this SAME photon angle; sampling
// the two marginal angles independently would discard the correlation.
template<class Uniform> inline float sample_modified_tsai_cosine(float energy,Uniform& uniform) {
    if(!std::isfinite(energy)||!(energy>0.f))return NAN;
    const float maximum=2.f*(1.f+energy/.51099895f);
    for(unsigned retry=0;retry<16384;++retry){
        const float a=uniform(),b=uniform(),c=uniform();
        if(!(a>0.f&&a<1.f&&b>0.f&&b<1.f&&c>0.f&&c<1.f))return NAN;
        const float u=-std::log(a*b)*(c<.25f?1.6f:1.6f/3.f);
        if(u<=maximum)return std::clamp(1.f-2.f*(u/maximum)*(u/maximum),-1.f,1.f);
    }
    return NAN;
}
inline float brems_primary_cosine(float energy,float photon_energy,float photon_cosine) {
    if(!(energy>0.f&&photon_energy>=0.f&&photon_energy<=energy)||
       !std::isfinite(photon_cosine)||std::abs(photon_cosine)>1.f)return NAN;
    const float momentum=std::sqrt(energy*(energy+1.0219979f));
    const float z=momentum-photon_energy*photon_cosine;
    const float transverse2=photon_energy*photon_energy*(1.f-photon_cosine)*(1.f+photon_cosine);
    return z/std::sqrt(z*z+transverse2);
}
} // namespace carbon::electron_composition26
