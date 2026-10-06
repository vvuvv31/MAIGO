#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace carbon {
// A composition has one shell law. Density changes the production cut, not
// this electron-number-weighted atomic mixture. No density response states.
struct ElectronBirthShell {std::uint64_t cdf{};float binding{};std::uint32_t z{};};
struct ElectronCutBirthComposition {
    const ElectronBirthShell* shells{};unsigned count{};float density_low{},density_high{};
};
struct ElectronCutBirthState {bool valid{};float energy{},cosine{};unsigned attempts{};};
struct ElectronProjectileBirthParameters {
    unsigned z{},a{};float mass{},bragg_switch{},secondary_floor{},form_factor{},magnetic_correction{};bool spin{};
};
inline ElectronProjectileBirthParameters electron_birth_projectile_parameters(unsigned a) {
    if(a!=1&&a!=12)return {};
    return a==1?ElectronProjectileBirthParameters{1,1,938.272013f,2.f,.00025f,1.439482e-6f,6.79943f,true}:
        ElectronProjectileBirthParameters{6,12,11174.863388f/12.f,0.f,0.f,5.51029e-6f,0.f,false};
}
inline float electron_birth_maximum(float parent,ElectronProjectileBirthParameters parameters) {
    const unsigned a=parameters.a;const float mass=parameters.mass;
    if((a!=1&&a!=12)||!(mass>0.f)||!std::isfinite(mass))return NAN;
    const float tau=parent/mass,ratio=.51099891f/(a*mass);
    return 2.f*.51099891f*tau*(tau+2.f)/(1.f+2.f*(tau+1.f)*ratio+ratio*ratio);
}
template<class Uniform> inline ElectronCutBirthState sample_cut_electron_birth(
    ElectronCutBirthComposition composition,float parent,ElectronProjectileBirthParameters parameters,float cut,Uniform& uniform) {
    ElectronCutBirthState out;
    const unsigned a=parameters.a;
    if((a!=1&&a!=12)||!(parameters.mass>0.f)||!std::isfinite(parameters.mass)||!composition.shells||!composition.count||
       !std::isfinite(parent)||parent<=0.f||!std::isfinite(cut)||cut<=0.f)return out;
    constexpr float me=.51099891f;
    const float mass=parameters.mass;
    const float kinetic=a*parent,total=kinetic+a*mass;
    const float momentum=a*std::sqrt(parent*(parent+2.f*mass));
    const float beta2=kinetic*(kinetic+2.f*a*mass)/(total*total);
    // TOPAS opt4 uses LindhardSorensen for GenericIon over the whole
    // energy domain. The proton's ICRU73QO inherits Bragg sampling below2MeV.
    const bool bragg=parameters.bragg_switch>0.f&&kinetic<=parameters.bragg_switch;
    // G4BraggModel's secondary floor is lowestKinEnergy * massRate.
    cut=bragg?std::max(cut,parameters.secondary_floor):cut;
    const float maximum=electron_birth_maximum(parent,parameters);
    if(!std::isfinite(maximum)||maximum<=cut)return out;
    const float width=std::log(maximum/cut);
    const bool spin=parameters.spin&&!bragg;
    const float spin_max=spin?maximum*maximum/(2.f*total*total):0.f;
    float transfer=0.f;
    for(unsigned attempt=1;attempt<=128;++attempt){
        out.attempts=attempt;
        const float u=uniform(),v=uniform();
        if(!(u>0.f&&u<1.f&&v>0.f&&v<1.f))return out;
        const float t=cut*std::exp(u*width);
        const float f=1.f-beta2*t/maximum+(spin?t*t/(2.f*total*total):0.f);
        if(v*(1.f+spin_max)>f)continue;
        // Same accepted-event form-factor correction as native Bethe/LS.
        const float x=parameters.form_factor*t;
        if(!bragg&&x>1e-6f){
            float accept=1.f/((1.f+x)*(1.f+x));
            if(spin){const float x2=.5f*me*t/(mass*mass),f1=.5f*t*t/(total*total);
                accept*=1.f+parameters.magnetic_correction*(x2-f1/f)/(1.f+x2);}
            if(float(uniform())>accept)continue;
        }
        transfer=t;break;
    }
    if(!(transfer>0.f))return out;
    // Flatten independent element and conditional shell choices into one CDF:
    // atom number density * Z, then shell occupation / binding energy.
    const auto draw=uniform();
    unsigned lo=0,hi=composition.count-1;
    while(lo<hi){const auto mid=lo+(hi-lo)/2;
        if(draw.bits<composition.shells[mid].cdf)hi=mid;else lo=mid+1;}
    float binding=composition.shells[lo].binding,cost=1.f;
    bool accepted=false;
    for(unsigned attempt=1;attempt<=101;++attempt){
        const float x=-std::log(float(uniform()));
        const float atomic_kinetic=binding*x,potential=binding*(1.f+x);
        const float e=transfer+potential+me;
        const float p=std::sqrt((e+me)*(e-me));
        const float atomic_total=atomic_kinetic+me;
        const float atomic_momentum=std::sqrt(atomic_kinetic*(atomic_total+me));
        const float c=2.f*float(uniform())-1.f,s=std::sqrt(std::max(0.f,1.f-c*c));
        if(attempt>=100){if(binding==0.f)accepted=true;binding=0.f;}
        const float x0=p*(momentum+atomic_momentum*c);
        if(x0>0.f){
            const float x1=p*atomic_momentum*s;
            // atomic_total-e = -transfer-binding, evaluated without losing
            // precision by subtracting two electron rest energies.
            const float x2=-total*(transfer+(potential-atomic_kinetic))-e*atomic_total
                -momentum*atomic_momentum*c+me*me;
            const float y=-x2/x0;
            if(std::abs(y)<=1.f){
                cost=-(x2+x1*std::sqrt(std::max(0.f,1.f-y*y)))/x0;
                if(std::abs(cost)<=1.f)accepted=true;else cost=1.f;
            }
        }
        if(accepted)break;
    }
    if(!accepted||!std::isfinite(cost))return out;
    out.valid=true;out.energy=transfer;out.cosine=cost;return out;
}
inline float electron_birth_maximum(float parent,unsigned a){return electron_birth_maximum(parent,electron_birth_projectile_parameters(a));}
template<class Uniform>inline ElectronCutBirthState sample_cut_electron_birth(
    ElectronCutBirthComposition composition,float parent,unsigned a,float cut,Uniform& uniform){
    return sample_cut_electron_birth(composition,parent,electron_birth_projectile_parameters(a),cut,uniform);
}
} // namespace carbon
