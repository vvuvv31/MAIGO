#pragma once
#include "carbon/electron_penelope_composition.hpp"
#include "carbon/electron_atomic_cascade26.hpp"
#include "carbon/electron_brems26.hpp"
#include "carbon/electron_continuous_scalar.hpp"
namespace carbon::electron_composition26 {
struct CollisionComposition26 {
    ElectronContinuousScalarView scalar{};
    PenelopeComposition penelope{};
    AtomicBank26 atomic{};
    BremsSpectrum26 brems{};
    ElectronCutBirthComposition shells{};
};
struct CompositionCollision26 {bool valid{};int pdg{};float energy{},cosine{};};
inline const AtomicVacancy26* atomic_shell_index26(AtomicBank26 bank,unsigned z,unsigned index) {
    unsigned low=0,high=bank.count;
    while(low<high){const auto mid=low+(high-low)/2;if(bank.vacancies[mid].z<z)low=mid+1;else high=mid;}
    const auto ordinal=low+index;if(ordinal>=bank.count||bank.vacancies[ordinal].z!=z)return nullptr;
    return bank.vacancies+ordinal;
}
// Emit the complete correlated hard event. The packet wrapper below chooses
// one energy-weighted continuation without discarding local/atomic branches.
template<class Uniform,class Emit> inline bool sample_collision_event26(
    const CollisionComposition26& v,float energy,float electron_cut,float photon_cut,unsigned process,Uniform& uniform,Emit& emit) {
    if(!(energy>0.f&&electron_cut>0.f&&photon_cut>0.f)||process>1)return false;
    if(process==1){
        const auto pair=sample_brems_pair26(v.brems,energy,photon_cut,uniform);if(!pair.valid)return false;
        emit(11,pair.primary_energy,pair.primary_cosine);emit(22,pair.photon_energy,pair.photon_cosine);return true;
    }
    if(energy>.1f){
        if(energy<=2.f*electron_cut){emit(11,energy,1.f);return true;}
        const auto pair=sample_moller_pair(v.shells,energy,electron_cut,uniform);if(!pair.valid)return false;
        emit(11,pair.primary_energy,pair.primary_cosine);emit(11,pair.secondary_energy,pair.secondary_cosine);return true;
    }
    const auto event=sample_penelope_event(v.penelope,energy,electron_cut,uniform);if(!event.valid)return false;
    emit(11,event.primary_energy,event.primary_cosine);emit(11,event.secondary_energy,event.secondary_cosine);
    float local=event.local;
    if(event.relaxation_energy>0.f){
        const auto o=v.penelope.oscillators[event.oscillator];
        const auto* vacancy=o.z&&o.shell_flag&&o.shell_flag<30?atomic_shell_index26(v.atomic,o.z,o.shell_flag-1):nullptr;
        if(!vacancy)return false;
        const auto cascade=sample_atomic_cascade26(v.atomic,o.z,vacancy->id,event.relaxation_energy,electron_cut,photon_cut,uniform,emit);
        if(!cascade.valid)return false;local+=cascade.local;
    }
    emit(0,local,1.f);return true;
}
template<class Uniform> inline CompositionCollision26 sample_collision_composition26(
    const CollisionComposition26& v,float energy,float electron_cut,float photon_cut,unsigned process,Uniform& uniform) {
    CompositionCollision26 selected{};float total=0.f;
    auto emit=[&](int pdg,float e,float cosine){
        if(!(e>0.f))return;
        const float next=total+e;
        if(total==0.f||uniform()<e/next)selected={true,pdg,e,cosine};total=next;
    };
    if(!sample_collision_event26(v,energy,electron_cut,photon_cut,process,uniform,emit)||
       !std::isfinite(total)||std::abs(total-energy)>energy*1e-6f)return {};
    return selected;
}
inline float continuous_inverse_range26(ElectronContinuousScalarView v,float density,float range) {
    if(!std::isfinite(range)||!v.energy||v.energies<2)return NAN;
    auto r=[&](unsigned i){return electron_continuous_scalar(v,density,v.energy[i]).range;};
    unsigned low=0,high=v.energies-1;if(range<r(low)||range>r(high))return NAN;
    while(low+1<high){const auto mid=low+(high-low)/2;if(r(mid)>range)high=mid;else low=mid;}
    const float a=r(low),b=r(high);if(!(b>a))return NAN;
    return v.energy[low]+(range-a)/(b-a)*(v.energy[high]-v.energy[low]);
}
}
