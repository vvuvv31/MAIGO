#pragma once
#include <algorithm>
#include <cmath>

namespace carbon::electron_composition26 {
struct PenelopeOscillator {
    float strength{},ionization{},resonance{},recoil_cut{},binding{};
    unsigned z{},shell_flag{};
};
struct PenelopeComposition {
    const PenelopeOscillator* oscillators{};unsigned count{};
    // Native density-correction function on the fixed log-energy grid;
    // these are scalar coefficients, not material-density response states.
    const float* delta{};unsigned delta_count{};
};
struct PenelopeHardChannels {float close{},longitudinal{},transverse{},minimum_recoil{};};
inline float penelope_density_correction(PenelopeComposition v,float energy) {
    if(!v.delta||v.delta_count<2||!(energy>0.f))return NAN;
    const float position=(std::log(energy)-std::log(1e-4f))*(99.f/std::log(1e9f));
    const unsigned i=std::min(unsigned(std::max(0.f,position)),v.delta_count-2);
    const float t=std::clamp(position-i,0.f,1.f);
    return v.delta[i]+t*(v.delta[i+1]-v.delta[i]);
}
inline PenelopeHardChannels penelope_hard_channels(PenelopeOscillator o,float energy,float cut,float delta,bool sampler) {
    PenelopeHardChannels out;constexpr float mass=.51099895f;
    if(!(energy>0.f&&cut>0.f&&o.resonance>0.f&&o.recoil_cut>0.f))return out;
    const float gamma=1.f+energy/mass,beta=energy*(energy+2.f*mass)/((energy+mass)*(energy+mass));
    const float amol=(energy/(energy+mass))*(energy/(energy+mass));
    out.minimum_recoil=o.recoil_cut;
    // The XS builder admits equality cut==resonance; EINa sampling uses >.
    // Retain this native distinction, including native null collisions.
    if(energy>o.resonance&&(sampler?o.resonance>cut:o.resonance>=cut)){
        const float p=std::sqrt(energy*(energy+2.f*mass));
        const float remaining=energy-o.resonance;
        const float pp=std::sqrt(remaining*(remaining+2.f*mass));
        const float difference=o.resonance*(2.f*energy+2.f*mass-o.resonance)/(p+pp);
        float qm=difference*difference/(std::sqrt(difference*difference+mass*mass)+mass);
        if(o.resonance<=1e-6f*energy){qm=o.resonance*o.resonance/(beta*2.f*mass);qm*=1.f-qm*.5f/mass;}
        if(qm<o.recoil_cut){
            out.minimum_recoil=qm;
            out.longitudinal=std::log(o.recoil_cut*(qm+2.f*mass)/(qm*(o.recoil_cut+2.f*mass)))/o.resonance;
            out.transverse=std::max(std::log(gamma*gamma)-beta-delta,0.f)/o.resonance;
        }
    }
    const float ee=energy+o.ionization,w=std::max(cut,o.recoil_cut),r=w/ee;
    if(w<.5f*ee-(sampler?0.f:1e-11f)){
        const float d=.5f-r;
        out.close=std::max(0.f,(amol*d+2.f*d/(r*(1.f-r))+
            (1.f-amol)*std::log1p(-2.f*d/(1.f-r)))/ee);
    }
    return out;
}
inline float penelope_shell_weight(PenelopeOscillator o,float energy,float cut,float delta) {
    const auto h=penelope_hard_channels(o,energy,cut,delta,false);
    const float beta=energy*(energy+1.0219979f)/((energy+.51099895f)*(energy+.51099895f));
    // pi*r_e^2*2m in mm^2 MeV; divide the native1e-40mm^2
    // log floor algebraically, so GPU flush-to-zero cannot erase it.
    constexpr float prefactor=2.5493841e-23f;
    const float floor=(1e-20f/prefactor)*beta*1e-20f;
    return std::max(floor,energy>=o.ionization?o.strength*(h.close+h.longitudinal+h.transverse):0.f);
}
inline float penelope_grid_energy(unsigned i) {
    return 1e-4f*std::exp(i*(std::log(1e9f)/99.f));
}
template<class Uniform> inline int sample_penelope_oscillator(PenelopeComposition v,float energy,float cut,Uniform& uniform) {
    if(!v.oscillators||!v.count||v.count>128||!v.delta||v.delta_count<2||!(energy>=1e-4f&&energy<=.100001f&&cut>0.f))return -1;
    const float position=std::clamp((std::log(energy)-std::log(1e-4f))*(99.f/std::log(1e9f)),0.f,33.f);
    const unsigned lower=std::min(unsigned(position),32u);const float t=position-lower;
    const float e0=penelope_grid_energy(lower),e1=penelope_grid_energy(lower+1);
    const float d0=penelope_density_correction(v,e0),d1=penelope_density_correction(v,e1);
    float sum0=0.f,sum1=0.f;
    for(unsigned i=0;i<v.count;++i){sum0+=penelope_shell_weight(v.oscillators[i],e0,cut,d0);sum1+=penelope_shell_weight(v.oscillators[i],e1,cut,d1);}
    if(!(sum0>0.f&&sum1>0.f))return -1;
    const float draw=uniform();float accumulated=0.f;
    for(unsigned i=0;i+1<v.count;++i){
        const float a=penelope_shell_weight(v.oscillators[i],e0,cut,d0)/sum0;
        const float b=penelope_shell_weight(v.oscillators[i],e1,cut,d1)/sum1;
        accumulated+=std::exp((1.f-t)*std::log(a)+t*std::log(b));
        if(accumulated>draw)return int(i);
    }
    return int(v.count-1); // Native interpolation residual; do not normalize.
}
struct PenelopeEvent {
    bool valid{},null{};unsigned oscillator{};
    float primary_energy{},primary_cosine{1.f},secondary_energy{},secondary_cosine{1.f};
    // Must subsequently sample atomic relaxation at actual cuts. This is a
    // budget, NOT an instruction to score all binding energy locally.
    float relaxation_energy{},local{};
};
template<class Uniform> inline PenelopeEvent sample_penelope_event(PenelopeComposition v,float energy,float cut,Uniform& uniform) {
    PenelopeEvent out;
    const int index=sample_penelope_oscillator(v,energy,cut,uniform);if(index<0)return out;
    out.valid=true;out.oscillator=unsigned(index);out.primary_energy=energy;
    const auto o=v.oscillators[index];
    const auto h=penelope_hard_channels(o,energy,cut,penelope_density_correction(v,energy),true);
    const float total=h.close+h.longitudinal+h.transverse;
    if(total<1e-36f){out.null=true;return out;}
    constexpr float mass=.51099895f;const float rb=energy+2.f*mass;
    const float choice=total*float(uniform());float transfer=0.f,secondary_cosine=1.f;
    if(choice<h.close){
        const float ee=energy+o.ionization,rcl=std::max(cut,o.recoil_cut)/ee;
        const float amol=(energy/(energy+mass))*(energy/(energy+mass)),a=5.f*amol,arcl=a*.5f*rcl;
        bool accepted=false;float r=0.f;
        for(unsigned retry=0;retry<16384;++retry){
            const float fb=(1.f+arcl)*float(uniform());
            r=fb<1.f?rcl/(1.f-fb*(1.f-2.f*rcl)):rcl+(fb-1.f)*(.5f-rcl)/arcl;
            const float rf=r/(1.f-r),r2=r*r,phi=1.f+rf*rf-rf+amol*(r2+rf);
            if(float(uniform())*(1.f+a*r2)<=phi){accepted=true;break;}
        }
        if(!accepted)return {};
        transfer=r*ee;out.primary_energy=energy-transfer;
        out.primary_cosine=std::sqrt(std::clamp(out.primary_energy*rb/(energy*(rb-transfer)),0.f,1.f));
        secondary_cosine=std::sqrt(std::clamp(transfer*rb/(energy*(transfer+2.f*mass)),0.f,1.f));
    }else{
        transfer=o.resonance;out.primary_energy=energy-transfer;
        if(choice<h.close+h.longitudinal){
            const float qm=h.minimum_recoil,qs=qm/(1.f+qm*.5f/mass);
            const float q=qs/(std::pow((qs/o.recoil_cut)*(1.f+o.recoil_cut*.5f/mass),float(uniform()))-qs*.5f/mass);
            const float q2=q*(q+2.f*mass),p2=energy*rb,pp2=out.primary_energy*(out.primary_energy+2.f*mass);
            out.primary_cosine=std::clamp((pp2+p2-q2)/(2.f*std::sqrt(p2*pp2)),-1.f,1.f);
            secondary_cosine=std::clamp(.5f*(transfer*(energy+rb-transfer)+q2)/std::sqrt(p2*q2),-1.f,1.f);
        }else{out.primary_cosine=1.f;secondary_cosine=.5f;}
    }
    // Binding correction is exactly the native delta energy adjustment.
    // A prompt delta below cut is local; atomic fluorescence/Auger has its
    // own actual-cut conditional law and consumes relaxation_energy later.
    const float secondary=std::max(0.f,transfer-o.binding);
    out.relaxation_energy=std::min(o.binding,transfer);
    if(secondary>cut){out.secondary_energy=secondary;out.secondary_cosine=secondary_cosine;}
    else out.local=secondary;
    out.valid=std::isfinite(out.primary_cosine)&&std::isfinite(out.secondary_cosine)&&out.primary_energy>=0.f;
    return out;
}
} // namespace carbon::electron_composition26
