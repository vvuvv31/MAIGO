#pragma once
#include "carbon/copper_proton_fluctuation.hpp"
#include "carbon/copper_ion_constants.hpp"
#include <algorithm>
#include <cmath>

namespace carbon {
// G4IonFluctuations::Factor specialized only to solid G4_Cu (Zeff=29).
// q2 is the active model's effective charge squared from the Urban package.
inline float copper_ion_dispersion_factor(float kinetic,float mass,int z,float q2,float beta2) {
    constexpr float zeff=29.f, electron_mass=.51099891f;
    const float fermi_beta2=2*copper_fermi_MeV/electron_mass;
    float s1=.4f*(1-beta2)/((1-.5f*beta2)*zeff);
    s1*=beta2>fermi_beta2
        ? std::log(2*electron_mass*beta2/copper_excitation_MeV)*fermi_beta2/beta2
        : std::log(4*copper_fermi_MeV/copper_excitation_MeV);
    s1+=1;
    float energy=kinetic*931.49410242f/mass;
    if(beta2<3*(.05f/938.272013f)*zeff) {
        const float ss=1-.4861f*std::pow(energy,-.6508f)+.8712f*std::pow(energy,-.8432f);
        if(ss<.001f)s1=1000;
        else if(s1*ss<1)s1=1/ss;
    }
    const float charge=static_cast<float>(z);
    const bool hadron=z==1;
    const float factor=hadron?1:charge*std::cbrt(charge/zeff);
    if(!hadron)energy/=charge*std::sqrt(charge*zeff);
    const float b0=hadron?.1955f:.01273f;
    const float b1=hadron?.6941f:.03458f;
    const float b2=hadron?2.522f:.3951f;
    const float b3=hadron?1.040f:3.812f;
    const float y=energy*b3;
    const float x=b2*(y<=.2f?y*(1-.5f*y):1-std::exp(-y));
    const float offset=energy-b1;
    return s1*q2/(charge*charge)+factor*x*b0/(offset*offset+x*x);
}

template<class Uniform>
RestrictedFluctuationDraw<float> copper_ion_total_fluctuation(
    float kinetic,float mass,int z,int a,float q2,float restricted_mean,
    float length,float density,Uniform& uniform,bool restricted_fluctuations) {
    if(!(kinetic>0&&mass>0&&z>0&&a>=z&&q2>0&&restricted_mean>=0&&length>=0) ||
       !std::isfinite(kinetic+mass+q2+restricted_mean+length+density) ||
       std::abs(density-8.96f)>1e-5f)return {};
    if(restricted_mean==0||length==0)return {0,true};
    const CopperIonIdentity* identity=nullptr;
    for(const auto& item:copper_ion_identities)if(item.z==z&&item.a==a){identity=&item;break;}
    if(!identity)return {};
    constexpr float electron_mass=.51099891f, radius=2.8179403262e-12f;
    const float tau=kinetic/mass,ratio=electron_mass/mass;
    const float beta2=tau*(tau+2)/((tau+1)*(tau+1));
    const float tmax=2*electron_mass*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);
    const float cut=std::min(tmax,copper_cut_MeV);
    const float coefficient=6.283185307179586f*electron_mass*radius*radius*
        copper_electron_density*length*(density/8.96f)*q2;
    // Disabling restricted fluctuations retains the mean and hard-delta
    // Poisson/spectrum sampling below; primary callers pass true explicitly.
    RestrictedFluctuationDraw<float> draw{restricted_mean,true};
    if(restricted_fluctuations) {
        const float universal_variance=(tmax/beta2-.5f*cut)*coefficient;
        const float factor=copper_ion_dispersion_factor(kinetic,mass,z,q2,beta2);
        const float cut_factor=1+(factor-1)*2*electron_mass*beta2/(tmax*(1-beta2));
        const float ion_variance=universal_variance*((cut_factor>.01f&&factor>.01f)?cut_factor:1);
        RestrictedFluctuationInput<float> input{kinetic,mass,restricted_mean,ion_variance,
            universal_variance,cut,tmax,copper_excitation_MeV,copper_e0_MeV};
        RestrictedFluctuationSampler<float,Uniform> sampler(uniform);
        draw=sampler.sample(input,!(z==1&&a==1),static_cast<float>(z));
    }
    if(!draw.valid||cut>=tmax)return draw;
    const float total2=(kinetic+mass)*(kinetic+mass);
    const float spin_term=identity->spin>0?.5f*(tmax-cut)/total2:0;
    float count=coefficient/beta2*(1/cut-1/tmax-beta2*std::log(tmax/cut)/tmax+spin_term);
    if(!(count>=0&&count<=4096&&std::isfinite(count)))return {};
    const float form_factor=2*electron_mass*std::pow(static_cast<float>(z),4.f/7.f)/(842.6f*842.6f);
    // Sum independent bounded Poissons; unlike the proton-only candidate,
    // high-charge ions can legitimately have more than eight delta proposals.
    unsigned proposals=0;
    while(count>0) {
        const float chunk=std::min(count,8.f),threshold=std::exp(-chunk);
        float product=1;unsigned n=0;
        do {
            float u=uniform();if(!(u>=0&&u<1)||++n>128)return {};
            product*=u;
        }while(product>threshold);
        proposals+=n-1;count-=chunk;
    }
    for(unsigned i=0;i<proposals;++i) {
        const auto delta=sample_charged_delta_candidate(cut,tmax,beta2,form_factor,mass,
            kinetic,identity->spin,identity->magnetic_moment2,uniform);
        if(delta.status==DeltaDrawStatus::form_factor_null)continue;
        if(delta.status!=DeltaDrawStatus::accepted)return {};
        draw.loss+=delta.energy_MeV;
    }
    return {draw.loss,std::isfinite(draw.loss)&&draw.loss>=0};
}
} // namespace carbon
