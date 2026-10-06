#pragma once
#include "carbon/electron_penelope_composition.hpp"
#include <array>
namespace carbon::electron_composition26 {
// Experimental actual-cut native lambda preparation. These transient tables
// are derived from a continuous composition law, not stored density nodes.
// Normalize XS by2.5493841e-23mm^2 MeV so GPU FTZ cannot erase log floors.
// Preparation is intended for a voxel/material refresh, never every step.
struct PenelopeLambda26 {
    bool valid{};unsigned count{};
    std::array<float,192> energy{},value{},second{};
};
inline PenelopeLambda26 prepare_penelope_lambda26(PenelopeComposition v,float cut) {
    PenelopeLambda26 out;
    if(!v.oscillators||!v.count||v.count>128||!v.delta||v.delta_count!=34||!(cut>=.00099f&&cut<.325f))return out;
    float logs[34],total_z=0.f;for(unsigned j=0;j<v.count;++j)total_z+=v.oscillators[j].strength;
    constexpr float scaled_floor=3.922515e-18f;
    for(unsigned i=0;i<34;++i){
        const float e=penelope_grid_energy(i),beta=e*(e+1.0219979f)/((e+.51099895f)*(e+.51099895f));float sum=0.f;
        for(unsigned j=0;j<v.count;++j){const auto o=v.oscillators[j];if(e<o.ionization)continue;
            const auto h=penelope_hard_channels(o,e,cut,v.delta[i],false);sum+=o.strength*(h.close+h.longitudinal+h.transverse);}
        logs[i]=std::log(std::max(scaled_floor,sum/beta));
    }
    auto penelope=[&](float e){
        const float position=std::clamp((std::log(e)-std::log(1e-4f))*(99.f/std::log(1e9f)),0.f,33.f);
        const unsigned i=std::min(unsigned(position),32u);const float t=position-i;
        return std::exp(logs[i]+t*(logs[i+1]-logs[i]));
    };
    auto moller=[&](float e){
        const float beta=e*(e+1.0219979f)/((e+.51099895f)*(e+.51099895f));
        PenelopeOscillator o{total_z,0.f,1e-10f,1e-10f,0.f,0,30};
        return total_z*penelope_hard_channels(o,e,cut,0.f,false).close/beta;
    };
    const float minimum=2.f*cut,logspan=std::log(1e4f/minimum);
    const unsigned bins=unsigned(std::max(5.f,std::floor(20.f*logspan/std::log(10.f)+.5f)));
    if(bins+1>out.energy.size())return out;out.count=bins+1;
    const float m=moller(.1f),correction=m>0.f?(penelope(.1f)/m-1.f)*.1f:0.f;
    for(unsigned i=0;i<=bins;++i){const float e=i==bins?1e4f:minimum*std::exp(i*logspan/bins);
        out.energy[i]=e;out.value[i]=i==0?0.f:e<=.1f?penelope(e):moller(e)*(1.f+correction/e);
        if(!std::isfinite(out.value[i])||out.value[i]<0.f)return {};}
    // Exact G4PhysicsVector::ComputeSecDerivative1 (Kvasov recurrence).
    const auto& x=out.energy;const auto& y=out.value;auto& sec=out.second;float u[192]{};const unsigned n=bins;
    u[1]=(y[2]-y[1])/(x[2]-x[1])-(y[1]-y[0])/(x[1]-x[0]);
    u[1]*=6.f*(x[2]-x[1])/((x[2]-x[0])*(x[2]-x[0]));
    sec[1]=(2.f*x[1]-x[0]-x[2])/(2.f*x[2]-x[0]-x[1]);
    for(unsigned i=2;i<n-1;++i){const float sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]),p=sig*sec[i-1]+2.f;
        sec[i]=(sig-1.f)/p;u[i]=(y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
        u[i]=6.f*u[i]/(x[i+1]-x[i-1])-sig*u[i-1]/p;}
    float sig=(x[n-1]-x[n-2])/(x[n]-x[n-2]),p=sig*sec[n-3]+2.f;
    u[n-1]=(y[n]-y[n-1])/(x[n]-x[n-1])-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]);
    u[n-1]=6.f*sig*u[n-1]/(x[n]-x[n-2])-(2.f*sig-1.f)*u[n-2]/p;
    p=1.f+sig+(2.f*sig-1.f)*sec[n-2];sec[n-1]=u[n-1]/p;
    for(unsigned k=n-2;k>1;--k)sec[k]*=sec[k+1]-u[k]*(x[k+1]-x[k-1])/(x[k+1]-x[k]);
    sec[n]=(sec[n-1]-(1.f-sig)*sec[n-2])/sig;
    sig=1.f-(x[2]-x[1])/(x[2]-x[0]);sec[1]*=sec[2]-u[1]/(1.f-sig);sec[0]=(sec[1]-sig*sec[2])/(1.f-sig);
    for(unsigned i=0;i<=n;++i)if(!std::isfinite(sec[i]))return {};
    out.valid=true;return out;
}
inline float penelope_lambda26(PenelopeLambda26 const& v,float energy) {
    if(!v.valid||!std::isfinite(energy)||energy>.65f||!(energy>0.f))return NAN;
    if(energy<=v.energy[0])return 0.f;
    unsigned low=0,high=v.count-1;while(low+1<high){const auto m=low+(high-low)/2;if(v.energy[m]>energy)high=m;else low=m;}
    const float width=v.energy[high]-v.energy[low],b=(energy-v.energy[low])/width;
    return std::max(0.f,v.value[low]+b*(v.value[high]-v.value[low])+b*(b-1.f)*((2.f-b)*v.second[low]+(1.f+b)*v.second[high])*width*width/6.f);
}
}
