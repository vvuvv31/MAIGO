#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace carbon::electron_composition26 {
// Experimental actual-cut function component. Each composition/process owns
// one descriptor; pieces partition actual FP32 cut, never density states.
struct NativeGridRateView {
    const float* bounds{};const float* coefficients{};
    const std::uint32_t* bins{};const std::uint32_t* intervals{};
    const std::uint32_t* offsets{};
    const std::uint32_t* pole{};
    const float* piece_majorants{};
    unsigned pieces{},degree{},process{};
    float cut_low{},cut_high{},mass_majorant{};
};
inline float native_grid_mass_rate(NativeGridRateView v,float cut,float energy) {
    if(!v.bounds||!v.coefficients||!v.bins||!v.intervals||!v.offsets||!v.pole||!v.pieces||v.process>1||v.degree>32||
       !std::isfinite(cut)||!std::isfinite(energy)||cut<v.cut_low||cut>v.cut_high||!(energy>0.f)||energy>.65f)return NAN;
    const float minimum=(v.process==0?2.f:1.f)*cut;
    if(energy<=minimum)return 0.f;
    unsigned low=0,high=v.pieces;
    while(low+1<high){const unsigned mid=low+(high-low)/2;if(v.bounds[mid]>cut)high=mid;else low=mid;}
    const unsigned piece=low,bins=v.bins[piece],intervals=v.intervals[piece];
    if(!bins||!intervals)return NAN;
    const float width=v.bounds[piece+1]-v.bounds[piece];
    const float x=width>0.f?std::clamp(2.f*(cut-v.bounds[piece])/width-1.f,-1.f,1.f):0.f;
    const float logstep=std::log(1e4f/minimum)/bins;
    const unsigned index=unsigned(std::max(0.f,std::log(energy/minimum)/logstep));
    if(index>=intervals)return NAN;
    const float a=minimum*std::exp(index*logstep),b=minimum*std::exp((index+1)*logstep);
    const float t=std::clamp((energy-a)/(b-a),0.f,1.f);
    const float* c=v.coefficients+v.offsets[piece]+index*4;
    float values[4];
    for(unsigned j=0;j<4;++j){float p=0.f,q=0.f;
        for(unsigned k=v.degree;k>0;--k){const float next=2.f*x*p-q+c[k*intervals*4+j];q=p;p=next;}
        values[j]=x*p-q+c[j];
    }
    constexpr float nodes[]={.125f,.375f,.625f,.875f};float rate=0.f;
    for(unsigned j=0;j<4;++j){float term=values[j];
        for(unsigned k=0;k<4;++k)if(k!=j)term*=(t-nodes[k])/(nodes[j]-nodes[k]);rate+=term;}
    if(v.pole[piece]){
        // Correct the rounding of the decimal.05 FP32 constant before
        // subtracting: near the pole that one-ulp bias is material.
        const float distance=(.05f-cut)-7.450580596923828e-10f;
        if(!(distance>0.f))return NAN;rate/=distance;
    }
    return std::max(0.f,rate);
}
inline float native_grid_rate(NativeGridRateView v,float density,float cut,float energy) {
    if(!(density>0.f)||!std::isfinite(density))return NAN;
    return density*native_grid_mass_rate(v,cut,energy);
}
inline float native_grid_cut_majorant(NativeGridRateView v,float density,float cut) {
    if(!v.bounds||!v.piece_majorants||!v.pole||!v.pieces||cut<v.cut_low||cut>v.cut_high||!(density>0.f))return NAN;
    unsigned low=0,high=v.pieces;
    while(low+1<high){const auto mid=low+(high-low)/2;if(v.bounds[mid]>cut)high=mid;else low=mid;}
    float value=v.piece_majorants[low];
    if(v.pole[low]){const float distance=(.05f-cut)-7.450580596923828e-10f;
        if(!(distance>0.f))return NAN;value/=distance;}
    return density*value;
}
} // namespace carbon::electron_composition26
