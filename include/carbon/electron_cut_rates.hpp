#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace carbon {
// Experimental scalar hard-rate function. Cut is a physical conditioning
// argument; these coefficients are not complete material-density states.
struct ElectronCutRateView {
    const float* energy_edges{};const float* cut_bounds{};const float* coefficients{};
    const float* density_slope{};unsigned intervals{},pieces{},coefficients_per_piece{};
    float density_low{},density_high{},mass_majorant{};
};
inline float electron_cut_rate(ElectronCutRateView v,float density,float cut,float energy){
    if(!v.energy_edges||!v.cut_bounds||!v.coefficients||!v.density_slope||!v.intervals||!v.pieces||!v.coefficients_per_piece||
       !std::isfinite(density)||density<v.density_low||density>v.density_high||!std::isfinite(cut)||cut<=0.f||
       !std::isfinite(energy)||energy<v.energy_edges[0]||energy>v.energy_edges[v.intervals])return NAN;
    const float logarithm=std::log(cut),lower=v.cut_bounds[0],upper=v.cut_bounds[v.pieces];
    // Roundoff in log(actual FP32 cut) may cross the fitted endpoint by one
    // representable log value. Only that numeric reserve is admitted.
    if(logarithm<std::nextafter(lower,-INFINITY)||logarithm>std::nextafter(upper,INFINITY))return NAN;
    unsigned p=0,q=v.pieces;
    while(p+1<q){const auto m=(p+q)/2;if(v.cut_bounds[m]>logarithm)q=m;else p=m;}
    const float width=v.cut_bounds[p+1]-v.cut_bounds[p];
    const float x=width>0.f?std::clamp(2.f*(logarithm-v.cut_bounds[p])/width-1.f,-1.f,1.f):0.f;
    unsigned lo=0,hi=v.intervals;
    while(lo+1<hi){const auto m=(lo+hi)/2;if(v.energy_edges[m]>energy)hi=m;else lo=m;}
    const float t=(energy-v.energy_edges[lo])/(v.energy_edges[lo+1]-v.energy_edges[lo]);
    constexpr float nodes[]={.125f,.375f,.625f,.875f};float result=0.f;
    const std::uint64_t stride=std::uint64_t(v.intervals)*4;
    for(unsigned field=0;field<4;++field){
        const auto* c=v.coefficients+std::uint64_t(p)*v.coefficients_per_piece*stride+lo*4+field;
        float a=0.f,b=0.f;
        for(unsigned k=v.coefficients_per_piece-1;k>0;--k){const float next=2.f*x*a-b+c[k*stride];b=a;a=next;}
        float value=x*a-b+c[0]+density*v.density_slope[lo*4+field];
        for(unsigned other=0;other<4;++other)if(other!=field)value*=(t-nodes[other])/(nodes[field]-nodes[other]);
        result+=value;
    }
    return density*std::max(0.f,result);
}
} // namespace carbon
