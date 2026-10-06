#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace carbon {
// Experimental scalar component only. Hard-process majorants and correlated
// collision/birth kernels must be supplied before use in a transport worker.
// There is one descriptor per composition, with no density-node directory.
struct ElectronContinuousScalarView {
    const float* energy{};
    // [coefficient][energy][mass_mfp, mass_stopping, mass_range]
    const float* scalar{};
    // [uniform log-density piece][coefficient][electron_cut, photon_cut]
    const float* cuts{};
    unsigned energies{}, scalar_coefficients{}, cut_pieces{}, cut_coefficients{};
    float log_density_min{}, log_density_max{};
};
struct ElectronContinuousScalarResult {
    bool valid{};
    float elastic_rate{}, stopping{}, range{}, electron_cut{}, photon_cut{};
};

inline float electron_scalar_chebyshev(const float* coefficients,unsigned count,
                                       unsigned stride,float x) {
    float a=0.f,b=0.f;
    for(unsigned k=count-1;k>0;--k) {
        const float next=2.f*x*a-b+coefficients[k*stride];b=a;a=next;
    }
    return x*a-b+coefficients[0];
}
inline ElectronContinuousScalarResult electron_continuous_scalar(
    ElectronContinuousScalarView v,float density,float energy) {
    ElectronContinuousScalarResult out;
    if(!v.energy||!v.scalar||!v.cuts||v.energies<2||!v.scalar_coefficients||
       !v.cut_pieces||!v.cut_coefficients||!std::isfinite(density)||density<=0.f||
       !std::isfinite(energy)||energy<v.energy[0]||energy>v.energy[v.energies-1])return out;
    const float raw_logarithm=std::log(density);
    // CPU/device log can differ by one ulp at a physical domain endpoint.
    // Physical density bounds are separately enforced by the runtime model.
    const float low_guard=std::nextafter(std::nextafter(v.log_density_min,-INFINITY),-INFINITY);
    const float high_guard=std::nextafter(std::nextafter(v.log_density_max,INFINITY),INFINITY);
    if(raw_logarithm<low_guard||raw_logarithm>high_guard)return out;
    const float logarithm=std::clamp(raw_logarithm,v.log_density_min,v.log_density_max);
    const float coordinate=v.log_density_min==v.log_density_max?0.f:
        (logarithm-v.log_density_min)/(v.log_density_max-v.log_density_min);
    const float x=2.f*coordinate-1.f;
    unsigned lo=0,hi=v.energies-1;
    while(lo+1<hi){const unsigned middle=lo+(hi-lo)/2;if(v.energy[middle]>energy)hi=middle;else lo=middle;}
    const float fraction=(energy-v.energy[lo])/(v.energy[hi]-v.energy[lo]);
    float value[3];
    for(unsigned field=0;field<3;++field) {
        const float a=std::exp(electron_scalar_chebyshev(v.scalar+3*lo+field,v.scalar_coefficients,3*v.energies,x));
        const float b=std::exp(electron_scalar_chebyshev(v.scalar+3*hi+field,v.scalar_coefficients,3*v.energies,x));
        value[field]=a+fraction*(b-a);
    }
    const float piece_coordinate=coordinate*v.cut_pieces;
    const unsigned piece=std::min(unsigned(piece_coordinate),v.cut_pieces-1);
    const float cut_x=2.f*(piece_coordinate-piece)-1.f;
    const float* cuts=v.cuts+piece*v.cut_coefficients*2;
    out.electron_cut=std::exp(electron_scalar_chebyshev(cuts,v.cut_coefficients,2,cut_x));
    out.photon_cut=std::exp(electron_scalar_chebyshev(cuts+1,v.cut_coefficients,2,cut_x));
    out.elastic_rate=density/value[0];out.stopping=density*value[1];out.range=value[2]/density;
    out.valid=std::isfinite(out.elastic_rate)&&out.elastic_rate>0.f&&
        std::isfinite(out.stopping)&&out.stopping>0.f&&std::isfinite(out.range)&&out.range>0.f&&
        std::isfinite(out.electron_cut)&&out.electron_cut>0.f&&std::isfinite(out.photon_cut)&&out.photon_cut>0.f;
    return out;
}
} // namespace carbon
