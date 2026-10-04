#pragma once
#include "carbon/electron_charged_response.hpp"
namespace carbon {
// Interpolate mass stopping/range and mass cross sections between actual
// density nodes in one composition, then restore the actual physical density.
// Rates are mixed after native max(raw,0); conditional final states are then
// mixed with process-rate weights, not with an unconditional density coin.
struct ChargedDensityModel {
    const ChargedResponseView* lower{};const ChargedResponseView* upper{};double fraction{};
    int section{-1};double density{},electron_cut{},photon_cut{},lowest_energy{},linear_loss_limit{};
    ElectronUrbanMaterial urban{};ElectronElasticView elastic{};
};
// Node descriptions are immutable and outlive this borrowed model. Accept
// pointers explicitly so a temporary view cannot silently leave dangling refs.
inline ChargedDensityModel charged_density_model(const ChargedResponseView* lower,const ChargedResponseView* upper,double density) {
    ChargedDensityModel out;if(!lower||!upper)return out;
    const auto& low=*lower;const auto& high=*upper;
    if(low.section!=high.section||!std::isfinite(density)||density<low.density||density>high.density||low.density<=0||high.density<low.density)return out;
    out.lower=lower;out.upper=upper;out.fraction=low.density==high.density?0:(density-low.density)/(high.density-low.density);
    out.section=low.section;out.density=density;out.lowest_energy=low.lowest_energy;out.linear_loss_limit=low.linear_loss_limit;
    out.electron_cut=low.electron_cut+out.fraction*(high.electron_cut-low.electron_cut);
    out.photon_cut=low.photon_cut+out.fraction*(high.photon_cut-low.photon_cut);
    // All82 extracted Urban constants are identical within each composition.
    // Mott angular functions differ <1e-6 relatively across density nodes;
    // use the lower-node angle law as an explicit, separately audited choice.
    out.urban=low.urban;out.elastic=low.elastic;return out;
}
inline double charged_raw_rate(const ChargedDensityModel& m,unsigned process,double energy) {
    if(!m.lower||!m.upper||m.section<0)return NAN;
    const double low=charged_raw_rate(*m.lower,process,energy),high=charged_raw_rate(*m.upper,process,energy);
    if(!std::isfinite(low)||!std::isfinite(high))return NAN;
    return m.density*((1-m.fraction)*std::max(0.,low)/m.lower->density+m.fraction*std::max(0.,high)/m.upper->density);
}
inline ChargedParameters charged_parameters(const ChargedDensityModel& m,double energy) {
    if(!m.lower||!m.upper||m.section<0)return {};
    const auto a=charged_parameters(*m.lower,energy),b=charged_parameters(*m.upper,energy);if(!a.valid||!b.valid||m.section<0)return {};
    ChargedParameters out;const double f=m.fraction;
    const double low_scale=(1-f)*m.density/m.lower->density,high_scale=f*m.density/m.upper->density;
    out.elastic_rate=low_scale*a.elastic_rate+high_scale*b.elastic_rate;
    out.dedx=low_scale*a.dedx+high_scale*b.dedx;
    out.range=((1-f)*m.lower->density*a.range+f*m.upper->density*b.range)/m.density;
    for(unsigned p=0;p<2;++p)out.hard_rate[p]=low_scale*a.hard_rate[p]+high_scale*b.hard_rate[p];out.valid=true;return out;
}
inline double charged_inverse_range(const ChargedDensityModel& m,double range) {
    if(!m.lower||!m.upper||m.section<0||!std::isfinite(range)||m.lower->parameter_count<2||m.lower->parameter_count!=m.upper->parameter_count)return NAN;
    const auto* a=m.lower->parameters;const auto* b=m.upper->parameters;const double f=m.fraction;
    auto r=[&](unsigned i){return ((1-f)*m.lower->density*a[i].range+f*m.upper->density*b[i].range)/m.density;};
    unsigned lo=0,hi=m.lower->parameter_count-1;if(range<r(lo)||range>r(hi))return NAN;
    while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(r(mid)>range)hi=mid;else lo=mid;}
    if(a[lo].energy!=b[lo].energy||a[hi].energy!=b[hi].energy)return NAN;
    return a[lo].energy+(range-r(lo))/(r(hi)-r(lo))*(a[hi].energy-a[lo].energy);
}
inline ChargedCollision sample_charged_collision(const ChargedDensityModel& m,double energy,unsigned process,double u) {
    if(!m.lower||!m.upper||m.section<0||!std::isfinite(u)||u<0||u>=1)return {};
    const double lr=charged_raw_rate(*m.lower,process,energy),hr=charged_raw_rate(*m.upper,process,energy);
    if(!std::isfinite(lr)||!std::isfinite(hr))return {};
    const double a=(1-m.fraction)*std::max(0.,lr)/m.lower->density,b=m.fraction*std::max(0.,hr)/m.upper->density,total=a+b;if(total<=0)return {};
    const double cut=a/total;
    if(u<cut)return sample_charged_collision(*m.lower,energy,process,u/cut);
    if(b<=0)return {};
    // Rounding can produce exactly1 when u<1; move the draw (not KE) to the
    // greatest representable value below1, preserving inverse-CDF semantics.
    return sample_charged_collision(*m.upper,energy,process,std::min((u-cut)/(1-cut),std::nextafter(1.,0.)));
}
} // namespace carbon
