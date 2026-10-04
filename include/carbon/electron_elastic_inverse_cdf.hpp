#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace carbon::electron_inverse_cdf_research {
struct GridView {
    const float* quantiles{};
    const float* energies{};
    const float* log_energies{};
    const float* screening{};
    unsigned count{},bins{};
};
// Full-support single-elastic inverse CDF, at a certified energy grid point.
// Not connected to production transport; continuous-energy interpolation and
// CT validation are separate requirements. Every table has 48*bins+1 floats.
// Mirrored dyadic probability intervals resolve both tails without a search.
inline float sample(const float* quantiles, unsigned bins, float p) {
    if(!quantiles || !(p>=0 && p<1))return NAN;
    const bool upper=p>.5f;
    const float t=upper?1-p:p;
    unsigned interval;
    float fraction;
    if(t<0x1p-24f){
        const float x=t*(0x1p24f*bins);
        interval=static_cast<unsigned>(x);fraction=x-interval;
    }else{
        const auto bits=std::bit_cast<std::uint32_t>(t);
        const int exponent=int((bits>>23)&255)-127;
        const float mantissa=std::bit_cast<float>((bits&0x7fffffu)|(127u<<23));
        const float x=(mantissa-1)*bins;
        const unsigned offset=static_cast<unsigned>(x);
        interval=unsigned(exponent+25)*bins+offset;fraction=x-offset;
    }
    const unsigned last=48*bins;
    const unsigned i=upper?last-interval:interval;
    const unsigned j=upper?i-1:i+1;
    return quantiles[i]+fraction*(quantiles[j]-quantiles[i]);
}
// Interpolate the Rutherford probability coordinate, retaining its complement
// in the upper probability half. Forming 1-u after rounding u to float would
// lose the rare large-angle tail when screening is small. Each endpoint table
// represents its own full normalized Mott mixture at its own screening value.
inline float sample_interpolated(const float* low, const float* high, unsigned bins,
    float screening_low, float screening_high, float screening, float weight, float p) {
    if(!(screening_low>0&&screening_high>0&&screening>0&&weight>=0&&weight<=1))return NAN;
    const float q0=sample(low,bins,p);
    if(weight==0)return q0;
    const float q1=sample(high,bins,p);
    if(weight==1)return q1;
    if(p>.5f){
        const float v0=screening_low*(2-q0)/(2*screening_low+q0);
        const float v1=screening_high*(2-q1)/(2*screening_high+q1);
        const float v=v0+weight*(v1-v0);
        return 2*screening*(1-v)/(screening+v);
    }
    const float u0=q0*(1+screening_low)/(2*screening_low+q0);
    const float u1=q1*(1+screening_high)/(2*screening_high+q1);
    const float u=u0+weight*(u1-u0);
    return 2*screening*u/(1-u+screening);
}
inline bool covers(GridView grid,float energy) {
    return grid.quantiles&&grid.count>=2&&energy>=grid.energies[0]&&energy<=grid.energies[grid.count-1];
}
inline float sample_energy(GridView grid,float energy,float screening,float p) {
    if(!covers(grid,energy))return NAN;
    unsigned lo=0,hi=grid.count-1;
    while(hi-lo>1){const unsigned mid=(lo+hi)/2;if(energy<grid.energies[mid])hi=mid;else lo=mid;}
    const float weight=energy==grid.energies[lo]?0:energy==grid.energies[hi]?1:
        (std::log(energy)-grid.log_energies[lo])/(grid.log_energies[hi]-grid.log_energies[lo]);
    const unsigned knots=48*grid.bins+1;
    return sample_interpolated(grid.quantiles+std::size_t(lo)*knots,grid.quantiles+std::size_t(hi)*knots,
        grid.bins,grid.screening[lo],grid.screening[hi],screening,weight,p);
}
} // namespace carbon::electron_inverse_cdf_research
