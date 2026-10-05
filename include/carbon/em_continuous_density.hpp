#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace carbon {
// One descriptor per composition/ion; no material-density response directory.
struct EmContinuousDescriptor {
    float log_min,log_max,base_density,base_cut,base_coefficient;
    float base_c,base_m,base_a,base_x0,base_x1,base_d0;
    float rate_max_energy;
    std::uint32_t range_offset,range_count,rate_offset,rate_count;
};
static_assert(sizeof(EmContinuousDescriptor)==64);
struct EmContinuousMaterial {
    float cut,coefficient,c,m,a,x0,x1,d0;
};
inline float em_density_effect(float x,float c,float m,float a,float x0,float x1,float d0) {
    return x>=x1?4.605170186f*x-c:x>=x0?4.605170186f*x-c+a*std::pow(std::max(0.f,x1-x),m):(d0==0.f?0.f:d0*std::pow(10.f,2*(x-x0)));
}
inline EmContinuousMaterial em_continuous_material(const float* table,unsigned count,float coordinate) {
    const float p=std::clamp(coordinate,0.f,1.f)*(count-1);
    const auto i=std::min(static_cast<unsigned>(p),count-2);const float w=p-i;
    EmContinuousMaterial out{};float* dst=&out.cut;
    for(unsigned k=0;k<8;++k)dst[k]=table[8*i+k]+w*(table[8*(i+1)+k]-table[8*i+k]);
    return out;
}
inline float em_chebyshev(const float* coeff,unsigned degree,float x) {
    float a=0,b=0;
    for(unsigned k=degree-1;k>0;--k){const float t=2*x*a-b+coeff[k];b=a;a=t;}
    return x*a-b+coeff[0];
}
} // namespace carbon
