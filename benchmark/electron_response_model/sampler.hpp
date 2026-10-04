#pragma once
#include <cmath>
#include <cstdint>

namespace electron_model_diagnostic {
// Portable diagnostic payload. No trajectories, ancestry or source files are
// accessed by sampling. Valid only for a homogeneous electron response and
// an explicitly selected measured energy window; not a boundary operator.
struct Cell {double cdf;float u0,u1,v0,v1;};
static_assert(sizeof(Cell)==24);
struct Draw {float x{},y{},z{};std::uint32_t cell{};bool valid{};};
inline Draw sample(const Cell* cells,std::uint32_t count,double choose,float ur,float uz,float uphi) {
    Draw result;
    if(!cells || !count || !(choose>=0 && choose<1) || !(ur>=0 && ur<1) ||
       !(uz>=0 && uz<1) || !(uphi>=0 && uphi<1))return result;
    std::uint32_t lo=0,hi=count-1;
    while(lo<hi) {const auto mid=lo+(hi-lo)/2;if(choose<cells[mid].cdf)hi=mid;else lo=mid+1;}
    const auto c=cells[lo];
    const float r=0.01f*std::sinh(c.u0+ur*(c.u1-c.u0));
    const float phi=6.2831853071795864769f*uphi;
    result.x=r*std::cos(phi);result.y=r*std::sin(phi);
    result.z=0.01f*std::sinh(c.v0+uz*(c.v1-c.v0));
    result.cell=lo;result.valid=true;return result;
}
inline std::uint32_t mix(std::uint32_t x) {
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;
}
inline float uniform(std::uint32_t x) {
    // 23-bit midpoint stays strictly below 1 even after FP32 rounding.
    return (static_cast<float>(x>>9)+0.5f)*0x1p-23f;
}
} // namespace electron_model_diagnostic
