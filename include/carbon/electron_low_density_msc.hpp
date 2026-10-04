#pragma once
// Research component only: FULL Mott fixed-energy GS angular/spatial segment.
// No production dispatch, energy interpolation, loss/hard-event approximation
// or minibeam integration. Tables are for the actual low-density compositions.
#include "carbon/electron_multiple_cdf.hpp"
namespace carbon::electron_low_density_msc_research {
namespace f=electron_fp32;
namespace c=electron_condensed_research;
namespace m=electron_multiple_cdf_research;
// Borrow the six already cached material faces. This fast guard is permitted
// only when material and scoring grids coincide (checked once by the caller).
// It never reconstructs voxel indices or keeps a second scoring box.
inline float cached_safety(f::Position position,f::Position lower,f::Position upper,
                          float grid_coordinate_scale) {
    float clearance=INFINITY;
    for(unsigned k=0;k<3;++k)
        clearance=std::min(clearance,std::min(position[k]-lower[k],upper[k]-position[k]));
    const float eps=std::numeric_limits<float>::epsilon();
    return std::max(0.f,(clearance-8*eps*grid_coordinate_scale)/(1+64*eps));
}
inline bool eligible(float density,float mu,float first_moment,float length,float safety) {
    return density>0&&density<=.1f&&mu>=4&&mu<=64&&mu*first_moment<=.02f&&
           length>0&&length<=.25f&&length<=safety;
}
template<class Uniform>
inline c::Segment segment(const c::SoftLaw& full_mott,m::View table,float mu,float length,
                          c::V direction,Uniform& uniform) {
    // Scalar N<=4 positions and forced inline are build requirements for this
    // component: the earlier array/live-state implementation was rejected.
    return m::segment(full_mott,table,mu,length,direction,uniform);
}
} // namespace carbon::electron_low_density_msc_research
