#pragma once
#include <algorithm>
#include <cmath>
namespace carbon {
// For constant-rate Poisson cuts on a step h, x=lambda*h:
// E[sum(segment_length^2)]/h^2 = 2/x - 2*(1-exp(-x))/x^2.
// F is the fraction of the continuous self-drift correction already present
// in native piecewise step-start Euler loss. Delta jump drift is NOT scaled.
inline float delta_partition_fraction(float x){
    if(!(x>0))return 0;
    if(x<.1f)return x*(1.f/3.f+x*(-1.f/12.f+x*(1.f/60.f+x*(-1.f/360.f+x/2520.f))));
    return std::clamp(1-2/x-2*std::expm1(-x)/(x*x),0.f,1.f);
}
}
