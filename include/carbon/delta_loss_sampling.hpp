#pragma once
#include <cmath>
#include <algorithm>
#include <limits>
namespace carbon {
// Moment-matched aggregate delta loss. Compound-Poisson cumulants are
// mean=h*lambda*E[accepted epsilon], variance=h*lambda*E[accepted epsilon^2].
// Gamma matches two moments; it does not preserve the zero-event atom or higher moments.
// Returned draw is uncapped; transport caps against remaining kinetic energy.
template<class Uniform> float delta_moment_draw(float mean,float variance,Uniform& uniform) {
    if(!(mean>=0 && std::isfinite(mean) && std::isfinite(variance)))return std::numeric_limits<float>::quiet_NaN();
    // A variance formed as a difference of moments can round to a tiny negative
    // under fp32 cancellation (backend/FMA dependent). Absorb only that noise;
    // any materially negative variance still signals a real bug.
    if(variance<0) {
        const float tolerance=1e-6f*(1.f+mean*mean);
        if(variance>-tolerance)variance=0.f;
        else return std::numeric_limits<float>::quiet_NaN();
    }
    if(mean==0 || variance==0)return mean;
    float shape=mean*(mean/variance),scale=variance/mean;
    if(!(shape>0 && std::isfinite(shape) && std::isfinite(scale)))return std::numeric_limits<float>::quiet_NaN();
    const bool small=shape<1;
    const float d=(small?shape+1:shape)-1.f/3.f,c=1/std::sqrt(9*d);
    auto u=[&](){float v=uniform();return (v>=0 && v<1)?std::max(v,1e-12f):std::numeric_limits<float>::quiet_NaN();};
    for(int tries=0;tries<256;++tries){
        const float x=std::sqrt(-2*std::log(u()))*std::cos(6.283185307179586f*u());
        float v=1+c*x;if(v<=0)continue;v=v*v*v;
        const float w=u();
        if(w<1-.0331f*x*x*x*x || std::log(w)<.5f*x*x+d*(1-v+std::log(v)))
            return scale*d*v*(small?std::pow(u(),1/shape):1.f);
    }
    return std::numeric_limits<float>::quiet_NaN();
}
}
