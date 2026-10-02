#pragma once
#include "carbon/electron_ct_boundary.hpp"

namespace carbon {
// Sample uniformly on the geometric chord inside the CT. A primary FP32
// face-distance division can round upward; do not extend the electron birth
// domain past the outer CT face. Internal material interfaces are untouched.
// Only an overshoot bounded by one FP32 length ULP is repaired. Larger
// geometry errors and starts outside the CT remain hard lookup failures.
inline std::array<double,3> electron_ct_birth_position(
    const ElectronCtGeometry& g, const std::array<double,3>& start,
    const std::array<double,3>& direction, float step, double u) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const std::array<double,3> invalid{nan,nan,nan};
    if(!std::isfinite(step) || step<=0 || !std::isfinite(u) || u<0 || u>=1)
        return invalid;
    double length=step;
    for(unsigned k=0;k<3;++k) {
        const double lo=g.origin[k],hi=lo+g.bins[k]*g.spacing[k];
        const double p=start[k],d=direction[k];
        if(!g.bins[k] || !std::isfinite(lo) || !std::isfinite(hi) ||
           !(g.spacing[k]>0) || !std::isfinite(p) || !std::isfinite(d) ||
           p<lo || p>hi || (p==lo && d<0) || (p==hi && d>=0))return invalid;
        const double end=p+double(step)*d;
        if((d>0 && end>hi) || (d<0 && end<lo)) {
            const double exit=((d>0?hi:lo)-p)/d;
            const float ceiling=std::nextafter(static_cast<float>(exit),
                                               std::numeric_limits<float>::infinity());
            if(!(exit>0) || double(step)>double(ceiling))return invalid;
            if(exit<length)length=exit;
        }
    }
    std::array<double,3> out;
    for(unsigned k=0;k<3;++k) {
        out[k]=start[k]+u*length*direction[k];
        const double lo=g.origin[k],hi=lo+g.bins[k]*g.spacing[k];
        // u<1 is mathematically interior at an exit face. If the final double
        // addition rounds to that face, choose its adjacent interior double.
        // This is outer-box representability, not a material-interface epsilon.
        if(direction[k]>0 && start[k]<hi && out[k]==hi)
            out[k]=std::nextafter(hi,start[k]);
        // Subtraction/division used by directed ownership may also round an
        // interior coordinate to bins. Keep the represented point interior in
        // both coordinate systems; never move an internal-interface point.
        for(unsigned j=0;j<4 && direction[k]>0 && start[k]<hi &&
            out[k]<=hi && (out[k]-lo)/g.spacing[k]>=g.bins[k];++j)
            out[k]=std::nextafter(out[k],start[k]);
        if(direction[k]<0 && start[k]>lo && out[k]==lo)
            out[k]=std::nextafter(lo,start[k]);
    }
    return out;
}
} // namespace carbon
