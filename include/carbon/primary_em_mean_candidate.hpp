#pragma once
#include <algorithm>

namespace carbon {
// Research-only default-Geant4 table representation. Each native spline
// interval is a cubic polynomial; four equally spaced samples reconstruct it
// without changing knot locations or smoothing the inverse-range table.
template<class Real> struct EmCubicSegmentCandidate {
    Real lower, upper, y0, ythird, ytwothirds, y1;
    Real derivative(Real x) const {
        if(x<lower || x>upper)return 0;
        const Real w=(x-lower)/(upper-lower),a=w-Real(1.0/3),b=w-Real(2.0/3),c=w-1;
        return (-Real(4.5)*y0*(a*b+a*c+b*c)
            +Real(13.5)*ythird*(b*c+w*(b+c))
            -Real(13.5)*ytwothirds*(a*c+w*(a+c))
            +Real(4.5)*y1*(a*b+w*(a+b)))/(upper-lower);
    }
    Real value(Real x) const {
        const Real w=std::clamp((x-lower)/(upper-lower),Real(0),Real(1));
        const Real a=w-Real(1.0/3),b=w-Real(2.0/3),c=w-1;
        return y0*(-Real(4.5)*a*b*c)+ythird*(Real(13.5)*w*b*c)
             + ytwothirds*(-Real(13.5)*w*a*c)+y1*(Real(4.5)*w*a*b);
    }
};

} // namespace carbon
