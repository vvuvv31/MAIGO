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

template<class Real> struct PrimaryRestrictedMeanCandidate {
    Real energy_loss{};
    bool range_branch{false};
    bool stopped{false};
};

// Callbacks use the SAME prestep dynamic mass/charge state for range inversion.
// correction(midpoint, basic_loss, step) returns the complete corrected loss,
// not an extra correction to already-corrected unrestricted stopping.
// Inputs and table domains must be validated by the caller before device use.
template<class Real,class InverseRange,class Correction>
PrimaryRestrictedMeanCandidate<Real> primary_restricted_mean_candidate(
    Real kinetic,Real step,Real prestep_restricted_stopping,Real range,
    InverseRange&& inverse_range,Correction&& correction,Real linear_limit=Real(.02)) {
    if(step<=0)return {};
    if(step>=range)return {kinetic,false,true};
    Real loss=prestep_restricted_stopping*step;
    bool use_range=loss>linear_limit*kinetic;
    if(use_range){Real de=kinetic-inverse_range(range-step);if(de>0)loss=de;}
    if(loss>=kinetic)return {kinetic,use_range,true};
    Real midpoint=std::max(kinetic-loss*Real(.5),kinetic*Real(.5));
    Real corrected=correction(midpoint,loss,step);
    if(corrected>kinetic || 2*corrected<loss)corrected=loss;
    corrected=std::max(Real(0),corrected);
    return {std::min(kinetic,corrected),use_range,corrected>=kinetic};
}
} // namespace carbon
