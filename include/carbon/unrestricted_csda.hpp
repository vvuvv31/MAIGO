#pragma once
#include <cmath>

namespace carbon {
// Exact CSDA primitive for stopping linear in kinetic energy. This is the
// same interpolation used by the native full-response nodes, not a fitted
// range/stopping model. The small-slope series avoids cancellation in FP32.
template<class Real>
Real linear_stopping_path(Real width, Real stopping0, Real stopping1) {
    const Real q = (stopping1 - stopping0) / stopping0;
    const Real ratio = std::abs(q) < Real(.001)
        ? 1 + q * (Real(-.5) + q * (Real(1.0/3) - q * Real(.25)))
        : std::log1p(q) / q;
    return width / stopping0 * ratio;
}

template<class Real>
Real linear_stopping_energy(Real path, Real stopping0, Real slope) {
    const Real x = path * slope;
    const Real ratio = std::abs(x) < Real(.001)
        ? 1 + x * (Real(.5) + x * (Real(1.0/6) + x * Real(1.0/24)))
        : std::expm1(x) / x;
    return path * stopping0 * ratio;
}
} // namespace carbon
