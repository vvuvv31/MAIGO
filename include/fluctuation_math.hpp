#pragma once
#include <cmath>
#include <sycl/sycl.hpp>

namespace carbon {
// NativeMath is independent of native Gaussian evaluation: only the measured
// proton CT broadbeam and dedicated proton Cu primary enable these operations.
template<bool Native=false, class Real>
inline Real fluctuation_log(Real x) {
    if constexpr(Native) return sycl::native::log(x);
    else return std::log(x);
}
template<bool Native=false, class Real>
inline Real fluctuation_exp(Real x) {
    if constexpr(Native) return sycl::native::exp(x);
    else return std::exp(x);
}
template<bool Native=false, class Real>
inline Real fluctuation_sqrt(Real x) {
    if constexpr(Native) return sycl::native::sqrt(x);
    else return std::sqrt(x);
}
template<bool Native=false, class Real>
inline Real fluctuation_pow(Real x, Real y) {
    if constexpr(Native) return sycl::native::powr(x,y);
    else return std::pow(x,y);
}
} // namespace carbon
