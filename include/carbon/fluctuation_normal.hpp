#pragma once
#include <cmath>
#include <sycl/sycl.hpp>
namespace carbon {
// CUDA intrinsic arithmetic is scoped to the measured proton/carbon CT broadbeam
// primary loss path. Sampling/rejection laws stay intact; intrinsic results
// have finite precision and are checked against the standard implementation.
template<bool Native=false,class Real> Real fluctuation_normal(Real a,Real b) {
    if constexpr(Native)return sycl::native::sqrt(-2*sycl::native::log(a))*sycl::native::cos(Real(6.283185307179586)*b);
    else return std::sqrt(-2*std::log(a))*std::cos(Real(6.283185307179586)*b);
}
}
