#pragma once

#include <type_traits>

namespace carbon::detail {

// Disk-only EMCNT001/URBANV22 spline payload. Keep its six-float layout for
// package validation; unrestricted transport does not evaluate these splines.
struct NativeSplineSegment {
    float lower, upper, y0, ythird, ytwothirds, y1;
};
static_assert(sizeof(NativeSplineSegment) == 24);
static_assert(std::is_trivially_copyable_v<NativeSplineSegment>);

} // namespace carbon::detail
