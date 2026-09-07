#pragma once

#include <cstdint>

namespace carbon {
// Reserved Philox dimension; distinct from event/energy selection (15/16).
inline constexpr std::uint32_t cinel03_event_azimuth_dimension = 60U;

struct Cinel03LocalDirection { float x, y, z; };

// Apply ONE common SO(2) rotation to every product of an unpolarized event.
// The caller draws the angle once, outside the product loop. Never rotate each
// product independently: that would destroy correlated-event kinematics.
inline Cinel03LocalDirection rotate_cinel03_event_azimuth(
    float x, float y, float z, float cosine, float sine) noexcept {
    return {cosine*x-sine*y, sine*x+cosine*y, z};
}
}  // namespace carbon
