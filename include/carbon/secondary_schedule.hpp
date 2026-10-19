#pragma once

namespace carbon {
// Scheduling only: suspend between complete physical iterations and retain RNG state.
// The release interval is fixed; there is no experimental build-time selector.
inline constexpr unsigned kSecondarySegmentSteps = 16;

} // namespace carbon
