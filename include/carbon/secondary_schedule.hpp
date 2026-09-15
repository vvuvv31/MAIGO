#pragma once

#ifndef CARBON_SECONDARY_SEGMENT_STEPS
#define CARBON_SECONDARY_SEGMENT_STEPS 16
#endif

namespace carbon {
// Scheduling only: suspend between complete physical iterations and retain RNG state.
inline constexpr unsigned kSecondarySegmentSteps = CARBON_SECONDARY_SEGMENT_STEPS;
static_assert(kSecondarySegmentSteps == 8 || kSecondarySegmentSteps == 16 || kSecondarySegmentSteps == 32 ||
              kSecondarySegmentSteps == 64 || kSecondarySegmentSteps == 128);
} // namespace carbon
