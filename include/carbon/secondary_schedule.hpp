#pragma once

#ifndef CARBON_SECONDARY_SEGMENT_STEPS
#define CARBON_SECONDARY_SEGMENT_STEPS 16
#endif

#ifndef CARBON_SECONDARY_HOT_SEGMENT_STEPS
#define CARBON_SECONDARY_HOT_SEGMENT_STEPS 16
#endif

namespace carbon {
// Scheduling only: suspend between complete physical iterations and retain RNG state.
inline constexpr unsigned kSecondarySegmentSteps = CARBON_SECONDARY_SEGMENT_STEPS;
static_assert(kSecondarySegmentSteps == 8 || kSecondarySegmentSteps == 16 || kSecondarySegmentSteps == 32 ||
              kSecondarySegmentSteps == 64 || kSecondarySegmentSteps == 128);
// Exact proton/deuteron slices may use a longer continuation budget without
// changing per-particle RNG identity or the suspended-step boundaries.
inline constexpr unsigned kSecondaryHotSegmentSteps = CARBON_SECONDARY_HOT_SEGMENT_STEPS;
static_assert(kSecondaryHotSegmentSteps == 8 || kSecondaryHotSegmentSteps == 16 ||
              kSecondaryHotSegmentSteps == 32 || kSecondaryHotSegmentSteps == 64 ||
              kSecondaryHotSegmentSteps == 128);
} // namespace carbon
