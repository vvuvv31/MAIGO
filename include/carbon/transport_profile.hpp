#pragma once

// Optional step-path profiling for SYCL transport.
// Enable with CMake -DCARBON_ENABLE_TRANSPORT_PROFILE=ON (defines
// CARBON_TRANSPORT_PROFILE=1). Production dose builds leave this OFF so kernels
// do not pay for global atomics on every step.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace carbon {

// Ct face-clamp path tags written by clamp helpers when profiling is enabled.
enum class CtClampPath : std::uint32_t {
    three_axis = 0,
    short_step_skip = 1,
    homogeneous_skip = 2,
    near_z_face = 3,
    near_z_homogeneous = 4,
    // Count of distinct tags (not a path).
    count = 5,
};

// Device/host counter slots. Primary and secondary are separated so hotspots
// can be attributed without double-counting.
enum class TransportProfileSlot : std::size_t {
    primary_steps = 0,
    primary_ct_samples,
    primary_mass_sp_lookups,
    primary_sp_table_lookups,
    primary_face_clamp_calls,
    primary_face_three_axis,
    primary_face_short_step_skip,
    primary_face_homogeneous_skip,
    primary_face_near_z_face,
    primary_face_near_z_homogeneous,
    primary_dose_depth_atomics,
    primary_dose_voxel_atomics,
    primary_straggling,
    primary_mcs,
    primary_nuclear,
    primary_boundary_nudge_continues,

    secondary_steps,
    secondary_ct_samples,
    secondary_mass_sp_lookups,
    secondary_sp_table_lookups,
    secondary_face_clamp_calls,
    secondary_face_three_axis,
    secondary_face_short_step_skip,
    secondary_face_homogeneous_skip,
    secondary_face_near_z_face,
    secondary_face_near_z_homogeneous,
    secondary_dose_atomics,
    secondary_straggling,
    secondary_mcs,
    secondary_cascade,

    // Track step-count histogram for secondary tracks: bucket b holds tracks
    // whose step count is in [2^b, 2^(b+1)). Bucket 0 is [1, 2).
    secondary_track_hist_base,
    // 16 buckets cover up to ~65k steps; overflow lands in last bucket.
    // Keep this contiguous for device memset / copy.
    // slots: secondary_track_hist_base .. secondary_track_hist_base+15

    count = secondary_track_hist_base + 16,
};

inline constexpr std::size_t transport_profile_slot_count() noexcept {
    return static_cast<std::size_t>(TransportProfileSlot::count);
}

inline constexpr std::size_t secondary_track_hist_buckets() noexcept { return 16; }

struct TransportProfile {
    std::array<std::uint64_t, static_cast<std::size_t>(TransportProfileSlot::count)>
        counters{};
    bool enabled{false};

    [[nodiscard]] std::uint64_t get(TransportProfileSlot slot) const noexcept {
        return counters[static_cast<std::size_t>(slot)];
    }

    void set(TransportProfileSlot slot, std::uint64_t value) noexcept {
        counters[static_cast<std::size_t>(slot)] = value;
    }

    [[nodiscard]] std::string summary() const;
};

// Map CtClampPath to primary face counters.
inline TransportProfileSlot primary_face_slot(CtClampPath path) noexcept {
    switch (path) {
        case CtClampPath::three_axis:
            return TransportProfileSlot::primary_face_three_axis;
        case CtClampPath::short_step_skip:
            return TransportProfileSlot::primary_face_short_step_skip;
        case CtClampPath::homogeneous_skip:
            return TransportProfileSlot::primary_face_homogeneous_skip;
        case CtClampPath::near_z_face:
            return TransportProfileSlot::primary_face_near_z_face;
        case CtClampPath::near_z_homogeneous:
            return TransportProfileSlot::primary_face_near_z_homogeneous;
        default:
            return TransportProfileSlot::primary_face_three_axis;
    }
}

inline TransportProfileSlot secondary_face_slot(CtClampPath path) noexcept {
    switch (path) {
        case CtClampPath::three_axis:
            return TransportProfileSlot::secondary_face_three_axis;
        case CtClampPath::short_step_skip:
            return TransportProfileSlot::secondary_face_short_step_skip;
        case CtClampPath::homogeneous_skip:
            return TransportProfileSlot::secondary_face_homogeneous_skip;
        case CtClampPath::near_z_face:
            return TransportProfileSlot::secondary_face_near_z_face;
        case CtClampPath::near_z_homogeneous:
            return TransportProfileSlot::secondary_face_near_z_homogeneous;
        default:
            return TransportProfileSlot::secondary_face_three_axis;
    }
}

inline std::size_t secondary_track_hist_bucket(std::uint32_t steps) noexcept {
    if (steps == 0) {
        return 0;
    }
    std::size_t bucket = 0;
    std::uint32_t v = steps;
    while (v > 1U && bucket + 1U < secondary_track_hist_buckets()) {
        v >>= 1U;
        ++bucket;
    }
    return bucket;
}

}  // namespace carbon
