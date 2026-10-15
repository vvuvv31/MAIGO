#pragma once
#include "common.hpp"

namespace carbon::pipelines {
// Independent compile-time pipeline. Generic secondary projectile species remain supported.
struct CarbonBroadbeam {
    static constexpr const char* name = "carbon_broadbeam";
    static constexpr bool broadbeam = true;
    static constexpr unsigned secondary_segment_steps = kSecondarySegmentSteps;
    static constexpr unsigned secondary_tail_threshold = 8192U;

    static constexpr std::size_t primary_workgroup(bool ct) {
        return ct ? 256U : 128U;
    }
    static constexpr std::size_t history_chunk(std::size_t histories) {
        return broadbeam_history_chunk(histories);
    }

    template<class Launch>
    static auto launch_primary(bool ct_specialized, bool nonion, int source_z, int source_a, Launch&& launch) {
        (void)source_z; (void)source_a;
        if (ct_specialized)
            return launch(std::false_type{}, std::integral_constant<int, 17>{});
        (void)nonion;
        return launch(std::false_type{}, std::integral_constant<int, 6>{});
    }
};
} // namespace carbon::pipelines
