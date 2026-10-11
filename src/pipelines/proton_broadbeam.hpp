#pragma once
#include "common.hpp"

namespace carbon::pipelines {
// Independent compile-time pipeline. Generic secondary projectile species remain supported.
struct ProtonBroadbeam {
    static constexpr const char* name = "proton_broadbeam";
    static constexpr int source_z = 1;
    static constexpr int source_a = 1;
    static constexpr bool broadbeam = true;
    static constexpr unsigned secondary_segment_steps = kSecondarySegmentSteps;
    static constexpr unsigned secondary_tail_threshold = 8192U;

    static constexpr std::size_t primary_workgroup(bool ct) {
        return ct ? 256U : 128U;
    }
    static constexpr std::size_t history_chunk(std::size_t histories) {
        return broadbeam_history_chunk(histories);
    }

    template<int JointResponse, class Launch>
    static auto launch_primary(bool ct_specialized, bool nonion, int source_z, int source_a, Launch&& launch) {
        (void)source_z; (void)source_a;
        if constexpr (JointResponse == 3) if (ct_specialized)
            return launch(std::false_type{}, std::integral_constant<int, 13>{});
        if constexpr (JointResponse == 0) if (ct_specialized)
            return launch(std::false_type{}, std::integral_constant<int, 15>{});
        if constexpr (JointResponse == 0) if (nonion)
            return launch(std::false_type{}, std::integral_constant<int, 7>{});
        return launch(std::false_type{}, std::integral_constant<int, 6>{});
    }
};
} // namespace carbon::pipelines
