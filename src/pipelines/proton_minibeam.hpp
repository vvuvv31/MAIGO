#pragma once
#include "common.hpp"

namespace carbon::pipelines {
// Independent compile-time pipeline. Generic secondary projectile species remain supported.
struct ProtonMinibeam {
    static constexpr const char* name = "proton_minibeam";
    static constexpr bool broadbeam = false;
    static constexpr unsigned secondary_segment_steps = kSecondarySegmentSteps;
    static constexpr unsigned secondary_tail_threshold = 8192U;

    static constexpr std::size_t primary_workgroup(bool ct) {
        (void)ct; return 128U;
    }
    static constexpr std::size_t history_chunk(std::size_t histories) {
        return minibeam_history_chunk(histories);
    }
};
} // namespace carbon::pipelines
