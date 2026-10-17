#pragma once
#include "common.hpp"

namespace carbon::pipelines {
// One general pipeline family, with the existing broadbeam/reference physics modes.
// Its source remains restricted to proton/C12; secondary fragments remain generic.
template <bool BroadBeam> struct Generic {
    static constexpr const char *name = "generic";
    static constexpr bool broadbeam = BroadBeam;
    static constexpr unsigned secondary_segment_steps = kSecondarySegmentSteps;
    static constexpr unsigned secondary_tail_threshold = 8192U;

    static constexpr std::size_t primary_workgroup(bool ct) {
        return BroadBeam && ct ? 256U : 128U;
    }
    static constexpr std::size_t history_chunk(std::size_t histories) {
        return BroadBeam ? broadbeam_history_chunk(histories) : minibeam_history_chunk(histories);
    }
    template <class Launch>
    static auto launch_primary(bool ct_specialized, bool nonion, int source_z, int source_a,
                               Launch &&launch) {

        (void)ct_specialized;
        (void)source_z;
        (void)source_a;
        if (nonion)
            return launch(std::false_type{},
                          std::integral_constant<int, primary_stage::full_transport_nonion>{});
        return launch(std::false_type{},
                      std::integral_constant<int, primary_stage::full_transport>{});
    }
};
} // namespace carbon::pipelines
