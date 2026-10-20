#pragma once

#include <string_view>

namespace carbon {

// Shared by compile-time kernel routing and host provenance. This policy is
// confined to patient-medium charged secondaries, never Cu/air or primaries.
constexpr bool uses_ct_minibeam_secondary_fe(std::string_view pipeline, bool ct) noexcept {
    return ct && (pipeline == "proton_minibeam" || pipeline == "carbon_minibeam");
}

} // namespace carbon
