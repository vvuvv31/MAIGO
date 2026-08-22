#pragma once

#include <cstdint>
#include <limits>

namespace carbon {

enum class CascadeSelectionPolicy : std::uint8_t {
    legacy_nearest,
    strict_coverage,
};

enum class CascadeSelectionStatus : std::uint8_t {
    exact_cell,
    expanded_window,
    nearest_fallback,
    no_energy_coverage,
};

struct CascadeSelectionResult {
    std::uint32_t interaction_index{std::numeric_limits<std::uint32_t>::max()};
    float source_energy_MeVu{};
    float energy_distance_MeVu{};
    CascadeSelectionStatus status{CascadeSelectionStatus::no_energy_coverage};
};

}  // namespace carbon
