#pragma once

#include "carbon/inelastic.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

struct FredEventLibrary {
    float reference_energy_MeVu{95.0F};
    std::uint16_t target_z{1};
    std::uint16_t target_a{1};
    std::uint32_t event_count{0};
    std::uint32_t max_fragments{8};
    std::vector<std::uint8_t> fragment_count;
    std::vector<float> neutron_ke_MeV;
    std::vector<std::int8_t> z;
    std::vector<std::int8_t> a;
    std::vector<float> ke_MeV;
    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
};

FredEventLibrary load_fred_event_library(const std::filesystem::path& path);

}  // namespace carbon
