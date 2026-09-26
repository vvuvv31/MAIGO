#pragma once

#include "carbon/particle.hpp"
#include <filesystem>

namespace carbon {

// Source parsing is independent of transport capability/data validation.
// Successfully describing a source does not authorize a physics model for it.
struct PrimarySourceParameters {
    PrimaryIonDefinition ion{};
    double initial_energy_MeVu{200.0};
    double beam_energy_spread{0.0}; // relative RMS, not percent
    double configured_rest_mass_MeV{0.0};

    [[nodiscard]] double initial_total_energy_MeV() const noexcept {
        return initial_energy_MeVu * ion.mass_number;
    }
};

[[nodiscard]] PrimarySourceParameters load_primary_source_parameters(
    const std::filesystem::path& config_path);

} // namespace carbon
