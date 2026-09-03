#pragma once

#include "carbon/schneider_rate_table.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

struct SchneiderTargetSample {
    int target_z{0};
    std::size_t target_index{0};
    float target_probability{0.0F};
};

struct SchneiderTargetSamplerDeviceTable {
    const float* cdf_table{nullptr};
    const float* total_mass_rates{nullptr};
    std::size_t num_sections{25};
    std::size_t num_energies{860};
    std::size_t num_targets{13};
    float energy_min_MeV_per_u{0.5F};
    float energy_step_MeV_per_u{0.5F};
    float inverse_energy_step{2.0F};
};

struct SchneiderTargetDiagnostics {
    std::uint64_t interactions_by_section[25]{};
    std::uint64_t interactions_by_target_z[32]{}; // indexed by Z <= 31
    std::uint64_t interactions_by_energy_bin[450]{}; // 1 MeV/u bins up to 450
    std::uint64_t lookup_success_count{0};
    std::uint64_t lookup_failure_count{0};
    std::uint64_t replayed_events_count{0};
    std::uint64_t unsupported_target_count{0};
};

class SchneiderTargetSampler {
public:
    SchneiderTargetSampler() = default;
    explicit SchneiderTargetSampler(const SchneiderRateTable& rate_table);

    [[nodiscard]] SchneiderTargetSample sample_target(
        std::size_t section_id, float energy_MeV_per_u, float u01) const;

    [[nodiscard]] std::array<float, 13> target_probabilities(
        std::size_t section_id, float energy_MeV_per_u) const;

    [[nodiscard]] float total_mass_rate(std::size_t section_id, float energy_MeV_per_u) const;

    [[nodiscard]] const std::vector<float>& cdf_table() const noexcept { return cdf_table_; }
    [[nodiscard]] const std::vector<float>& total_mass_rates() const noexcept { return total_mass_rates_; }

    [[nodiscard]] SchneiderTargetSamplerDeviceTable device_table() const noexcept {
        return SchneiderTargetSamplerDeviceTable{
            cdf_table_.data(),
            total_mass_rates_.data(),
            num_sections_,
            num_energies_,
            num_targets_,
            static_cast<float>(energy_min_MeV_per_u_),
            static_cast<float>(energy_step_MeV_per_u_),
            static_cast<float>(1.0 / energy_step_MeV_per_u_)
        };
    }

private:
    std::size_t num_sections_{25};
    std::size_t num_energies_{860};
    std::size_t num_targets_{13};
    double energy_min_MeV_per_u_{0.5};
    double energy_step_MeV_per_u_{0.5};
    std::vector<float> cdf_table_;
    std::vector<float> total_mass_rates_;
};

// Device-safe target sampling helper
inline int sample_schneider_target_device(
    const SchneiderTargetSamplerDeviceTable& table,
    std::size_t section_id,
    float energy_MeV_per_u,
    float u01) noexcept
{
    if (section_id >= table.num_sections || table.cdf_table == nullptr) {
        return 0;
    }
    const float e_clamped = energy_MeV_per_u < table.energy_min_MeV_per_u
                                ? table.energy_min_MeV_per_u
                                : (energy_MeV_per_u > 430.0F ? 430.0F : energy_MeV_per_u);
    const float node_flt = (e_clamped - table.energy_min_MeV_per_u) * table.inverse_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(table.num_energies)) node_idx = static_cast<int>(table.num_energies - 1);

    const std::size_t base = section_id * (table.num_energies * table.num_targets) +
                             static_cast<std::size_t>(node_idx) * table.num_targets;
    const float u = u01 < 0.0F ? 0.0F : (u01 >= 1.0F ? 0.9999999F : u01);

    for (std::size_t k = 0; k < table.num_targets; ++k) {
        if (u < table.cdf_table[base + k]) {
            return kSchneiderCanonicalZ[k];
        }
    }
    return kSchneiderCanonicalZ[table.num_targets - 1];
}

} // namespace carbon
